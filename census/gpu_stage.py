"""GPU stage (torch + circuit-tracer). Writes the run directory described in
census/runio.py. Written against circuit-tracer 0.5 (TransformerLens backend):

  model.transcoders[layer]          SingleLayerTranscoder; W_enc, W_dec are (d_tc, d_model)
  model.transcoders.encode_layer    JumpReLU/ReLU activations for one layer
  blocks.{l}.{model.feature_input_hook}            where transcoders read
  blocks.{l}.{model.original_feature_output_hook}  where the MLP output is written

Per-layer transcoders only for now; see README for the CLT extension.
"""

import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch

from .data import load_or_build_tokens

STRATA = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]


def load_model(cfg):
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder import TranscoderSet
    model = ReplacementModel.from_pretrained(cfg.model_name, cfg.transcoder_set,
                                             dtype=torch.bfloat16)
    if not isinstance(model.transcoders, TranscoderSet):
        raise NotImplementedError("CLT support: see README ('Extending to CLTs').")
    return model


def _unembed(model):
    """(d_model, V) unembedding. `unembed_proj` exists only in the PyPI build of
    circuit-tracer; upstream (decoderesearch) exposes it as `unembed.W_U`."""
    W_U = getattr(model, "unembed_proj", None)
    return model.unembed.W_U if W_U is None else W_U


def _batches(seqs, bs, device):
    for a in range(0, len(seqs), bs):
        yield a, torch.as_tensor(seqs[a:a + bs], device=device)


def _in_hook(l, model):
    return f"blocks.{l}.{model.feature_input_hook}"


def frequency_pass(model, seqs, layers, cfg):
    d_tc = model.transcoders.d_transcoder
    dev = model.cfg.device
    counts = {l: torch.zeros(d_tc, device=dev) for l in layers}

    def hook(acts, hook, layer):
        z = model.transcoders.encode_layer(acts, layer)
        z[:, 0] = 0                                    # ignore BOS, as circuit-tracer does
        counts[layer] += (z > 0).sum((0, 1)).float()

    hooks = [(_in_hook(l, model), partial(hook, layer=l)) for l in layers]
    sub = seqs[: cfg.freq_pass_seqs]
    with torch.inference_mode(), model.hooks(hooks):
        for _, b in _batches(sub, cfg.batch_size, dev):
            model(b, stop_at_layer=max(layers) + 1)
    n_tok = len(sub) * (seqs.shape[1] - 1)
    return {l: (counts[l] / n_tok).cpu().numpy() for l in layers}


def sample_latents(densities, cfg):
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for layer, dens in densities.items():
        strata = np.digitize(dens, STRATA[1:-1], right=False)   # 0 == density < 1e-6
        sizes = np.bincount(strata, minlength=len(STRATA) - 1)
        alloc = np.maximum(np.round(cfg.latents_per_layer * sizes / sizes.sum()),
                           np.minimum(sizes, cfg.min_per_stratum)).astype(int)
        for s, k in enumerate(alloc):
            pool = np.flatnonzero(strata == s)
            k = min(k, len(pool))
            if k == 0:
                continue
            for f in rng.choice(pool, size=k, replace=False):
                rows.append({"layer": layer, "feature": int(f), "density_est": float(dens[f]),
                             "stratum": s, "weight": sizes[s] / k})
    df = pd.DataFrame(rows).sort_values(["layer", "feature"]).reset_index(drop=True)
    df.insert(0, "uid", np.arange(len(df)))
    return df


def cache_pass(model, seqs, latents, cfg, out_dir):
    dev, L = model.cfg.device, seqs.shape[1]
    layers = sorted(latents["layer"].unique())
    sel = {l: torch.as_tensor(g["feature"].values, device=dev) for l, g in latents.groupby("layer")}
    uids = {l: torch.as_tensor(g["uid"].values, device=dev) for l, g in latents.groupby("layer")}
    recs = {"uid": [], "tok_idx": [], "val": []}
    d = model.cfg.d_model
    cov = {l: [torch.zeros(d, device=dev, dtype=torch.float64),
               torch.zeros(d, d, device=dev, dtype=torch.float64), 0] for l in layers}
    state = {"offset": 0}

    def tc_hook(acts, hook, layer):
        z = model.transcoders.encode_layer(acts, layer)[..., sel[layer]]
        z[:, 0] = 0
        b, p, j = z.nonzero(as_tuple=True)
        recs["uid"].append(uids[layer][j].cpu())
        recs["tok_idx"].append(((state["offset"] + b) * L + p).cpu())
        recs["val"].append(z[b, p, j].float().cpu())

    def resid_hook(acts, hook, layer):
        c = cov[layer]
        if c[2] >= cfg.resid_cov_tokens:
            return
        x = acts[:, 1:].reshape(-1, d).double()
        c[0] += x.sum(0)
        c[1] += x.T @ x
        c[2] += x.shape[0]

    hooks = [(_in_hook(l, model), partial(tc_hook, layer=l)) for l in layers]
    hooks += [(f"blocks.{l}.hook_resid_post", partial(resid_hook, layer=l)) for l in layers]
    with torch.inference_mode(), model.hooks(hooks):
        for a, b in _batches(seqs, cfg.batch_size, dev):
            state["offset"] = a
            model(b, stop_at_layer=max(layers) + 1)

    np.savez(os.path.join(out_dir, "records.npz"),
             uid=torch.cat(recs["uid"]).numpy().astype(np.int32),
             tok_idx=torch.cat(recs["tok_idx"]).numpy().astype(np.int32),
             val=torch.cat(recs["val"]).numpy().astype(np.float32))
    os.makedirs(os.path.join(out_dir, "resid"), exist_ok=True)
    for l, (s1, s2, n) in cov.items():
        mean = s1 / n
        np.savez(os.path.join(out_dir, "resid", f"L{l}.npz"), mean=mean.cpu().numpy(),
                 cov=(s2 / n - torch.outer(mean, mean)).cpu().numpy())


def export_weights(model, latents, cfg, out_dir):
    n, d = len(latents), model.cfg.d_model
    enc, dec = np.zeros((n, d), np.float16), np.zeros((n, d), np.float16)
    b_enc, thr = np.zeros(n, np.float32), np.full(n, np.nan, np.float32)
    for l, g in latents.groupby("layer"):
        tc = model.transcoders[l]
        idx = torch.as_tensor(g["feature"].values, device=tc.b_enc.device)
        enc[g.index] = tc.W_enc[idx].float().cpu().numpy()
        dec[g.index] = tc._get_decoder_vectors(idx.cpu()).float().cpu().numpy()
        b_enc[g.index] = tc.b_enc[idx].float().cpu().numpy()
        t = getattr(tc.activation_function, "threshold", None)
        if t is not None:
            t = t.detach().float()
            thr[g.index] = (t[idx] if t.ndim else t.expand(len(idx))).cpu().numpy()
    np.savez(os.path.join(out_dir, "weights.npz"), enc=enc, dec=dec, b_enc=b_enc, threshold=thr)

    os.makedirs(os.path.join(out_dir, "downstream"), exist_ok=True)
    n_layers = model.cfg.n_layers
    needed = {L2 for l in latents["layer"].unique()
              for L2 in range(l + 1, min(n_layers, l + 1 + cfg.downstream_layers))}
    for L2 in sorted(needed):
        W = model.transcoders[L2].W_enc.detach().to(torch.float16).cpu().numpy()
        np.save(os.path.join(out_dir, "downstream", f"enc_L{L2}.npy"), W)

    # effective unembedding: fold the final norm's weight in (circuit-tracer loads fold_ln=False)
    W_U = _unembed(model).detach().float()                     # (d, V)
    w = getattr(model.ln_final, "w", None)
    if w is not None:
        W_U = W_U * w.detach().float()[:, None]
    V = W_U.shape[1]
    mm = np.lib.format.open_memmap(os.path.join(out_dir, "unembed.npy"), mode="w+",
                                   dtype=np.float16, shape=(V, d))
    for a in range(0, V, 32768):
        mm[a:a + 32768] = W_U[:, a:a + 32768].T.cpu().numpy()
    mm.flush()


def relevance_pass(model, seqs, latents, cfg, out_dir):
    """Zero-ablate one latent's contribution and measure next-token KL."""
    r = np.load(os.path.join(out_dir, "records.npz"))
    r_uid, r_tok, r_val = r["uid"], r["tok_idx"], r["val"]           # load once
    order_all = np.argsort(r_uid, kind="stable")
    r_uid, r_tok, r_val = r_uid[order_all], r_tok[order_all], r_val[order_all]
    dev, L = model.cfg.device, seqs.shape[1]
    out_hook = lambda l: f"blocks.{l}.{model.original_feature_output_hook}"
    rows = []
    for _, lat in latents.iterrows():
        a, b = np.searchsorted(r_uid, lat["uid"]), np.searchsorted(r_uid, lat["uid"], "right")
        if a == b:
            continue
        toks, vals = r_tok[a:b], r_val[a:b]
        order = np.argsort(-vals)
        seq_ids, fire_pos = [], []
        for t in toks[order]:
            s, p = divmod(int(t), L)
            if s not in seq_ids:
                seq_ids.append(s); fire_pos.append(p)
            if len(seq_ids) == cfg.relevance_contexts:
                break
        batch = torch.as_tensor(seqs[seq_ids], device=dev)
        l, f = int(lat["layer"]), int(lat["feature"])
        tc = model.transcoders[l]
        dvec = tc._get_decoder_vectors(torch.tensor([f]))[0].to(dev)
        stash = {}

        def read(acts, hook):
            z = model.transcoders.encode_layer(acts, l)[..., f]
            z[:, 0] = 0                                  # never ablate at BOS
            stash["z"] = z

        def ablate(acts, hook):
            return acts - stash["z"][..., None].to(acts.dtype) * dvec.to(acts.dtype)

        with torch.inference_mode():
            clean = model(batch).float().log_softmax(-1)
            with model.hooks([(_in_hook(l, model), read), (out_hook(l), ablate)]):
                abl = model(batch).float().log_softmax(-1)
            kl = (clean.exp() * (clean - abl)).sum(-1).cpu().numpy()      # (K, L)
        at_fire = [kl[i, p] for i, p in enumerate(fire_pos)]
        after = [kl[i, p:].mean() for i, p in enumerate(fire_pos)]
        rows.append({"uid": lat["uid"], "kl_at_fire": float(np.mean(at_fire)),
                     "kl_after": float(np.mean(after)), "n_contexts": len(seq_ids)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "relevance.csv"), index=False)


def _safe_decode(tok, i):
    try:
        return tok.decode([i])
    except Exception:              # unembedding can be wider than the tokenizer
        return f"<id{i}>"


def run_gpu_stage(cfg, out_dir, skip_relevance=False, prep_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if prep_only:                  # CPU only: needs the tokenizer, not the model
        from transformers import AutoTokenizer
        load_or_build_tokens(cfg, out_dir, AutoTokenizer.from_pretrained(cfg.model_name))
        return
    model = load_model(cfg)
    tok = model.tokenizer
    arrays, pos_tags, domains = load_or_build_tokens(cfg, out_dir, tok)
    V = _unembed(model).shape[1]
    with open(os.path.join(out_dir, "vocab.json"), "w") as fh:
        json.dump([_safe_decode(tok, i) for i in range(V)], fh)
    seqs = arrays["tok"]

    dens = frequency_pass(model, seqs, cfg.layers, cfg)
    latents = sample_latents(dens, cfg)
    latents.to_csv(os.path.join(out_dir, "latents.csv"), index=False)
    meta = {"model_name": cfg.model_name, "transcoder_set": cfg.transcoder_set,
            "layers": cfg.layers, "seq_len": cfg.seq_len, "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model, "d_transcoder": model.transcoders.d_transcoder,
            "vocab_size": V, "pos_tags": pos_tags, "domains": domains}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    cache_pass(model, seqs, latents, cfg, out_dir)
    export_weights(model, latents, cfg, out_dir)
    if not skip_relevance:
        relevance_pass(model, seqs, latents, cfg, out_dir)
