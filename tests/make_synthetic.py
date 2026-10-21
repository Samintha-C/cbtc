"""Write a synthetic run directory (same schema as the GPU stage) with planted
latents whose correct tags are known, so the offline stage can be tested
without a GPU or model weights."""

import json
import os

import numpy as np
import pandas as pd

POS = ["BOS", "SPACE", "X", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
       "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]
PI = {t: i for i, t in enumerate(POS)}


def make_run(out, S=600, L=64, V=600, d=256, d_tc=300, n_layers=12, k_null=12, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out, "downstream"), exist_ok=True)
    os.makedirs(os.path.join(out, "resid"), exist_ok=True)

    # ---- tokens and metadata ----------------------------------------------
    BOS, NL, DOT = 0, 1, 2
    p = 1.0 / (np.arange(1, V) + 5.0) ** 1.1
    tok = np.zeros((S, L), np.int32)
    tok[:, 1:] = rng.choice(np.arange(1, V), size=(S, L - 1), p=p / p.sum())
    tok_pos = rng.choice([PI[t] for t in ["VERB", "ADJ", "DET", "ADP", "PRON", "ADV", "X"]], V)
    tok_pos[np.arange(50, 250)] = PI["NOUN"]
    tok_pos[[BOS, NL, DOT]] = [PI["BOS"], PI["SPACE"], PI["PUNCT"]]
    pos_tag = tok_pos[tok].astype(np.int16)
    prev = np.concatenate([np.full((S, 1), BOS), tok[:, :-1]], 1).astype(np.int32)
    dn = np.zeros((S, L), np.int16)
    ds = np.zeros((S, L), np.int16)
    for q in range(1, L):
        dn[:, q] = np.where(tok[:, q] == NL, 0, dn[:, q - 1] + 1)
        ds[:, q] = np.where(prev[:, q] == DOT, 0, ds[:, q - 1] + 1)
    domains = ["web", "code", "arxiv"]
    domain = rng.choice(3, S, p=[0.6, 0.25, 0.15]).astype(np.int16)   # unequal, like the Pile
    np.savez(os.path.join(out, "tokens.npz"), tok=tok, prev=prev, pos_tag=pos_tag,
             dist_newline=dn, dist_sentence=ds, domain=domain,
             doc_id=np.arange(S, dtype=np.int32))
    vocab = ["<bos>", "\n", "."] + [f" t{i}" for i in range(3, V)]
    with open(os.path.join(out, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"model_name": "synthetic", "layers": [2, 8], "seq_len": L,
                   "n_layers": n_layers, "d_model": d, "d_transcoder": d_tc,
                   "vocab_size": V, "pos_tags": POS, "domains": domains}, f)

    flat_tok, flat_prev = tok.ravel(), prev.ravel()
    nxt = np.full((S, L), -1)
    nxt[:, :-1] = tok[:, 1:]
    flat_next = nxt.ravel()
    position = np.tile(np.arange(L), S)
    flat_dom = np.repeat(domain, L)
    not_bos = position > 0
    N = S * L

    # ---- weights -----------------------------------------------------------
    G = rng.normal(size=(V, d))
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    s[-k_null:] = 0.01                                    # an effective null space
    W_U = (U * s) @ Vt
    null_dirs = Vt[-k_null:].T                            # (d, k)
    np.save(os.path.join(out, "unembed.npy"), W_U.astype(np.float16))
    enc_down = {L2: rng.normal(size=(d_tc, d)).astype(np.float16)
                for L2 in [3, 4, 5, 9, 10, 11]}
    for L2, W in enc_down.items():
        np.save(os.path.join(out, "downstream", f"enc_L{L2}.npy"), W)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = np.ones(d); lam[0] = 500.0
    for l in [2, 8]:
        np.savez(os.path.join(out, "resid", f"L{l}.npz"), mean=np.zeros(d),
                 cov=(Q * lam) @ Q.T)

    unit = lambda x: x / np.linalg.norm(x)
    counts = np.bincount(flat_tok, minlength=V)
    f = np.log(counts + 0.5)
    f = (f - f.mean()) / f.std()

    # ---- planted latents: (layer, read mask/values, decoder, expected) -------
    def rand_fires(rate):
        m = not_bos & (rng.random(N) < rate)
        return m, rng.uniform(1, 3, N)

    plants = []
    m = flat_tok == 7
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B1_single_token", None))
    m = (flat_prev == 5) & not_bos
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B2_ngram_prev_token", None))
    plants.append((2, not_bos.copy(), 3 * position / L + 0.1 * rng.random(N), rng.normal(size=d),
                   "B3_position_structure", None))
    m = (pos_tag.ravel() == PI["NOUN"]) & not_bos
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B4_syntax", None))
    m = (flat_dom == 1) & not_bos & (rng.random(N) < 0.7)
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B5_context", None))
    plants.append((2, np.zeros(N, bool), np.zeros(N), rng.normal(size=d), "dead", "dead"))
    m, v = rand_fires(0.005)
    plants.append((2, m, v, rng.normal(size=d), "B_unresolved", "C_unresolved"))
    # fires right before token 9, whatever the current token is
    m = (flat_next == 9) & not_bos
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B6_next_token", None))
    # sparse position latent (positions 1-3 only): too sparse for the correlation test
    m = (position >= 1) & (position <= 3) & (rng.random(N) < 0.3)
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B3_position_structure", None))
    # sparse and confined to the smallest domain: skewed, but not document state
    m = (flat_dom == 2) & not_bos & (rng.random(N) < 0.03)
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B5_domain_skewed", None))
    m = np.zeros(N, bool)                                 # live, but too few fires to read
    m[rng.choice(np.flatnonzero(not_bos), 10, replace=False)] = True
    plants.append((2, m, rng.uniform(1, 3, N), rng.normal(size=d), "B_few_fires", None))

    wr = lambda vec, tag: plants.append((8, *rand_fires(0.003), vec, None, tag))
    wr(unit(W_U[10]) + unit(W_U[11]) + unit(W_U[12]), "C1_promote")
    wr(-(unit(W_U[20]) + unit(W_U[21]) + unit(W_U[22])), "C2_suppress")
    wr(null_dirs @ rng.normal(size=k_null), "C5_entropy")
    keep = slice(0, d - k_null)                           # truncated pinv: stay out of null space
    wr(Vt[keep].T @ ((U[:, keep].T @ f) / s[keep]), "C4_token_frequency")
    wr(enc_down[9][5].astype(np.float64), "C7_internal")
    wr(Q[:, 0], "C8_pca_scale")
    for _ in range(40):            # filler, random fires and decoder: must get no tags
        plants.append((8, *rand_fires(0.003), rng.normal(size=d), "B_unresolved", "C_unresolved"))

    rows, dec, rec = [], [], {"uid": [], "tok_idx": [], "val": []}
    for uid, (layer, mask, vals, dvec, eb, ec) in enumerate(plants):
        rows.append({"uid": uid, "layer": layer, "feature": uid, "density_est": mask.mean(),
                     "stratum": 0, "weight": 1.0, "expect_B": eb, "expect_C": ec})
        idx = np.flatnonzero(mask)
        rec["uid"].append(np.full(len(idx), uid, np.int32))
        rec["tok_idx"].append(idx.astype(np.int32))
        rec["val"].append(vals[idx].astype(np.float32))
        dec.append(dvec)
    lat = pd.DataFrame(rows)
    lat.drop(columns=["expect_B", "expect_C"]).to_csv(os.path.join(out, "latents.csv"), index=False)
    np.savez(os.path.join(out, "records.npz"), **{k: np.concatenate(v) for k, v in rec.items()})
    dec = np.stack(dec).astype(np.float16)
    n = len(rows)
    np.savez(os.path.join(out, "weights.npz"), enc=rng.normal(size=(n, d)).astype(np.float16),
             dec=dec, b_enc=np.zeros(n, np.float32), threshold=np.full(n, np.nan, np.float32))
    pd.DataFrame({"uid": lat["uid"], "kl_at_fire": rng.random(n), "kl_after": rng.random(n),
                  "n_contexts": 8}).to_csv(os.path.join(out, "relevance.csv"), index=False)
    return lat[["uid", "expect_B", "expect_C"]]
