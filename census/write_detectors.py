"""Axis C (write type) statistics, computed from weights only.

  C1-C3 promote / suppress / partition: moments of cos(f_dec, W_U[v]) over the
        vocabulary (Gurnee et al. 2024).
  C4    token-frequency shift: corr(logit effect, log unigram frequency)
        (Stolfo et al. 2024).
  C5    entropy regulation: fraction of f_dec in the effective null space of
        W_U, i.e. its bottom-k singular directions (Stolfo et al. 2024).
  C7    internal / representation-building: max |cos(f_dec, downstream f_enc)|
        (the Eq. 7 wiring).
  C8    PCA / scale: |cos(f_dec, top residual principal components)|.

C3, C5 and C7 are tagged on *_rz columns: robust z-scores against the latent's
own layer (sampling-weighted median and MAD of its live latents). Their random-
direction baselines (vocab_var_z, null_ratio, downstream_ratio) are kept for
reference only: real decoders are not random directions, and on Gemma-2-2B the
median latent already sits at 1.2-2.2x those baselines.

The unembedding is read as a float16 memmap in vocab chunks, so this runs on CPU.
"""

import numpy as np
import pandas as pd


def _unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _merge_topk(vals, idx, new_vals, new_idx, k, largest=True):
    v = np.concatenate([vals, new_vals], axis=1)
    i = np.concatenate([idx, new_idx], axis=1)
    sel = np.argpartition(-v if largest else v, k - 1, axis=1)[:, :k]
    return np.take_along_axis(v, sel, 1), np.take_along_axis(i, sel, 1)


def vocab_stats(dec, W_U, log_freq, chunk=16384, k=10):
    """One pass over the vocab: cosine moments, frequency corr, top +/- tokens."""
    n, _ = dec.shape
    V = W_U.shape[0]
    dec_u = _unit(dec)
    f = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-12)
    s = np.zeros((4, n))
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    ef = np.zeros(n)
    top_v = np.full((n, k), -np.inf)
    top_i = np.zeros((n, k), dtype=np.int64)
    bot_v = np.full((n, k), np.inf)
    bot_i = np.zeros((n, k), dtype=np.int64)
    for a in range(0, V, chunk):
        Wc = np.asarray(W_U[a:a + chunk], dtype=np.float32)
        raw = dec @ Wc.T                                   # (n, c) logit effects
        cos = dec_u @ _unit(Wc).T
        for p in range(4):
            s[p] += (cos ** (p + 1)).sum(1)
        e1 += raw.sum(1)
        e2 += (raw ** 2).sum(1)
        ef += raw @ f[a:a + chunk]
        kk = min(k, raw.shape[1])
        ids = np.arange(a, a + raw.shape[1])
        ti = np.argpartition(-raw, kk - 1, axis=1)[:, :kk]
        top_v, top_i = _merge_topk(top_v, top_i, np.take_along_axis(raw, ti, 1), ids[ti], k)
        bi = np.argpartition(raw, kk - 1, axis=1)[:, :kk]
        bot_v, bot_i = _merge_topk(bot_v, bot_i, np.take_along_axis(raw, bi, 1), ids[bi], k,
                                   largest=False)
    m1, m2, m3, m4 = s / V
    var = m2 - m1 ** 2
    sd = np.sqrt(np.maximum(var, 1e-20))
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / sd ** 3
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / sd ** 4 - 3.0
    e_mean = e1 / V
    e_sd = np.sqrt(np.maximum(e2 / V - e_mean ** 2, 1e-20))
    freq_corr = (ef / V) / e_sd
    order_t = np.argsort(-top_v, 1)
    order_b = np.argsort(bot_v, 1)
    return {
        "vocab_cos_var": var, "vocab_skew": skew, "vocab_kurtosis": kurt,
        "logit_gain": e_sd / np.linalg.norm(dec, axis=1).clip(1e-12),
        "freq_corr": freq_corr,
        "top_promoted": np.take_along_axis(top_i, order_t, 1),
        "top_suppressed": np.take_along_axis(bot_i, order_b, 1),
    }


def null_space_basis(W_U, k, chunk=16384):
    """Bottom-k right-singular directions of W_U (V, d), via the d x d Gram matrix."""
    d = W_U.shape[1]
    G = np.zeros((d, d))
    for a in range(0, W_U.shape[0], chunk):
        Wc = np.asarray(W_U[a:a + chunk], dtype=np.float64)
        G += Wc.T @ Wc
    _, vecs = np.linalg.eigh(G)          # ascending eigenvalues
    return vecs[:, :k]


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]


def robust_z_by_layer(x, layer, weight, live, min_pop):
    """(x - median) / (1.4826 * MAD), with the sampling-weighted median and MAD of
    the live latents in the same layer; all layers pooled if a layer has too few."""
    z = np.full(len(x), np.nan)
    ok = live & np.isfinite(x)
    for l in np.unique(layer):
        idx = np.flatnonzero(layer == l)
        ref = idx[ok[idx]]
        if len(ref) < min_pop:
            ref = np.flatnonzero(ok)
        if len(ref) == 0:
            continue
        med = _wmedian(x[ref], weight[ref])
        mad = _wmedian(np.abs(x[ref] - med), weight[ref])
        z[idx] = (x[idx] - med) / max(1.4826 * mad, 1e-12)
    return z


def max_abs_cos(dec_u, enc, chunk=8192):
    best = np.zeros(dec_u.shape[0])
    for a in range(0, enc.shape[0], chunk):
        e = _unit(np.asarray(enc[a:a + chunk], dtype=np.float32))
        best = np.maximum(best, np.abs(dec_u @ e.T).max(1))
    return best


def compute_write_stats(run, cfg):
    w = run.weights()
    dec = w["dec"].astype(np.float32)
    lat = run.latents
    d = dec.shape[1]
    W_U = run.unembed()
    counts = np.bincount(run.tok.ravel(), minlength=W_U.shape[0])
    log_freq = np.log(counts + 0.5)

    # null: random unit directions pushed through the same unembedding
    rng = np.random.default_rng(0)
    rand = _unit(rng.normal(size=(cfg.n_null_directions, d))).astype(np.float32)
    vs = vocab_stats(np.concatenate([dec, rand]), W_U, log_freq)
    n = len(dec)
    null_var = vs["vocab_cos_var"][n:]
    vs = {k: v[:n] for k, v in vs.items()}
    out = pd.DataFrame({"uid": lat["uid"]})
    out["vocab_var_z"] = (vs["vocab_cos_var"] - null_var.mean()) / (null_var.std() + 1e-12)
    for key in ["vocab_cos_var", "vocab_skew", "vocab_kurtosis", "logit_gain", "freq_corr"]:
        out[key] = vs[key]
    fmt = lambda ids: " | ".join(repr(run.vocab[i]) for i in ids)
    out["top_promoted"] = [fmt(r) for r in vs["top_promoted"]]
    out["top_suppressed"] = [fmt(r) for r in vs["top_suppressed"]]

    U0 = null_space_basis(W_U, cfg.null_space_k)
    dec_u = _unit(dec)
    rho = np.linalg.norm(dec_u @ U0, axis=1)
    out["null_frac"] = rho
    out["null_ratio"] = rho / np.sqrt(cfg.null_space_k / d)

    out["downstream_max_cos"] = np.nan
    out["downstream_ratio"] = np.nan
    out["pca_max_cos"] = np.nan
    n_layers = run.meta["n_layers"]
    for layer, idx in lat.groupby("layer").indices.items():
        best, M = np.zeros(len(idx)), 0
        for L2 in range(layer + 1, min(n_layers, layer + 1 + 8)):
            enc = run.downstream_enc(L2)
            if enc is None:
                continue
            best = np.maximum(best, max_abs_cos(dec_u[idx], enc))
            M += enc.shape[0]
        if M:
            baseline = np.sqrt(2 * np.log(2 * M) / d)
            out.loc[idx, "downstream_max_cos"] = best
            out.loc[idx, "downstream_ratio"] = best / baseline
        res = run.resid(layer)
        if res is not None:
            _, vecs = np.linalg.eigh(res["cov"])
            pcs = vecs[:, ::-1][:, :3]
            out.loc[idx, "pca_max_cos"] = np.abs(dec_u[idx] @ pcs).max(1)

    # C3 / C5 / C7 are judged against the latent's own layer
    live = (run.r_end - run.r_start)[lat["uid"].values] >= cfg.min_fires
    for col, z_col in [("vocab_cos_var", "vocab_var_rz"), ("null_frac", "null_frac_rz"),
                       ("downstream_max_cos", "downstream_rz")]:
        out[z_col] = robust_z_by_layer(out[col].values.astype(float), lat["layer"].values,
                                       lat["weight"].values.astype(float), live,
                                       cfg.min_z_population)
    return out
