"""Axis A (activation profile) and axis B (read type) statistics.

Everything is computed from sparse fire records plus the per-token metadata
table, using sufficient statistics so dense latents stay cheap: correlations
treat every non-firing token as activation 0.
"""

import numpy as np
import pandas as pd

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def _top_share(keys, vals, k):
    """Share of activation mass carried by the top-k keys."""
    if len(keys) == 0:
        return np.nan
    uniq, inv = np.unique(keys, return_inverse=True)
    mass = np.bincount(inv, weights=vals)
    mass.sort()
    return float(mass[-k:].sum() / mass.sum())


def _global_moments(x):
    x = x.astype(np.float64)
    return x.mean(), x.std()


def _corr_with_zeros(tok_idx, vals, x, x_mean, x_std, N):
    """Pearson corr between activation a (0 off-records) and covariate x."""
    if len(vals) < 2 or x_std == 0:
        return np.nan
    a_mean = vals.sum() / N
    a_var = (vals.astype(np.float64) ** 2).sum() / N - a_mean ** 2
    if a_var <= 0:
        return np.nan
    cov = (vals * x[tok_idx]).sum() / N - a_mean * x_mean
    return float(cov / (np.sqrt(a_var) * x_std))


def compute_read_stats(run):
    N, L = run.N, run.L
    tags = run.meta["pos_tags"]
    content_ids = np.array([i for i, t in enumerate(tags) if t in CONTENT_POS])
    pos_base = np.bincount(run.pos_tag, minlength=len(tags)) / N
    dom_base = np.bincount(run.domain, minlength=len(run.meta["domains"])) / N
    content_base = pos_base[content_ids].sum() if len(content_ids) else np.nan

    covs = {
        "position": np.log1p(run.position),
        "dist_newline": np.log1p(run.dist_newline),
        "dist_sentence": np.log1p(run.dist_sentence),
    }
    cov_moments = {k: _global_moments(v) for k, v in covs.items()}
    bigram_key = run.prev.astype(np.int64) * (int(run.tok.max()) + 1) + run.tok.ravel()
    tok_flat = run.tok.ravel()

    rows = []
    for uid in run.latents["uid"]:
        t_idx, v = run.fires(uid)
        r = {"uid": uid, "n_fires": len(v), "density": len(v) / N}
        if len(v) == 0:
            rows.append(r)
            continue
        r["mean_act"] = float(v.mean())
        r["max_act"] = float(v.max())
        # B1/B2: lexical and n-gram concentration
        r["cur_top1_share"] = _top_share(tok_flat[t_idx], v, 1)
        r["cur_top5_share"] = _top_share(tok_flat[t_idx], v, 5)
        r["prev_top5_share"] = _top_share(run.prev[t_idx], v, 5)
        r["bigram_top5_share"] = _top_share(bigram_key[t_idx], v, 5)
        # B3: position / structure
        for name, x in covs.items():
            m, s = cov_moments[name]
            r[f"corr_{name}"] = _corr_with_zeros(t_idx, v, x, m, s, N)
        # B4: syntax
        pos_mass = np.bincount(run.pos_tag[t_idx], weights=v, minlength=len(tags))
        pos_share = pos_mass / pos_mass.sum()
        best = int(pos_share.argmax())
        r["pos_best"] = tags[best]
        r["pos_best_share"] = float(pos_share[best])
        r["pos_best_lift"] = float(pos_share[best] / max(pos_base[best], 1e-9))
        if len(content_ids):
            cs = pos_share[content_ids].sum()
            r["content_share"] = float(cs)
            r["content_lift"] = float(cs / max(content_base, 1e-9))
        # B5: context / document state
        seqs = t_idx // L
        per_seq = np.bincount(seqs, minlength=run.S)
        fired = per_seq > 0
        r["seq_coverage"] = float((per_seq[fired] / (L - 1)).mean())
        dom_mass = np.bincount(run.domain[t_idx], weights=v, minlength=len(dom_base))
        dom_share = dom_mass / dom_mass.sum()
        bd = int(dom_share.argmax())
        r["domain_best"] = run.meta["domains"][bd]
        r["domain_best_share"] = float(dom_share[bd])
        r["domain_best_lift"] = float(dom_share[bd] / max(dom_base[bd], 1e-9))
        # top tokens + contexts for prompts / validation sheets
        uniq, inv = np.unique(tok_flat[t_idx], return_inverse=True)
        mass = np.bincount(inv, weights=v)
        top = uniq[np.argsort(-mass)[:8]]
        r["top_tokens"] = " | ".join(repr(run.vocab[i]) for i in top)
        top_fire = t_idx[np.argsort(-v)[:6]]
        r["top_contexts"] = " ||| ".join(run.context(i) for i in top_fire)
        rows.append(r)
    return pd.DataFrame(rows)
