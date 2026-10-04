"""Axis A (activation profile) and axis B (read type) statistics.

Everything is computed from sparse fire records plus the per-token metadata
table, using sufficient statistics so dense latents stay cheap: correlations
treat every non-firing token as activation 0.

That zero-filling caps correlations for sparse latents, so position and
structure also get a KS statistic computed on the fires alone: the gap between
the activation-weighted distribution of a covariate where the latent fires and
its distribution over all tokens of the same domains (code has short lines, so
an unmatched baseline makes anything active in code look newline-driven).

Next-token concentration is compared with what a bigram model predicts from the
latent's current tokens, because most of what follows a token is implied by the
token itself (" through" is followed by " the" whatever latent fires on it).
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


def _top_keys(keys, vals, k):
    """The k keys carrying the most activation mass, largest first."""
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq[np.argsort(-np.bincount(inv, weights=vals))[:k]]


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


def _ks_gap(tok_idx, vals, x, base_cdf):
    """Max gap between the activation-weighted CDF of integer covariate x over
    the fires and a baseline CDF."""
    h = np.bincount(x[tok_idx], weights=vals, minlength=len(base_cdf))[:len(base_cdf)]
    return float(np.abs(np.cumsum(h) / h.sum() - base_cdf).max())


class BigramModel:
    """P(next | current) from the run's own tokens."""

    def __init__(self, cur, nxt, V):
        self.V = V
        self.keys, self.counts = np.unique(cur.astype(np.int64) * V + nxt, return_counts=True)
        self.cur_counts = np.bincount(cur, minlength=V)

    def prob(self, cur, nxt):
        k = cur.astype(np.int64) * self.V + nxt
        i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return np.where(self.keys[i] == k, self.counts[i], 0) / np.maximum(self.cur_counts[cur], 1)


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
    structure = {"position": run.position, "dist_newline": run.dist_newline,
                 "dist_sentence": run.dist_sentence}
    real = run.position > 0                       # BOS never fires
    n_dom = len(dom_base)
    dom_cdf = {}                                  # (domain, L) CDF of each covariate
    for k, x in structure.items():
        rows_ = []
        for d in range(n_dom):
            m = real & (run.domain == d)
            rows_.append(np.cumsum(np.bincount(x[m], minlength=L)[:L]) / max(m.sum(), 1))
        dom_cdf[k] = np.stack(rows_)
    V = int(run.tok.max()) + 1
    bigram_key = run.prev.astype(np.int64) * V + run.tok.ravel()
    tok_flat = run.tok.ravel()
    nxt = np.full_like(run.tok, -1)
    nxt[:, :-1] = run.tok[:, 1:]                  # -1: last position, next token unseen
    next_flat = nxt.ravel()
    pair = real & (next_flat >= 0)
    bigrams = BigramModel(tok_flat[pair], next_flat[pair], V)

    rows = []
    for uid in run.latents["uid"]:
        t_idx, v = run.fires(uid)
        r = {"uid": uid, "n_fires": len(v), "density": len(v) / N}
        if len(v) == 0:
            rows.append(r)
            continue
        r["mean_act"] = float(v.mean())
        r["max_act"] = float(v.max())
        r["n_eff"] = float(v.sum() ** 2 / (v.astype(np.float64) ** 2).sum())
        # B1/B2: lexical and n-gram concentration
        r["cur_top1_share"] = _top_share(tok_flat[t_idx], v, 1)
        r["cur_top5_share"] = _top_share(tok_flat[t_idx], v, 5)
        r["cur_top20_share"] = _top_share(tok_flat[t_idx], v, 20)
        r["prev_top5_share"] = _top_share(run.prev[t_idx], v, 5)
        r["bigram_top5_share"] = _top_share(bigram_key[t_idx], v, 5)
        # B6: what comes next, and how much of it the current token already implies
        nt = next_flat[t_idx]
        seen = nt >= 0
        r["next_top5_share"] = _top_share(nt[seen], v[seen], 5)
        if seen.any():
            cur, w = tok_flat[t_idx][seen], v[seen]
            exp = sum((w * bigrams.prob(cur, np.full(len(cur), x))).sum()
                      for x in _top_keys(nt[seen], w, 5)) / w.sum()
            r["next_top5_expected"] = float(exp)
            r["next_top5_excess"] = r["next_top5_share"] - float(exp)
        # B3: position / structure, against the latent's own domain mix
        dom_mass = np.bincount(run.domain[t_idx], weights=v, minlength=n_dom)
        dom_share = dom_mass / dom_mass.sum()
        for name, x in covs.items():
            m, s = cov_moments[name]
            r[f"corr_{name}"] = _corr_with_zeros(t_idx, v, x, m, s, N)
        for name, x in structure.items():
            base = (dom_share[:, None] * dom_cdf[name]).sum(0)
            r[f"ks_{name}"] = _ks_gap(t_idx, v, x, base)
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
        r["n_seqs_fired"] = int(fired.sum())
        r["seq_coverage"] = float((per_seq[fired] / (L - 1)).mean())
        bd = int(dom_share.argmax())
        r["domain_best"] = run.meta["domains"][bd]
        r["domain_best_share"] = float(dom_share[bd])
        r["domain_best_lift"] = float(dom_share[bd] / max(dom_base[bd], 1e-9))
        # top tokens + contexts for prompts / validation sheets
        fmt = lambda ids: " | ".join(repr(run.vocab[i]) for i in ids)
        r["top_tokens"] = fmt(_top_keys(tok_flat[t_idx], v, 8))
        r["top_next_tokens"] = fmt(_top_keys(nt[seen], v[seen], 8)) if seen.any() else ""
        top_fire = t_idx[np.argsort(-v)[:6]]
        r["top_contexts"] = " ||| ".join(run.context(i) for i in top_fire)
        rows.append(r)
    return pd.DataFrame(rows)
