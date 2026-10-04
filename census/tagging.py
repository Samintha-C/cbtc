"""Turn detector statistics into tags. Each axis gets a primary tag (first rule
that fires, in precedence order) plus a list of every rule that fired, so you
can audit overlaps instead of trusting the precedence order blindly.

Latents whose read tag is "unresolved" are the ones to send to the LLM labeler.
Live latents with fewer than min_read_fires fires get "B_few_fires" instead of
read rules: with a handful of fires, any top-k share is high by construction.
"""

import numpy as np
import pandas as pd


def _gt(x, t):
    return (not pd.isna(x)) and x >= t


def _le(x, t):
    return (not pd.isna(x)) and x <= t


def activation_tag(r, c):
    if r["n_fires"] < c.min_fires:
        return "A_dead"
    if r["density"] >= c.dense_density:
        return "A_dense"
    if r["density"] < c.rare_density:
        return "A_rare"
    return "A_sparse"


def read_rules(r, c):
    """Precedence: B1, B2, B6, B3, B4, B5_context, B5_domain_skewed. B6 is numbered
    after B5 only because it was added later; it is more specific than B3-B5."""
    fired = []
    if _gt(r.get("cur_top5_share"), c.lexical_top5_share):
        fired.append("B1_single_token" if _gt(r.get("cur_top1_share"), c.single_token_top1_share)
                     else "B1_lexical")
    if (_gt(r.get("bigram_top5_share"), c.ngram_bigram_top5_share)
            or _gt(r.get("prev_top5_share"), c.prev_token_top5_share)):
        fired.append("B2_ngram_prev_token")
    if (_gt(r.get("next_top5_share"), c.next_token_top5_share)
            and _gt(r.get("next_top5_excess"), c.next_token_excess)):
        fired.append("B6_next_token")
    # correlation for dense latents; KS on the fires for any density, with the bar
    # raised for latents with little evidence so chance gaps do not count. Fires in
    # one sequence sit next to each other, so evidence is capped at sequences fired.
    n_ev = min(r.get("n_eff", 0) or 0, r.get("n_seqs_fired", 0) or 0)
    ks_bar = max(c.position_ks, c.ks_crit / np.sqrt(max(n_ev, 1e-9)))
    if (any(_gt(abs(r.get(k, np.nan)), c.position_abs_corr)
            for k in ["corr_position", "corr_dist_newline", "corr_dist_sentence"])
            or any(_gt(r.get(k), ks_bar)
                   for k in ["ks_position", "ks_dist_newline", "ks_dist_sentence"])):
        fired.append("B3_position_structure")
    # one part of speech, spread over many tokens: a semantic class (cities, plural
    # job titles) is also one part of speech but concentrates on few tokens
    if (_gt(r.get("pos_best_share"), c.syntax_pos_share)
            and _gt(r.get("pos_best_lift"), c.syntax_pos_lift)
            and _le(r.get("cur_top20_share"), c.syntax_max_top20_share)):
        fired.append("B4_syntax")
    if _gt(r.get("seq_coverage"), c.context_seq_coverage):
        fired.append("B5_context")
    # a corpus-domain correlate (topic, language or genre), not a mechanism
    if (_gt(r.get("domain_best_share"), c.domain_share)
            and _gt(r.get("domain_best_lift"), c.domain_lift)):
        fired.append("B5_domain_skewed")
    return fired


def write_rules(r, c, early):
    fired = []
    if _gt(r.get("null_frac_rz"), c.entropy_z):
        fired.append("C5_entropy")
    if _gt(abs(r.get("freq_corr", np.nan)), c.freq_abs_corr):
        fired.append("C4_token_frequency")
    if _gt(r.get("vocab_kurtosis"), c.promote_kurtosis):
        fired.append("C1_promote" if r["vocab_skew"] > 0 else "C2_suppress")
    elif _gt(r.get("vocab_var_rz"), c.partition_z):
        fired.append("C3_partition")
    if _gt(r.get("pca_max_cos"), c.pca_abs_cos):
        fired.append("C8_pca_scale")
    if _gt(r.get("downstream_rz"), c.internal_z):
        fired.append("C7_internal")
    if early:   # vocab projections are mostly noise in early layers
        fired = [f for f in fired if not f.startswith(("C1", "C2", "C3", "C4"))]
    return fired


def tag_all(latents, read_df, write_df, relevance_df, cfg, n_layers):
    df = latents.merge(read_df, on="uid", how="left").merge(write_df, on="uid", how="left")
    if relevance_df is not None:
        df = df.merge(relevance_df, on="uid", how="left")
    A, B, Ball, C, Call = [], [], [], [], []
    for i, r in df.iterrows():
        a = activation_tag(r, cfg)
        A.append(a)
        if a == "A_dead":
            B.append("dead"); Ball.append(""); C.append("dead"); Call.append("")
            continue
        rb = read_rules(r, cfg) if r["n_fires"] >= cfg.min_read_fires else []
        early = r["layer"] < cfg.early_layer_frac * n_layers
        wc = write_rules(r, cfg, early)
        if r["n_fires"] < cfg.min_read_fires:
            B.append("B_few_fires")
        else:
            B.append(rb[0] if rb else "B_unresolved")
        Ball.append(";".join(rb))
        C.append(wc[0] if wc else "C_unresolved")
        Call.append(";".join(wc))
    df["tag_A"], df["tag_B"], df["tags_B_all"] = A, B, Ball
    df["tag_C"], df["tags_C_all"] = C, Call
    df["needs_llm"] = df["tag_B"].eq("B_unresolved") | df["tag_C"].eq("C_unresolved")
    return df
