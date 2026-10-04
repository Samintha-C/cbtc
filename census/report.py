"""Census outputs. Proportions are reweighted by each latent's sampling weight,
because stratified sampling deliberately over-represents rare and dense latents."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def weighted_share(df, tag_col):
    t = df.pivot_table(index="layer", columns=tag_col, values="weight", aggfunc="sum", fill_value=0)
    return t.div(t.sum(1), axis=0)


def stacked_bar(share, title, path):
    ax = share.plot(kind="bar", stacked=True, figsize=(9, 4.5), colormap="tab20", width=0.8)
    ax.set_ylabel("share of latents (sampling-weighted)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def crosstab_heatmap(df, path):
    live = df[df["tag_A"] != "A_dead"]
    ct = pd.crosstab(live["tag_B"], live["tag_C"], values=live["weight"], aggfunc="sum").fillna(0)
    ct = ct / ct.values.sum()
    fig, ax = plt.subplots(figsize=(1.1 * ct.shape[1] + 3, 0.5 * ct.shape[0] + 2))
    im = ax.imshow(ct.values, cmap="viridis")
    ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(ct.shape[0]), ct.index, fontsize=8)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_title("Read tag x write tag (weighted share of live latents)")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return ct


def write_report(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "census.csv"), index=False)
    for col, name in [("tag_A", "activation"), ("tag_B", "read"), ("tag_C", "write")]:
        share = weighted_share(df, col)
        share.to_csv(os.path.join(out_dir, f"share_{name}_by_layer.csv"))
        stacked_bar(share, f"{name} tags by layer", os.path.join(out_dir, f"share_{name}_by_layer.png"))
    ct = crosstab_heatmap(df, os.path.join(out_dir, "read_x_write.png"))
    ct.to_csv(os.path.join(out_dir, "read_x_write.csv"))
    if "kl_at_fire" in df:
        rel = df.groupby(["layer", "tag_B"])["kl_at_fire"].median().unstack()
        rel.to_csv(os.path.join(out_dir, "relevance_median_by_read_tag.csv"))


def validation_sheet(df, path, n=150, seed=0):
    """Blind sheet for hand labeling: evidence only, no automatic tags."""
    live = df[df["tag_A"] != "A_dead"]
    per = max(1, n // max(1, live["tag_B"].nunique()))
    sample = (live.groupby("tag_B", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
    sample = sample.sample(frac=1, random_state=seed)
    cols = ["uid", "layer", "feature", "density", "top_contexts", "top_tokens",
            "top_next_tokens", "top_promoted", "top_suppressed"]
    sheet = sample[[c for c in cols if c in sample]].copy()
    sheet["human_read_tag"] = ""
    sheet["human_write_tag"] = ""
    sheet["notes"] = ""
    sheet.to_csv(path, index=False)
    return sheet


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def agreement(census, labeled, axis="B"):
    m = labeled.merge(census[["uid", f"tag_{axis}"]], on="uid")
    m = m[m[f"human_{'read' if axis == 'B' else 'write'}_tag"].astype(str).str.len() > 0]
    h, a = m[f"human_{'read' if axis == 'B' else 'write'}_tag"], m[f"tag_{axis}"]
    per_tag = (pd.DataFrame({"auto": a, "match": h.values == a.values})
                 .groupby("auto")["match"].agg(["mean", "count"]).rename(columns={"mean": "precision"}))
    return {"n": len(m), "accuracy": float((h.values == a.values).mean()) if len(m) else np.nan,
            "kappa": cohen_kappa(list(h), list(a)) if len(m) else np.nan, "per_tag": per_tag}
