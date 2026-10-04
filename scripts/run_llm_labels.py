"""Stage 3b (CPU, network): LLM labels for what the rules leave unresolved.

python scripts/run_llm_labels.py --run runs/gemma2b-l0-50 [--model qwen3] [--max 0]

Reads the rule-based census written by run_offline.py (runs/<run>/census/), asks
the NRP LLM gateway about each latent whose read or write tag is unresolved, and
writes a second report to runs/<run>/census_llm/. The LLM only fills unresolved
tags; it never relabels what a rule claimed. Needs NRP_API_KEY.

Labels are appended to census_llm/llm_labels.jsonl as they arrive, so rerunning
the same command resumes instead of starting over.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from census.labeling import label_latents
from census.report import validation_sheet, write_report

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--census", default=None, help="rule-based census dir (default: <run>/census)")
ap.add_argument("--out", default=None, help="output dir (default: <run>/census_llm)")
ap.add_argument("--model", default="qwen3")
ap.add_argument("--workers", type=int, default=8, help="concurrent calls, capped at the "
                "model's fair-use limit")
ap.add_argument("--max", type=int, default=0, help="label at most this many, balanced "
                "across layers (0: all)")
args = ap.parse_args()

src = args.census or os.path.join(args.run, "census")
out = args.out or os.path.join(args.run, "census_llm")
os.makedirs(out, exist_ok=True)
df = pd.read_csv(os.path.join(src, "census.csv"))

# few-fires latents have too little evidence to read; dead ones have none
todo = df[df["needs_llm"] & ~df["tag_B"].isin(["dead", "B_few_fires"])]
if args.max and len(todo) > args.max:   # round-robin over layers, random within a layer
    todo = (todo.sample(frac=1, random_state=0)
                .assign(_k=lambda d: d.groupby("layer").cumcount())
                .sort_values("_k", kind="stable").head(args.max))
print(f"{len(todo)} of {len(df)} latents to label | by layer {todo.layer.value_counts().sort_index().to_dict()}")

labels = pd.DataFrame(label_latents(todo, os.path.join(out, "llm_labels.jsonl"),
                                    model=args.model, workers=args.workers))
if len(labels):
    labels = labels.rename(columns={c: f"llm_{c}" for c in labels.columns if c != "uid"})
    df = df.merge(labels, on="uid", how="left")
    fill_b = df["tag_B"].eq("B_unresolved") & df["llm_read_tag"].notna()
    fill_c = df["tag_C"].eq("C_unresolved") & df["llm_write_tag"].notna()
    df.loc[fill_b, "tag_B"] = df.loc[fill_b, "llm_read_tag"]
    df.loc[fill_c, "tag_C"] = df.loc[fill_c, "llm_write_tag"]
    if "llm_parse_error" in df:
        print(f"unparseable replies: {int(df['llm_parse_error'].fillna(False).astype(bool).sum())}")

write_report(df, out)
validation_sheet(df, os.path.join(out, "validation_sheet.csv"))
print(df.groupby("layer")["tag_B"].value_counts().unstack(0).fillna(0).astype(int).to_string())
print(f"report -> {out}")
