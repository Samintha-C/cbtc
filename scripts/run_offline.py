"""Stage 2 (CPU): python scripts/run_offline.py --run runs/gemma2b [--llm]"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from census.config import TagConfig
from census.read_detectors import compute_read_stats
from census.report import validation_sheet, write_report
from census.runio import Run
from census.tagging import tag_all
from census.write_detectors import compute_write_stats

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--llm", action="store_true", help="label unresolved latents with Claude")
ap.add_argument("--llm-model", default="claude-haiku-4-5-20251001")
ap.add_argument("--llm-max", type=int, default=500)
args = ap.parse_args()

out = args.out or os.path.join(args.run, "census")
os.makedirs(out, exist_ok=True)
cfg = TagConfig()
cfg.save(os.path.join(out, "tag_config.json"))

run = Run(args.run)
print(f"{len(run.latents)} latents, {run.N:,} tokens")
read_df = compute_read_stats(run)
print("read stats done")
write_df = compute_write_stats(run, cfg)
print("write stats done")
df = tag_all(run.latents, read_df, write_df, run.relevance(), cfg, run.meta["n_layers"])

if args.llm:
    from census.labeling import label_with_claude
    todo = df[df["needs_llm"] & (df["tag_A"] != "A_dead")].head(args.llm_max)
    labels = pd.DataFrame(label_with_claude(todo, model=args.llm_model))
    labels = labels.rename(columns={c: f"llm_{c}" for c in labels.columns if c != "uid"})
    df = df.merge(labels, on="uid", how="left")
    fill_b = df["tag_B"].eq("B_unresolved") & df["llm_read_tag"].notna()
    fill_c = df["tag_C"].eq("C_unresolved") & df["llm_write_tag"].notna()
    df.loc[fill_b, "tag_B"] = df.loc[fill_b, "llm_read_tag"]
    df.loc[fill_c, "tag_C"] = df.loc[fill_c, "llm_write_tag"]

write_report(df, out)
validation_sheet(df, os.path.join(out, "validation_sheet.csv"))
print(df.groupby("layer")[["tag_A", "tag_B", "tag_C"]].agg(lambda s: s.value_counts().to_dict()))
print(f"report -> {out}")
