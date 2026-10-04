"""Stage 3 (CPU): python scripts/run_offline.py --run runs/gemma2b

Rule-based tags and report. LLM labels for the unresolved: scripts/run_llm_labels.py."""
import argparse
import os
import sys

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

write_report(df, out)
validation_sheet(df, os.path.join(out, "validation_sheet.csv"))
print(df.groupby("layer")[["tag_A", "tag_B", "tag_C"]].agg(lambda s: s.value_counts().to_dict()))
print(f"report -> {out}")
