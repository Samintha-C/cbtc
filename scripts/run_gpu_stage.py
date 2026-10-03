"""Stage 1 (GPU): python scripts/run_gpu_stage.py --out runs/gemma2b --layers 2 8 14 20 24"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from census.config import GPUConfig
from census.gpu_stage import run_gpu_stage

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, nargs="+")
ap.add_argument("--n-seqs", type=int)
ap.add_argument("--latents-per-layer", type=int)
ap.add_argument("--model", dest="model_name")
ap.add_argument("--transcoders", dest="transcoder_set")
ap.add_argument("--skip-relevance", action="store_true")
ap.add_argument("--prep-only", action="store_true",
                help="CPU only: build tokens.npz and exit (no model, no GPU)")
args = ap.parse_args()

cfg = GPUConfig()
for k in ["layers", "n_seqs", "latents_per_layer", "model_name", "transcoder_set"]:
    if getattr(args, k) is not None:
        setattr(cfg, k, getattr(args, k))
run_gpu_stage(cfg, args.out, skip_relevance=args.skip_relevance, prep_only=args.prep_only)
print(f"done -> {args.out}")
