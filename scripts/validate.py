"""python scripts/validate.py --census runs/x/census/census.csv --labeled my_labels.csv"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from census.report import agreement

ap = argparse.ArgumentParser()
ap.add_argument("--census", required=True)
ap.add_argument("--labeled", required=True)
args = ap.parse_args()
census, labeled = pd.read_csv(args.census), pd.read_csv(args.labeled).fillna("")
for axis in ["B", "C"]:
    res = agreement(census, labeled, axis)
    print(f"\nAxis {axis}: n={res['n']}  accuracy={res['accuracy']:.3f}  kappa={res['kappa']:.3f}")
    print(res["per_tag"].to_string())
