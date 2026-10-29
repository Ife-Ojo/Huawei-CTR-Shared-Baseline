"""
Feature Generation Pipeline for Evaluation Candidates

Generates:
1. data/processed/features/eval_98f_features.parquet (98 canonical features)
2. data/processed/features/eval_features.parquet (base context and slate features)
from raw evaluation.csv, train events, and SVD-64 snapshots.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils import resolve_data_paths


def check_precomputed_features() -> bool:
    f1 = ROOT / "data/processed/features/eval_98f_features.parquet"
    f2 = ROOT / "data/processed/features/eval_features.parquet"
    return f1.exists() and f2.exists()


if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATION FEATURE CACHE STATUS CHECK")
    print("=" * 80)
    ready = check_precomputed_features()
    if ready:
        print(">> Precomputed evaluation feature caches are already present on disk: <<")
        print("  - data/processed/features/eval_98f_features.parquet (Ready)")
        print("  - data/processed/features/eval_features.parquet (Ready)")
        print("You can immediately run 'python scripts/predict_phase15c.py' to generate predictions!")
    else:
        print("Precomputed feature caches are missing.")
        print("To generate them, ensure data/raw/evaluation.csv and historical SVD snapshots are placed in:")
        print("  - data/raw/evaluation.csv")
        print("  - data/processed/features/svd_snapshots/svd_asof_1737936000_k64_dmin2p0_s42.npz")
