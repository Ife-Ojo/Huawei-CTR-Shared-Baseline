"""
Verification script for Huawei TechArena CTR submission files.

Checks:
1. File exists and is non-empty.
2. Exactly 529,390 rows plus header.
3. Exact columns: ['rowid', 'predicted_probability'].
4. Rowids match evaluation.csv exactly in order and values.
5. All probabilities are finite (no NaN, null, or Inf).
6. All probabilities are bounded within [0.0, 1.0].
7. Computes and displays SHA-256 hash.
"""

import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_HASHES = {
    "Phase 15C Champion (50% SVD Event Intent)": "0e992d347d2b01e3e17ad05280f69491ce8e411ec4008cbd4fa0d23d46e4da9c",
    "Phase 14 Champion (20% Conditional Choice)": "b82a1595f131c1107b47a920b86b012781e4331624b20b830deab2c0973cc059",
    "Phase 12C Champion (3-Model + Method C)": "5d28cc555310f20a01892820a0d425e01677988aebb0a3c7b5c83e8fac305d3a",
}


def verify_submission(csv_path: str) -> bool:
    path = Path(csv_path)
    if not path.is_absolute():
        path = ROOT / path
        
    print("=" * 80)
    print(f"VERIFYING SUBMISSION: {path.name}")
    print("=" * 80)
    
    if not path.exists():
        print(f"ERROR: File does not exist at {path}")
        return False
        
    # Check SHA-256
    with open(path, "rb") as f:
        file_bytes = f.read()
        sha256 = hashlib.sha256(file_bytes).hexdigest()
    print(f"File SHA-256: {sha256}")
    
    for name, official_sha in OFFICIAL_HASHES.items():
        if sha256.lower() == official_sha.lower():
            print(f">> MATCHED KNOWN BENCHMARK: {name} <<")
            
    # Load DataFrame
    df = pd.read_csv(path)
    print(f"Total Rows:   {len(df):,}")
    print(f"Columns:      {list(df.columns)}")
    
    if len(df) != 529390:
        print(f"ERROR: Expected 529,390 rows, got {len(df)}")
        return False
        
    if list(df.columns) != ["rowid", "predicted_probability"]:
        print(f"ERROR: Invalid columns {list(df.columns)}")
        return False
        
    probs = df["predicted_probability"].values
    if not np.isfinite(probs).all():
        print("ERROR: Non-finite values (NaN / Inf) detected!")
        return False
        
    p_min = float(np.min(probs))
    p_max = float(np.max(probs))
    p_mean = float(np.mean(probs))
    p_std = float(np.std(probs))
    print(f"Probabilities: min={p_min:.6f}, max={p_max:.6f}, mean={p_mean:.6f}, std={p_std:.6f}")
    
    if p_min < 0.0 or p_max > 1.0:
        print(f"ERROR: Probabilities outside [0, 1] range! Min={p_min}, Max={p_max}")
        return False
        
    # Check against evaluation.csv if present
    eval_csv = ROOT / "data/raw/evaluation.csv"
    if eval_csv.exists():
        eval_raw = pd.read_csv(eval_csv, usecols=["rowid"])
        if not (df["rowid"].values == eval_raw["rowid"].values).all():
            print("ERROR: Rowid sequence does NOT match evaluation.csv!")
            return False
        print("Rowid alignment with evaluation.csv: 100% verified.")
        
    print("=" * 80)
    print("ALL SUBMISSION INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase15c_final_50pct_svd_event_intent_submission.csv"
    success = verify_submission(target)
    sys.exit(0 if success else 1)
