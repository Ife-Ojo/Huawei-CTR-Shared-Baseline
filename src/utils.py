"""
Utility functions for metrics, validation, reproducibility, memory monitoring, and logging.
"""

import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score


def get_logger(name: str = "Huawei-TUDers") -> logging.Logger:
    """Configures and returns a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("Utils")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info("Random seed fixed to %d", seed)


def compute_auc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Computes global pooled ROC AUC metric across all impressions.
    Per competition specifications, metric is evaluated globally across rows,
    not averaged per event or user.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    # Check for constant ground truth
    if len(np.unique(y_true_arr)) < 2:
        logger.warning("Ground truth has only 1 class present. Returning default AUC 0.5.")
        return 0.5
        
    return float(roc_auc_score(y_true_arr, y_pred_arr))


def get_project_root() -> Path:
    """Returns the root directory of the Huawei-TUDers project."""
    return Path(__file__).resolve().parent.parent


def check_disk_space(path: Union[str, Path], min_free_gb: float = 10.0) -> float:
    """
    Checks free disk space for the volume hosting the given path.
    Raises RuntimeError if free space is below min_free_gb.
    Returns free space in GB.
    """
    p = Path(path).resolve()
    # Find existing ancestor directory
    while not p.exists() and p.parent != p:
        p = p.parent
    usage = shutil.disk_usage(p)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient free disk space on volume {p}: {free_gb:.2f} GB available, "
            f"minimum required is {min_free_gb:.2f} GB."
        )
    logger.info("Free disk space check passed: %.2f GB available on %s", free_gb, p)
    return free_gb


def resolve_data_paths() -> Dict[str, Optional[Path]]:
    """
    Detects dataset locations, checking both data/raw/ and data/ root directories,
    as well as processed Parquet locations.
    """
    root = get_project_root()
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    temp_dir = data_dir / "temp_duckdb"

    folds_dir = processed_dir / "folds"
    features_dir = processed_dir / "features"

    train_path = None
    for candidate in [raw_dir / "train.csv", data_dir / "train.csv"]:
        if candidate.exists():
            train_path = candidate
            break

    eval_path = None
    for candidate in [raw_dir / "evaluation.csv", data_dir / "evaluation.csv"]:
        if candidate.exists():
            eval_path = candidate
            break

    example_sub_path = None
    for candidate in [raw_dir / "example_submission.csv", data_dir / "example_submission.csv"]:
        if candidate.exists():
            example_sub_path = candidate
            break

    return {
        "root": root,
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "temp_dir": temp_dir,
        "folds_dir": folds_dir,
        "features_dir": features_dir,
        "train_csv": train_path,
        "evaluation_csv": eval_path,
        "example_submission_csv": example_sub_path,
        "train_cands_parquet": processed_dir / "train_cands.parquet",
        "train_events_parquet": processed_dir / "train_events.parquet",
        "eval_cands_parquet": processed_dir / "eval_cands.parquet",
        "eval_events_parquet": processed_dir / "eval_events.parquet",
        # Aliases for CONTEXT.md compatibility
        "cands_parquet": processed_dir / "cands.parquet",
        "events_parquet": processed_dir / "events.parquet",
    }


def get_peak_memory_mb() -> Optional[float]:
    """
    Returns the peak resident memory (working set) of the current process in MB.
    Uses the Win32 GetProcessMemoryInfo API on Windows (no extra dependencies) and
    resource.getrusage elsewhere. DuckDB allocations live inside this process, so the
    value covers Python + DuckDB + PyArrow. Returns None if measurement is unavailable.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return round(pmc.PeakWorkingSetSize / (1024 ** 2), 1)
            return None
        import resource  # POSIX only
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return None


def validate_submission_format(df_or_path: Union[str, Path, object], expected_rows: int = 529390) -> bool:
    """
    Validates submission against competition constraints:
    - Header: rowid,predicted_probability
    - Exact row count: 529,390
    - No NaN / null / Inf values
    - Probabilities strictly within [0.0, 1.0]
    - Unique rowids
    """
    import pandas as pd

    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path

    expected_cols = ["rowid", "predicted_probability"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columns mismatch: expected {expected_cols}, got {list(df.columns)}")

    if len(df) != expected_rows:
        raise ValueError(f"Row count mismatch: expected {expected_rows}, got {len(df)}")

    if df["rowid"].duplicated().any():
        raise ValueError("Duplicate rowids detected in submission!")

    if df["predicted_probability"].isna().any():
        raise ValueError("Missing / NaN values detected in predicted_probability!")

    probs = df["predicted_probability"].to_numpy()
    if np.isinf(probs).any():
        raise ValueError("Infinite values detected in predictions!")

    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("Probabilities must be bounded within [0.0, 1.0]!")

    logger.info("Submission validation passed: %d valid rows.", len(df))
    return True
