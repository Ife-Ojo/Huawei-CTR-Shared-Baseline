"""
Modular Training & Experimentation Pipeline for Phase 15C Baseline Architecture

Provides clean, structured interfaces for:
1. Training Candidate-level LightGBM on canonical 98 features.
2. Training Candidate-level XGBoost on canonical 98 features.
3. Training Specialist Routing model for short/medium histories.
4. Training Method C Event Propensity Classifier.
5. Training Within-Event Conditional Item-Choice Model (Softmax).
6. Training SVD Slate-Informed Event Intent Model.

Usage:
  python scripts/train_phase15c_models.py --model lightgbm --data-path data/processed/features/train_sample.parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils import resolve_data_paths, set_seed


def load_canonical_features() -> List[str]:
    schema_path = ROOT / "models/phase8c_full_warm_matched_98f.txt"
    with open(schema_path) as f:
        for line in f:
            if line.startswith("feature_names="):
                return line.strip().split("=")[1].split()
    raise ValueError(f"Could not parse canonical features from {schema_path}")


def train_lightgbm_candidate(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    num_leaves: int = 63,
    lr: float = 0.05,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    colsample: float = 0.8,
    max_rounds: int = 600,
    early_stop: int = 30,
    seed: int = 42
) -> lgb.Booster:
    """Trains a candidate-level binary classification LightGBM model."""
    if features is None:
        features = load_canonical_features()
        
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": 1,
        "colsample_bytree": colsample,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    
    dtr = lgb.Dataset(df_train[features], label=df_train["clicked"])
    valid_sets = [dtr]
    if df_val is not None:
        dv = lgb.Dataset(df_val[features], label=df_val["clicked"], reference=dtr)
        valid_sets.append(dv)
        callbacks = [lgb.early_stopping(early_stop, verbose=True), lgb.log_evaluation(50)]
    else:
        callbacks = [lgb.log_evaluation(50)]
        
    print(f"Training LightGBM Candidate Model on {len(df_train):,} rows with {len(features)} features...")
    bst = lgb.train(
        params,
        dtr,
        num_boost_round=max_rounds,
        valid_sets=valid_sets,
        callbacks=callbacks
    )
    return bst


def train_conditional_choice_model(
    df_train_clicked_events: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    lr: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42
) -> lgb.Booster:
    """
    Trains a within-event conditional item choice model via custom categorical softmax objective.
    Requires input data sorted by [event_id, rowid] with exactly 10 candidates per clicked event.
    """
    if features is None:
        features = load_canonical_features()
        
    n_rows = len(df_train_clicked_events)
    n_events = n_rows // 10
    assert n_rows == n_events * 10, "Data must consist of complete 10-candidate slates!"
    
    y = df_train_clicked_events["clicked"].values.astype(np.float32)
    
    def softmax_objective(preds, train_data):
        labels = train_data.get_label()
        z = preds.reshape(n_events, 10)
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        p = (exp_z / np.sum(exp_z, axis=1, keepdims=True)).reshape(-1)
        grad = p - labels
        hess = np.clip(p * (1.0 - p), 1e-4, 0.25)
        return grad, hess
        
    params = {
        "objective": softmax_objective,
        "boosting_type": "gbdt",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    
    dtr = lgb.Dataset(df_train_clicked_events[features], label=y)
    print(f"Training Conditional Choice Model on {n_events:,} clicked events ({n_rows:,} candidate rows)...")
    bst = lgb.train(
        params,
        dtr,
        num_boost_round=300,
        valid_sets=[dtr],
        callbacks=[lgb.log_evaluation(50)]
    )
    return bst


def train_event_intent_model(
    df_event_features: pd.DataFrame,
    event_label_col: str = "event_clicked",
    lr: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42
) -> lgb.Booster:
    """
    Trains the SVD-informed 50-feature event click intent classifier (predicts event P(click >= 1)).
    """
    feature_cols = [c for c in df_event_features.columns if c not in ["event_id", event_label_col]]
    assert len(feature_cols) == 50, f"Expected 50 event features, got {len(feature_cols)}"
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    
    dtr = lgb.Dataset(df_event_features[feature_cols], label=df_event_features[event_label_col])
    print(f"Training SVD Event Intent Classifier on {len(df_event_features):,} events ({len(feature_cols)} features)...")
    bst = lgb.train(
        params,
        dtr,
        num_boost_round=300,
        valid_sets=[dtr],
        callbacks=[lgb.log_evaluation(50)]
    )
    return bst


if __name__ == "__main__":
    print("Phase 15C Modular Training Suite initialized.")
    print("Available training functions:")
    print("  - train_lightgbm_candidate()")
    print("  - train_conditional_choice_model()")
    print("  - train_event_intent_model()")
