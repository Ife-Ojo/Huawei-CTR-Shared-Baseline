"""
Phase 5C: Event-level click propensity classifier and two-stage CTR modelling.

Hypothesis:
P(cand_i clicked) = P(event has any click) * P(cand_i selected | event has a click)
                  = q_e * r_i

where:
  q_e = P(event has any click) estimated via an event-level LightGBM classifier.
  r_i = p_i / sum_{j=1..10}(p_j) represents the within-event conditional candidate preference.

Blended candidate probability:
  p_{blend, i} = (1 - alpha) * p_i + alpha * (q_e * r_i)
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

from src.features import training_partition_paths
from src.utils import get_logger, get_peak_memory_mb

logger = get_logger("EventModel")

# Canonical 23 event features
EVENT_USER_COLS: List[str] = [
    "u_ctr_smoothed",
    "u_prior_no_click_rate",
    "u_history_length",
    "u_prior_event_count",
    "u_prior_click_count",
    "u_seconds_since_last_event",
    "u_seconds_since_last_click",
    "c_time_bucket",
    "c_day_type",
    "c_hour_of_day",
    "c_activity_velocity",
]

EVENT_AGG_FEATURE_COLS: List[str] = [
    "ev_svd_hist_cos_max",
    "ev_svd_hist_cos_mean",
    "ev_svd_hist_cos_std",
    "ev_svd_hist_cos_top1_minus_top2",
    "ev_svd_hist_cos_top1_minus_mean",
    "ev_svd_user_cos_max",
    "ev_svd_user_cos_mean",
    "ev_svd_user_cos_top1_minus_top2",
    "ev_item_ctr_max",
    "ev_item_ctr_mean",
    "ev_cands_in_history_count",
    "ev_cf_nonzero_count",
]

EVENT_FEATURE_COLS: List[str] = EVENT_USER_COLS + EVENT_AGG_FEATURE_COLS

FORBIDDEN_EVENT_COLS = frozenset({
    "rowid",
    "event_id",
    "user_id",
    "clicked",
    "event_clicked",
    "event_ts",
    "timestamp",
    "split",
    "day",
})


def extract_event_features_from_tables(
    tbl_base: pa.Table,
    tbl_svd: pa.Table,
    tbl_cf: pa.Table,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Extracts 23 event-level features from aligned candidate tables.
    Each event must consist of exactly 10 contiguous rows with identical event_id.
    """
    n_rows = len(tbl_base)
    if n_rows % 10 != 0:
        raise ValueError(f"Candidate table row count {n_rows} must be a multiple of 10.")
    n_events = n_rows // 10

    # Verify event_id contiguous integrity
    event_ids_2d = tbl_base["event_id"].to_numpy(zero_copy_only=False).reshape(n_events, 10)
    if not (event_ids_2d == event_ids_2d[:, [0]]).all():
        raise ValueError("Candidates for each event must be contiguous in blocks of 10.")
    event_id_1d = event_ids_2d[:, 0]

    # Target
    event_clicked = None
    if include_target:
        if "clicked" not in tbl_base.column_names:
            raise ValueError("Target column 'clicked' not found in table.")
        clicked_2d = tbl_base["clicked"].to_numpy(zero_copy_only=False).reshape(n_events, 10)
        clicks_per_ev = clicked_2d.sum(axis=1)
        if (clicks_per_ev > 1).any():
            raise ValueError("At most one candidate may be clicked per event.")
        event_clicked = clicked_2d.max(axis=1).astype(np.uint8)

    feats: Dict[str, np.ndarray] = {}

    # User & Context features (invariant across 10 candidates in the event)
    for col in EVENT_USER_COLS:
        arr_2d = tbl_base[col].to_numpy(zero_copy_only=False).reshape(n_events, 10)
        feats[col] = arr_2d[:, 0].astype(np.float32)

    # Candidate Aggregations
    # 1. SVD history cosine
    svd_hist = tbl_svd["svd_history_mean_cos_32"].to_numpy(zero_copy_only=False).reshape(n_events, 10).astype(np.float32)
    feats["ev_svd_hist_cos_max"] = np.max(svd_hist, axis=1)
    feats["ev_svd_hist_cos_mean"] = np.mean(svd_hist, axis=1)
    feats["ev_svd_hist_cos_std"] = np.std(svd_hist, axis=1)
    sorted_hist = np.sort(svd_hist, axis=1)
    feats["ev_svd_hist_cos_top1_minus_top2"] = sorted_hist[:, -1] - sorted_hist[:, -2]
    feats["ev_svd_hist_cos_top1_minus_mean"] = sorted_hist[:, -1] - feats["ev_svd_hist_cos_mean"]

    # 2. SVD direct user-item cosine
    svd_user = tbl_svd["svd_user_item_cos_32"].to_numpy(zero_copy_only=False).reshape(n_events, 10).astype(np.float32)
    feats["ev_svd_user_cos_max"] = np.max(svd_user, axis=1)
    feats["ev_svd_user_cos_mean"] = np.mean(svd_user, axis=1)
    sorted_user = np.sort(svd_user, axis=1)
    feats["ev_svd_user_cos_top1_minus_top2"] = sorted_user[:, -1] - sorted_user[:, -2]

    # 3. Item prior CTR
    i_ctr = tbl_base["i_ctr_smoothed"].to_numpy(zero_copy_only=False).reshape(n_events, 10).astype(np.float32)
    feats["ev_item_ctr_max"] = np.max(i_ctr, axis=1)
    feats["ev_item_ctr_mean"] = np.mean(i_ctr, axis=1)

    # 4. History sequence presence count
    h_cand = tbl_base["h_candidate_in_history"].to_numpy(zero_copy_only=False).reshape(n_events, 10)
    feats["ev_cands_in_history_count"] = np.sum(h_cand == 1, axis=1).astype(np.float32)

    # 5. Non-zero CF similarity count
    cf_sim = tbl_cf["cf_max_similarity"].to_numpy(zero_copy_only=False).reshape(n_events, 10).astype(np.float32)
    feats["ev_cf_nonzero_count"] = np.sum(cf_sim > 0.0, axis=1).astype(np.float32)

    df = pd.DataFrame(feats)
    df.insert(0, "event_id", event_id_1d)
    if event_clicked is not None:
        df["event_clicked"] = event_clicked

    return df


def load_fold_b_event_datasets(
    features_dir: Union[str, Path] = "data/processed/features",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the complete Fold B warm training (Jan 21-25) and validation (Jan 26) event tables.
    """
    fdir = Path(features_dir)
    base_val_p = fdir / "fold_b_val_features.parquet"
    svd_val_p = fdir / "fold_b_val_svd_features.parquet"
    cf_val_p = fdir / "fold_b_val_cf_features.parquet"

    # Validation
    base_needed = ["rowid", "event_id", "clicked"] + EVENT_USER_COLS + ["i_ctr_smoothed", "h_candidate_in_history"]
    svd_needed = ["rowid", "svd_history_mean_cos_32", "svd_user_item_cos_32"]
    cf_needed = ["rowid", "cf_max_similarity"]

    tbl_b_val = pq.read_table(base_val_p, columns=base_needed)
    tbl_s_val = pq.read_table(svd_val_p, columns=svd_needed)
    tbl_c_val = pq.read_table(cf_val_p, columns=cf_needed)

    df_val = extract_event_features_from_tables(tbl_b_val, tbl_s_val, tbl_c_val, include_target=True)
    del tbl_b_val, tbl_s_val, tbl_c_val
    gc.collect()

    # Training (Jan 21-25)
    warm_paths = training_partition_paths("fold_b", features_dir=fdir)[1:]  # Jan 21-25
    svd_tr_p = fdir / "fold_b_train_full_warm_svd_features.parquet"
    cf_tr_p = fdir / "fold_b_train_full_warm_cf_features.parquet"

    tbl_s_tr_full = pq.read_table(svd_tr_p, columns=svd_needed)
    tbl_c_tr_full = pq.read_table(cf_tr_p, columns=cf_needed)

    tr_dfs: List[pd.DataFrame] = []
    row_offset = 0

    for p in warm_paths:
        day_tbl = pq.read_table(p, columns=base_needed)
        n_day_rows = len(day_tbl)
        s_slice = tbl_s_tr_full.slice(row_offset, n_day_rows)
        c_slice = tbl_c_tr_full.slice(row_offset, n_day_rows)
        row_offset += n_day_rows

        df_day = extract_event_features_from_tables(day_tbl, s_slice, c_slice, include_target=True)
        tr_dfs.append(df_day)

    df_tr = pd.concat(tr_dfs, ignore_index=True)
    del tbl_s_tr_full, tbl_c_tr_full, tr_dfs
    gc.collect()

    return df_tr, df_val


def train_event_propensity_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Trains a binary LightGBM event propensity model predicting whether an event yields a click.
    """
    fcols = list(feature_cols or EVENT_FEATURE_COLS)
    for forbidden in FORBIDDEN_EVENT_COLS:
        if forbidden in fcols:
            raise ValueError(f"Forbidden column '{forbidden}' in predictor features.")

    X_tr = df_train[fcols].to_numpy(dtype=np.float32)
    y_tr = df_train["event_clicked"].to_numpy(dtype=np.uint8)
    X_val = df_val[fcols].to_numpy(dtype=np.float32)
    y_val = df_val["event_clicked"].to_numpy(dtype=np.uint8)

    lgb_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=fcols, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=fcols, reference=lgb_tr, free_raw_data=False)

    default_params: Dict[str, Any] = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "seed": 42,
        "deterministic": True,
        "verbose": -1,
        "num_threads": 4,
    }
    if params:
        default_params.update(params)

    t0 = time.time()
    booster = lgb.train(
        default_params,
        lgb_tr,
        num_boost_round=1000,
        valid_sets=[lgb_tr, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    train_time = time.time() - t0

    q_e_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_auc = float(roc_auc_score(y_val, q_e_val))
    val_loss = float(log_loss(y_val, q_e_val))

    imp_split = booster.feature_importance(importance_type="split")
    imp_gain = booster.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": fcols,
        "split": imp_split,
        "gain": imp_gain,
    }).sort_values("gain", ascending=False)

    metrics = {
        "best_iteration": int(booster.best_iteration),
        "val_event_auc": val_auc,
        "val_event_log_loss": val_loss,
        "train_time_sec": train_time,
        "peak_ram_mb": get_peak_memory_mb(),
        "importance": importance_df.to_dict(orient="records"),
    }
    return booster, metrics


def compute_two_stage_probabilities(
    p_candidates: np.ndarray,
    q_events: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes two-stage candidate probabilities:
      p_{two_stage, i} = q_e * r_i
      where r_i = p_i / sum_{j=1..10}(p_j)

    Parameters:
      p_candidates: (10*N,) or (N, 10) candidate probabilities from pointwise model.
      q_events: (N,) event click propensity from event model.
    """
    if p_candidates.ndim == 1:
        if len(p_candidates) != len(q_events) * 10:
            raise ValueError(f"p_candidates length {len(p_candidates)} does not match 10 * {len(q_events)}")
        p_2d = p_candidates.reshape(-1, 10)
        was_1d = True
    else:
        if p_candidates.shape[1] != 10:
            raise ValueError(f"p_candidates must have 10 columns, got {p_candidates.shape[1]}")
        p_2d = p_candidates
        was_1d = False

    event_sums = np.sum(p_2d, axis=1, keepdims=True)
    # Conditional candidate preference r_i
    r_2d = np.where(event_sums > eps, p_2d / np.maximum(event_sums, eps), 0.1)

    # Safe bounds
    q_2d = q_events[:, np.newaxis]
    p_two_stage_2d = np.clip(q_2d * r_2d, 0.0, 1.0)

    if was_1d:
        return p_two_stage_2d.flatten()
    return p_two_stage_2d


def evaluate_two_stage_blend_grid(
    y_true_cands: np.ndarray,
    p_candidates: np.ndarray,
    q_events: np.ndarray,
    alpha_grid: Sequence[float] = (0.00, 0.25, 0.50, 0.75, 1.00),
) -> List[Dict[str, Any]]:
    """
    Evaluates the predeclared blend grid:
      p_{blend, i} = (1 - alpha) * p_i + alpha * p_{two_stage, i}
    on global candidate-level ROC AUC, log loss, and clicked-event top-1 accuracy.
    """
    p_two_stage = compute_two_stage_probabilities(p_candidates, q_events)
    n_events = len(q_events)
    p_cand_2d = p_candidates.reshape(n_events, 10)
    y_true_2d = y_true_cands.reshape(n_events, 10)
    has_click = y_true_2d.max(axis=1) == 1
    clicked_indices = np.where(has_click)[0]

    baseline_auc = float(roc_auc_score(y_true_cands, p_candidates))
    results = []

    for alpha in alpha_grid:
        p_blend = (1.0 - alpha) * p_candidates + alpha * p_two_stage
        p_blend = np.clip(p_blend, 0.0, 1.0)

        auc = float(roc_auc_score(y_true_cands, p_blend))
        loss = float(log_loss(y_true_cands, p_blend))

        # Clicked-event top-1 accuracy
        p_blend_2d = p_blend.reshape(n_events, 10)
        top1_cands = np.argmax(p_blend_2d, axis=1)
        top1_acc = float((y_true_2d[clicked_indices, top1_cands[clicked_indices]] == 1).mean())

        results.append({
            "alpha": float(alpha),
            "global_roc_auc": auc,
            "delta_vs_baseline": auc - baseline_auc,
            "log_loss": loss,
            "clicked_top1_accuracy": top1_acc,
        })

    return results
