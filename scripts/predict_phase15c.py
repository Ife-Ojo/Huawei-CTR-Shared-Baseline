"""
Phase 15C Champion Official Inference & Reproduction Pipeline

Official Score: 0.810459 global ROC AUC
Architecture:
  - 50% Phase 12C 3-Model Ensemble (45% LightGBM + 30% XGBoost + 25% CatBoost + Specialist Routing + Method C Calibration)
  - 50% Phase 15 SVD Slate Event Intent * Phase 14 Conditional Item-Choice

Reproduces:
  outputs/phase15c_final_50pct_svd_event_intent_submission.csv
Expected SHA-256:
  0e992d347d2b01e3e17ad05280f69491ce8e411ec4008cbd4fa0d23d46e4da9c
"""

import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Set project root dynamically
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils import resolve_data_paths

EXPECTED_SHA256 = "0e992d347d2b01e3e17ad05280f69491ce8e411ec4008cbd4fa0d23d46e4da9c"


def compute_svd_slate_features(df_cands: pd.DataFrame, tau_p75: float = 0.175988) -> pd.DataFrame:
    """Computes the 14 SVD slate features representing cross-candidate collaborative geometry."""
    n_rows = len(df_cands)
    n_events = n_rows // 10
    assert n_rows == n_events * 10, "Candidate rows must be exactly 10 per event!"
    
    u_aff = df_cands["svd64_refined_user_affinity"].values.astype(np.float32).reshape(n_events, 10)
    u_aff_sorted = np.sort(u_aff, axis=1)
    u_max = u_aff_sorted[:, 9]
    u_mean = np.mean(u_aff, axis=1)
    u_std = np.std(u_aff, axis=1)
    u_margin = u_aff_sorted[:, 9] - u_aff_sorted[:, 8]
    u_top3 = np.sum(u_aff_sorted[:, 7:10], axis=1)
    u_cnt_p75 = np.sum(u_aff >= tau_p75, axis=1).astype(np.float32)
    
    h_aff = df_cands["svd64_hybrid_user_affinity"].values.astype(np.float32).reshape(n_events, 10)
    h_aff_sorted = np.sort(h_aff, axis=1)
    h_max = h_aff_sorted[:, 9]
    h_mean = np.mean(h_aff, axis=1)
    h_std = np.std(h_aff, axis=1)
    h_margin = h_aff_sorted[:, 9] - h_aff_sorted[:, 8]
    
    hist_max = np.max(df_cands["svd_history_max_item_cos_32"].values.astype(np.float32).reshape(n_events, 10), axis=1)
    hist_mean = np.mean(df_cands["svd_history_mean_cos_32"].values.astype(np.float32).reshape(n_events, 10), axis=1)
    
    dot_arr = df_cands["svd_user_item_dot_32"].values.astype(np.float32).reshape(n_events, 10)
    dot_max = np.max(dot_arr, axis=1)
    dot_mean = np.mean(dot_arr, axis=1)
    
    event_ids = df_cands["event_id"].values[::10]
    return pd.DataFrame({
        "event_id": event_ids,
        "slate_svd64_user_affinity_max": u_max,
        "slate_svd64_user_affinity_mean": u_mean,
        "slate_svd64_user_affinity_std": u_std,
        "slate_svd64_user_affinity_top1_minus_top2": u_margin,
        "slate_svd64_user_affinity_top3_sum": u_top3,
        "slate_svd64_user_affinity_cnt_ge_p75": u_cnt_p75,
        "slate_svd64_hybrid_affinity_max": h_max,
        "slate_svd64_hybrid_affinity_mean": h_mean,
        "slate_svd64_hybrid_affinity_std": h_std,
        "slate_svd64_hybrid_top1_minus_top2": h_margin,
        "slate_svd_history_max_cos": hist_max,
        "slate_svd_history_mean_cos": hist_mean,
        "slate_svd_user_dot_max": dot_max,
        "slate_svd_user_dot_mean": dot_mean,
    })


def run_prediction_pipeline(output_csv_path: Optional[str] = None) -> str:
    print("=" * 80)
    print("PHASE 15C OFFICIAL CHAMPION PREDICTION PIPELINE (OFFICIAL SCORE: 0.810459)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Resolve paths
    paths = resolve_data_paths()
    feat_dir = ROOT / "data/processed/features"
    eval_csv_path = ROOT / "data/raw/evaluation.csv"
    p12c_csv_path = ROOT / "outputs/phase12c_final_3model_champion_submission.csv"
    
    if output_csv_path is None:
        output_csv_path = str(ROOT / "outputs/phase15c_final_50pct_svd_event_intent_submission.csv")
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify inputs
    assert p12c_csv_path.exists(), f"Missing Phase 12C baseline submission at {p12c_csv_path}"
    
    # Load canonical feature schema
    schema_file = ROOT / "models/phase8c_full_warm_matched_98f.txt"
    with open(schema_file) as f:
        for line in f:
            if line.startswith("feature_names="):
                canonical_98 = line.strip().split("=")[1].split()
                break
    print(f"1. Canonical feature schema loaded ({len(canonical_98)} features).")
    
    # Load raw evaluation rowids and order
    if eval_csv_path.exists():
        eval_raw = pd.read_csv(eval_csv_path, usecols=["rowid", "event_id"])
        print(f"2. Loaded evaluation metadata from {eval_csv_path} ({len(eval_raw):,} rows).")
    else:
        # Graceful fallback for fresh clones before evaluation.csv is manually placed:
        # Reconstructs exact sequence from Phase 12C submission rowids and eval_features.parquet event_ids
        print(f"2. Notice: data/raw/evaluation.csv not found (excluded from Git per competition rules).")
        print(f"   Using canonical rowid sequence from outputs/phase12c_final_3model_champion_submission.csv")
        print(f"   and event_id mapping from data/processed/features/eval_features.parquet.")
        df_12c_rids = pd.read_csv(p12c_csv_path, usecols=["rowid"])
        df_ev_map = pq.read_table(feat_dir / "eval_features.parquet", columns=["rowid", "event_id"]).to_pandas()
        eval_raw = df_12c_rids.merge(df_ev_map, on="rowid")
        
    assert len(eval_raw) == 529390, f"Expected 529,390 rows, got {len(eval_raw)}"
    n_eval_evs = len(eval_raw) // 10
    print(f"   Evaluation candidates verified: {len(eval_raw):,} rows across {n_eval_evs:,} events.")
    
    # Load Phase 12C Champion predictions
    df_champ = pd.read_csv(p12c_csv_path)
    assert (df_champ["rowid"].values == eval_raw["rowid"].values).all(), "Rowid sequence mismatch with Phase 12C!"
    p_champ_eval = df_champ["predicted_probability"].values.astype(np.float64)
    print(f"3. Loaded Phase 12C Champion base predictions (Mean: {p_champ_eval.mean():.6f}).")
    
    # 2. Extract Event Features for Phase 15 SVD-Informed Event Intent
    print("4. Computing Event Intent features (36 base + 14 slate SVD = 50 features)...")
    con = duckdb.connect()
    df_eval_ev_base = con.execute(f"""
    SELECT 
        event_id,
        FIRST(u_history_length) AS u_history_length, FIRST(u_ctr_smoothed) AS u_ctr_smoothed,
        FIRST(u_prior_event_count) AS u_prior_event_count, FIRST(u_prior_click_count) AS u_prior_click_count,
        FIRST(u_prior_no_click_rate) AS u_prior_no_click_rate, FIRST(u_events_per_observed_day) AS u_events_per_observed_day,
        FIRST(u_seconds_since_last_event) AS u_seconds_since_last_event, FIRST(u_seconds_since_last_click) AS u_seconds_since_last_click,
        FIRST(u_usual_time_bucket) AS u_usual_time_bucket, FIRST(u_time_bucket_activity_share) AS u_time_bucket_activity_share,
        FIRST(u_weekend_activity_share) AS u_weekend_activity_share, FIRST(u_is_unseen_in_fit) AS u_is_unseen,
        FIRST(c_hour_of_day) AS c_hour_of_day, FIRST(c_hour_sin) AS c_hour_sin, FIRST(c_hour_cos) AS c_hour_cos,
        FIRST(c_time_bucket) AS c_time_bucket, FIRST(c_day_type) AS c_day_type, FIRST(c_activity_velocity) AS c_activity_velocity,
        AVG(i_ctr_smoothed) AS slate_ctr_mean, MAX(i_ctr_smoothed) AS slate_ctr_max, MIN(i_ctr_smoothed) AS slate_ctr_min,
        STDDEV(i_ctr_smoothed) AS slate_ctr_std, AVG(i_prior_impression_count) AS slate_imp_mean,
        MAX(i_prior_impression_count) AS slate_imp_max, MIN(i_prior_impression_count) AS slate_imp_min,
        STDDEV(i_prior_impression_count) AS slate_imp_std, AVG(i_prior_click_count) AS slate_clk_mean,
        MAX(i_prior_click_count) AS slate_clk_max, AVG(i_recent_vs_long_term_ratio) AS slate_recent_ratio_mean,
        MAX(i_recent_vs_long_term_ratio) AS slate_recent_ratio_max, SUM(h_candidate_in_history) AS slate_hist_count,
        MAX(h_candidate_in_history) AS slate_hist_max, SUM(h_candidate_recency_decay_count) AS slate_hist_decay_sum,
        MAX(h_candidate_recency_decay_count) AS slate_hist_decay_max, SUM(ui_prior_impressions) AS slate_ui_imp_sum,
        MAX(ui_prior_impressions) AS slate_ui_imp_max
    FROM '{feat_dir / "eval_features.parquet"}'
    GROUP BY event_id
    ORDER BY event_id
    """).df()
    con.close()
    
    svd_cols_eval = ["rowid", "svd64_refined_user_affinity", "svd64_hybrid_user_affinity", "svd_history_max_item_cos_32", "svd_history_mean_cos_32", "svd_user_item_dot_32"]
    df_eval_cands = pq.read_table(feat_dir / "eval_98f_features.parquet", columns=svd_cols_eval).to_pandas()
    df_eval_cands = df_eval_cands.merge(eval_raw, on="rowid").sort_values(by=["event_id", "rowid"]).reset_index(drop=True)
    
    df_eval_svd = compute_svd_slate_features(df_eval_cands)
    del df_eval_cands
    gc.collect()
    
    df_eval_ev_all = df_eval_ev_base.merge(df_eval_svd, on="event_id")
    del df_eval_ev_base, df_eval_svd
    gc.collect()
    
    base_cols = [c for c in df_eval_ev_all.columns if c not in ["event_id"] and not c.startswith("slate_svd")]
    svd_cols = [c for c in df_eval_ev_all.columns if c.startswith("slate_svd")]
    cols_50 = base_cols + svd_cols
    assert len(cols_50) == 50, f"Expected 50 event features, got {len(cols_50)}"
    
    # Predict event click intent
    ckpt_ev_up = ROOT / "models/checkpoints/phase15_production_svd_event_classifier_lgb.txt"
    bst_prod_ev = lgb.Booster(model_file=str(ckpt_ev_up))
    q_eval_pred = bst_prod_ev.predict(df_eval_ev_all[cols_50].values.astype(np.float32))
    q_map_eval = dict(zip(df_eval_ev_all["event_id"], q_eval_pred))
    print(f"   Event intent predicted: Mean={q_eval_pred.mean():.6f}, Std={q_eval_pred.std():.6f}")
    
    # 3. Compute Phase 14 Conditional Choice Probabilities
    print("5. Predicting within-event conditional item choice (Softmax)...")
    ckpt_cond = ROOT / "models/checkpoints/phase14_production_conditional_choice_lgb.txt"
    booster_cond_prod = lgb.Booster(model_file=str(ckpt_cond))
    
    cols_to_load = list(set(["rowid"] + canonical_98))
    df_eval_98 = pq.read_table(feat_dir / "eval_98f_features.parquet", columns=cols_to_load).to_pandas()
    df_eval_sorted = df_eval_98.merge(eval_raw, on="rowid").sort_values(by=["event_id", "rowid"]).reset_index(drop=True)
    del df_eval_98
    gc.collect()
    
    z_eval = booster_cond_prod.predict(df_eval_sorted[canonical_98].values.astype(np.float32))
    z_mat_eval = z_eval.reshape(n_eval_evs, 10)
    z_max_eval = np.max(z_mat_eval, axis=1, keepdims=True)
    exp_z_eval = np.exp(z_mat_eval - z_max_eval)
    p_choice_mat_eval = exp_z_eval / np.sum(exp_z_eval, axis=1, keepdims=True)
    p_choice_eval = p_choice_mat_eval.reshape(-1)
    
    df_eval_sorted["p_choice"] = p_choice_eval
    df_eval_sorted["q_event_up"] = df_eval_sorted["event_id"].map(q_map_eval).values
    df_eval_sorted["p_cond_up"] = df_eval_sorted["q_event_up"] * df_eval_sorted["p_choice"]
    
    map_cond_up = dict(zip(df_eval_sorted["rowid"], df_eval_sorted["p_cond_up"]))
    eval_raw["p_cond_up"] = eval_raw["rowid"].map(map_cond_up).values
    del df_eval_sorted
    gc.collect()
    
    # 4. Phase 15C 50% Final Blend
    print("6. Blending 50% Phase 12C + 50% (q_event_up * p_choice)...")
    w = 0.50
    p_final_50 = (1.0 - w) * p_champ_eval + w * eval_raw["p_cond_up"].values
    
    df_sub = pd.DataFrame({
        "rowid": eval_raw["rowid"].values,
        "predicted_probability": p_final_50
    })
    
    # Rigorous checks
    assert len(df_sub) == 529390, f"Expected 529,390 rows, got {len(df_sub)}"
    assert (df_sub["rowid"].values == eval_raw["rowid"].values).all(), "Rowid order mismatch!"
    assert np.isfinite(df_sub["predicted_probability"].values).all(), "NaN or Inf detected!"
    assert (df_sub["predicted_probability"].values >= 0.0).all() and (df_sub["predicted_probability"].values <= 1.0).all()
    
    df_sub.to_csv(out_path, index=False)
    print(f"7. Submission written to {out_path}.")
    
    # Verify SHA-256
    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    
    print("-" * 80)
    print(f"Generated SHA-256: {sha}")
    print(f"Expected SHA-256:  {EXPECTED_SHA256}")
    match = (sha == EXPECTED_SHA256)
    print(f"Exact Byte-for-Byte Match: {match}")
    print("-" * 80)
    assert match, f"SHA-256 mismatch! Got {sha}, expected {EXPECTED_SHA256}"
    
    elapsed = time.time() - t0
    print(f">> SUCCESS: Phase 15C Champion perfectly reproduced in {elapsed:.2f}s! <<\n")
    return sha


if __name__ == "__main__":
    run_prediction_pipeline()
