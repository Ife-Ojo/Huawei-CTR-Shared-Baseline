"""
Competition-grade temporal validation framework for TechArena CTR.
Implements chronological folds (Fold A and Fold B), leakage prevention,
reusable fold artifacts, rowid-matched global ROC AUC evaluation, and leakage-safe benchmarks.

Differences from official evaluation:
- In the official competition, evaluation covers users' first events after Jan 27 00:00 UTC over a multi-day window.
- Fold A (Jan 25) and Fold B (Jan 26) are practical 24-hour chronological approximations of this scheme,
  where each user active on that validation day has only their first event evaluated in the primary slice.
"""

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import get_logger, resolve_data_paths

logger = get_logger("Validate")

# Unix Timestamps (UTC)
TS_JAN_20_0000 = 1737331200  # 2025-01-20 00:00:00 UTC (Training start)
TS_JAN_25_0000 = 1737763200  # 2025-01-25 00:00:00 UTC (Fold A cutoff)
TS_JAN_26_0000 = 1737849600  # 2025-01-26 00:00:00 UTC (Fold B cutoff / Primary)
TS_JAN_27_0000 = 1737936000  # 2025-01-27 00:00:00 UTC (Official competition cutoff)

FOLD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fold_a": {
        "name": "fold_a",
        "description": "Fold A: Fit Jan 20-24, Validate Jan 25 (24-hour first-event approximation)",
        "fit_start_ts": TS_JAN_20_0000,
        "cutoff_ts": TS_JAN_25_0000,
        "val_end_ts": TS_JAN_26_0000,
        "fit_start_utc": "2025-01-20T00:00:00+00:00",
        "cutoff_utc": "2025-01-25T00:00:00+00:00",
        "val_end_utc": "2025-01-26T00:00:00+00:00",
    },
    "fold_b": {
        "name": "fold_b",
        "description": "Fold B (Primary): Fit Jan 20-25, Validate Jan 26 (24-hour first-event approximation)",
        "fit_start_ts": TS_JAN_20_0000,
        "cutoff_ts": TS_JAN_26_0000,
        "val_end_ts": TS_JAN_27_0000,
        "fit_start_utc": "2025-01-20T00:00:00+00:00",
        "cutoff_utc": "2025-01-26T00:00:00+00:00",
        "val_end_utc": "2025-01-27T00:00:00+00:00",
    },
}


def build_fold_artifacts(
    fold_name: str = "fold_b",
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Builds lightweight, deterministic fold definitions using DuckDB out-of-core.
    Saves:
    - <fold_name>_metadata.json
    - <fold_name>_events.parquet: event_id and split ('fit', 'val_first', 'val_later')
    - <fold_name>_val_first.parquet: rowid, event_id, item_id, clicked
    - <fold_name>_val_all.parquet: rowid, event_id, item_id, clicked, is_first
    """
    if fold_name not in FOLD_CONFIGS:
        raise ValueError(f"Unknown fold '{fold_name}'. Available: {list(FOLD_CONFIGS.keys())}")

    cfg = FOLD_CONFIGS[fold_name]
    paths = resolve_data_paths()
    folds_dir = output_dir or paths["folds_dir"]
    folds_dir.mkdir(parents=True, exist_ok=True)

    meta_file = folds_dir / f"{fold_name}_metadata.json"
    events_file = folds_dir / f"{fold_name}_events.parquet"
    val_first_file = folds_dir / f"{fold_name}_val_first.parquet"
    val_all_file = folds_dir / f"{fold_name}_val_all.parquet"

    if meta_file.exists() and events_file.exists() and val_first_file.exists() and val_all_file.exists() and not force:
        logger.info("Fold artifacts for '%s' already exist. Loading metadata.", fold_name)
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Building fold artifacts for '%s' (%s)...", fold_name, cfg["description"])
    t0 = time.time()

    train_events_p = paths["train_events_parquet"].as_posix()
    train_cands_p = paths["train_cands_parquet"].as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit = '1GB';")
    con.execute("PRAGMA threads = 2;")

    # 1. Classify all events into 'fit', 'val_first', and 'val_later'
    # Deterministic tie-breaking on timestamp ASC, event_id ASC
    # Crucial: Rank only within the validation window to find the first event on that day
    q_events = f"""
    COPY (
        WITH val_events AS (
            SELECT
                event_id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp ASC, event_id ASC
                ) AS val_rn
            FROM '{train_events_p}'
            WHERE timestamp >= {cfg["cutoff_ts"]} AND timestamp < {cfg["val_end_ts"]}
        ),
        classified AS (
            SELECT
                e.event_id,
                CASE
                    WHEN e.timestamp < {cfg["cutoff_ts"]} THEN 'fit'
                    WHEN v.val_rn = 1 THEN 'val_first'
                    ELSE 'val_later'
                END AS split
            FROM '{train_events_p}' e
            LEFT JOIN val_events v ON e.event_id = v.event_id
            WHERE e.timestamp < {cfg["val_end_ts"]}
        )
        SELECT event_id, split
        FROM classified
        WHERE split IN ('fit', 'val_first', 'val_later')
        ORDER BY event_id ASC
    ) TO '{(events_file.as_posix())}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_events)

    # 2. Extract val_first candidate rows
    q_val_first = f"""
    COPY (
        SELECT
            c.rowid,
            c.event_id,
            c.item_id,
            c.clicked
        FROM '{train_cands_p}' c
        JOIN '{events_file.as_posix()}' e ON c.event_id = e.event_id
        WHERE e.split = 'val_first'
        ORDER BY c.rowid ASC
    ) TO '{(val_first_file.as_posix())}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_val_first)

    # 3. Extract val_all candidate rows (diagnostic)
    q_val_all = f"""
    COPY (
        SELECT
            c.rowid,
            c.event_id,
            c.item_id,
            c.clicked,
            CAST(e.split = 'val_first' AS UTINYINT) AS is_first
        FROM '{train_cands_p}' c
        JOIN '{events_file.as_posix()}' e ON c.event_id = e.event_id
        WHERE e.split IN ('val_first', 'val_later')
        ORDER BY c.rowid ASC
    ) TO '{(val_all_file.as_posix())}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_val_all)

    # 4. Compute exact summary metrics
    fit_event_cnt = con.execute(f"SELECT count(*) FROM '{events_file.as_posix()}' WHERE split = 'fit'").fetchone()[0]
    val_first_event_cnt = con.execute(f"SELECT count(*) FROM '{events_file.as_posix()}' WHERE split = 'val_first'").fetchone()[0]
    val_all_event_cnt = con.execute(f"SELECT count(*) FROM '{events_file.as_posix()}' WHERE split IN ('val_first', 'val_later')").fetchone()[0]

    val_first_rows_cnt = con.execute(f"SELECT count(*) FROM '{val_first_file.as_posix()}'").fetchone()[0]
    val_first_pos_cnt = con.execute(f"SELECT count(*) FROM '{val_first_file.as_posix()}' WHERE clicked = 1").fetchone()[0]
    val_first_neg_cnt = val_first_rows_cnt - val_first_pos_cnt

    val_all_rows_cnt = con.execute(f"SELECT count(*) FROM '{val_all_file.as_posix()}'").fetchone()[0]
    val_all_pos_cnt = con.execute(f"SELECT count(*) FROM '{val_all_file.as_posix()}' WHERE clicked = 1").fetchone()[0]
    val_all_neg_cnt = val_all_rows_cnt - val_all_pos_cnt

    # No-click events count in val_first
    no_click_first = con.execute(f"""
        SELECT count(*) FROM (
            SELECT sum(clicked) as s FROM '{val_first_file.as_posix()}' GROUP BY event_id
        ) WHERE s = 0
    """).fetchone()[0]

    # Unseen users in val_first (users who never appeared in fit events)
    unseen_users = con.execute(f"""
        SELECT count(distinct v.user_id)
        FROM '{train_events_p}' v
        JOIN '{events_file.as_posix()}' ef ON v.event_id = ef.event_id
        WHERE ef.split = 'val_first'
          AND NOT EXISTS (
              SELECT 1
              FROM '{train_events_p}' f
              JOIN '{events_file.as_posix()}' ef2 ON f.event_id = ef2.event_id
              WHERE ef2.split = 'fit' AND f.user_id = v.user_id
          )
    """).fetchone()[0]

    fit_rows_cnt = fit_event_cnt * 10

    metadata = {
        "fold": fold_name,
        "description": cfg["description"],
        "fit_start_utc": cfg["fit_start_utc"],
        "cutoff_utc": cfg["cutoff_utc"],
        "val_end_utc": cfg["val_end_utc"],
        "fit_start_ts": cfg["fit_start_ts"],
        "cutoff_ts": cfg["cutoff_ts"],
        "val_end_ts": cfg["val_end_ts"],
        "fit_events": fit_event_cnt,
        "fit_rows": fit_rows_cnt,
        "val_first_events": val_first_event_cnt,
        "val_first_rows": val_first_rows_cnt,
        "val_first_pos_rows": val_first_pos_cnt,
        "val_first_neg_rows": val_first_neg_cnt,
        "val_first_ctr": val_first_pos_cnt / val_first_rows_cnt if val_first_rows_cnt > 0 else 0.0,
        "val_first_no_click_events": no_click_first,
        "val_first_pct_no_click": (no_click_first / val_first_event_cnt * 100.0) if val_first_event_cnt > 0 else 0.0,
        "val_first_unseen_users": unseen_users,
        "val_first_pct_unseen_users": (unseen_users / val_first_event_cnt * 100.0) if val_first_event_cnt > 0 else 0.0,
        "val_all_events": val_all_event_cnt,
        "val_all_rows": val_all_rows_cnt,
        "val_all_pos_rows": val_all_pos_cnt,
        "val_all_neg_rows": val_all_neg_cnt,
        "val_all_ctr": val_all_pos_cnt / val_all_rows_cnt if val_all_rows_cnt > 0 else 0.0,
        "artifacts": {
            "events_parquet": str(events_file),
            "val_first_parquet": str(val_first_file),
            "val_all_parquet": str(val_all_file),
            "metadata_json": str(meta_file),
        },
        "build_duration_seconds": round(time.time() - t0, 2),
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    con.close()
    logger.info("Fold '%s' built in %.2fs: %d fit events, %d val_first events (%d rows).",
                fold_name, metadata["build_duration_seconds"], fit_event_cnt, val_first_event_cnt, val_first_rows_cnt)
    return metadata


def evaluate_predictions(
    predictions: Union[pd.DataFrame, Dict[str, float], Any],
    ground_truth: Optional[pd.DataFrame] = None,
    fold_name: str = "fold_b",
    slice_type: str = "first",
) -> Dict[str, Any]:
    """
    Rigorously evaluates predictions against ground truth for a given validation fold.
    Matches strictly on 'rowid'.

    Args:
        predictions: DataFrame with ['rowid', 'predicted_probability' (or 'pred')], or dict {rowid: prob}.
        ground_truth: DataFrame with ['rowid', 'clicked'], or None (loads from fold parquet).
        fold_name: 'fold_a' or 'fold_b'.
        slice_type: 'first' (primary 1st event) or 'all' (all events on val day).

    Returns:
        Dict containing global AUC, row counts, pos/neg counts, and CTR.
    """
    paths = resolve_data_paths()
    folds_dir = paths["folds_dir"]

    # Load ground truth if not supplied
    if ground_truth is None:
        val_file = folds_dir / f"{fold_name}_val_{slice_type}.parquet"
        if not val_file.exists():
            build_fold_artifacts(fold_name=fold_name)
        con = duckdb.connect()
        gt_df = con.execute(f"SELECT rowid, clicked FROM '{val_file.as_posix()}'").df()
        con.close()
    else:
        gt_df = ground_truth if isinstance(ground_truth, pd.DataFrame) else pd.DataFrame(ground_truth)

    for col in ["rowid", "clicked"]:
        if col not in gt_df.columns:
            raise KeyError(f"Ground truth missing required column: '{col}'")

    # Format predictions to DataFrame
    if isinstance(predictions, dict):
        pred_df = pd.DataFrame(list(predictions.items()), columns=["rowid", "pred"])
    elif isinstance(predictions, (str, Path)):
        pred_df = pd.read_parquet(predictions) if str(predictions).endswith(".parquet") else pd.read_csv(predictions)
    elif hasattr(predictions, "to_pandas"):
        pred_df = predictions.to_pandas()
    else:
        pred_df = predictions.copy()

    # Normalize pred column name
    pred_col = None
    for candidate in ["predicted_probability", "pred", "probability", "score"]:
        if candidate in pred_df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        raise KeyError(f"Predictions DataFrame must contain probability column. Available: {list(pred_df.columns)}")

    # 1. Check for duplicate rowids in predictions
    dup_rowids = pred_df["rowid"].duplicated().sum()
    if dup_rowids > 0:
        raise ValueError(f"Found {dup_rowids} duplicate rowids in predictions!")

    # 2. Check for non-finite values
    probs = pred_df[pred_col].to_numpy()
    if np.isnan(probs).any() or np.isinf(probs).any():
        raise ValueError("Non-finite values (NaN / Inf) detected in predictions!")

    # 3. Match by rowid
    merged = pd.merge(gt_df, pred_df[["rowid", pred_col]], on="rowid", how="left")

    # 4. Check for missing rowids
    missing_count = merged[pred_col].isna().sum()
    if missing_count > 0:
        raise ValueError(f"Mismatched predictions: {missing_count} ground truth rowids are missing from predictions!")

    y_true = merged["clicked"].to_numpy()
    y_pred = merged[pred_col].to_numpy()

    # 5. Check for single-class validation set
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning("Ground truth has only 1 class present (%s). Global AUC is undefined; returning 0.5.", unique_classes)
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_pred))

    pos_rows = int((y_true == 1).sum())
    neg_rows = int((y_true == 0).sum())
    total_rows = len(y_true)
    ctr = pos_rows / total_rows if total_rows > 0 else 0.0

    result = {
        "fold": fold_name,
        "slice_type": slice_type,
        "auc": round(auc, 6),
        "total_rows": total_rows,
        "pos_rows": pos_rows,
        "neg_rows": neg_rows,
        "ctr": round(ctr, 6),
    }
    logger.info("Evaluation Result [%s - %s] -> AUC: %.5f (pos: %d, neg: %d, total: %d)",
                fold_name, slice_type, auc, pos_rows, neg_rows, total_rows)
    return result


def run_baseline_benchmarks(
    fold_name: str = "fold_b",
    smoothing_k: float = 50.0,
    force_build: bool = False,
) -> Dict[str, Any]:
    """
    Executes leakage-safe baseline benchmarks using DuckDB:
    A. Constant prediction (fitting-window global CTR)
    B. Smoothed Item CTR (k=50, unseen items get global CTR)
    C. Smoothed User CTR (k=50, unseen users get global CTR)
    Evaluated on both primary 'val_first' and diagnostic 'val_all'.
    """
    if fold_name not in FOLD_CONFIGS:
        raise ValueError(f"Unknown fold '{fold_name}'")

    meta = build_fold_artifacts(fold_name=fold_name, force=force_build)
    paths = resolve_data_paths()
    folds_dir = paths["folds_dir"]

    events_file = folds_dir / f"{fold_name}_events.parquet"
    val_first_file = folds_dir / f"{fold_name}_val_first.parquet"
    val_all_file = folds_dir / f"{fold_name}_val_all.parquet"
    train_cands_p = paths["train_cands_parquet"].as_posix()
    train_events_p = paths["train_events_parquet"].as_posix()

    logger.info("--- Running Leakage-Safe Baseline Benchmarks for %s (k=%.1f) ---", fold_name, smoothing_k)
    t_start = time.time()

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit = '1GB';")
    con.execute("PRAGMA threads = 2;")

    # 1. Compute fitting-window global CTR
    q_global_ctr = f"""
    SELECT sum(c.clicked) * 1.0 / count(*)
    FROM '{train_cands_p}' c
    JOIN '{events_file.as_posix()}' e ON c.event_id = e.event_id
    WHERE e.split = 'fit'
    """
    fit_global_ctr = float(con.execute(q_global_ctr).fetchone()[0])
    logger.info("Fitting-window Global CTR: %.6f", fit_global_ctr)

    benchmarks: Dict[str, Any] = {
        "fold": fold_name,
        "smoothing_k": smoothing_k,
        "fit_global_ctr": fit_global_ctr,
        "results": {},
    }

    # Iterate over both slices: primary 'val_first' and diagnostic 'val_all'
    for slice_name, val_path in [("val_first", val_first_file), ("val_all", val_all_file)]:
        logger.info("Benchmarking slice: %s (%s)...", slice_name, val_path.name)

        # Baseline A: Constant Prediction
        # Constant predictions rank all pairs equally, ROC AUC mathematically equals 0.5
        gt_df = con.execute(f"SELECT rowid, clicked FROM '{val_path.as_posix()}'").df()
        const_preds = {r: fit_global_ctr for r in gt_df["rowid"]}
        const_res = evaluate_predictions(const_preds, ground_truth=gt_df, fold_name=fold_name, slice_type=slice_name)

        # Baseline B: Smoothed Item CTR
        # Items computed exclusively from fit split; unseen items imputed with fit_global_ctr
        t0 = time.time()
        q_item = f"""
        WITH fit_items AS (
            SELECT
                c.item_id,
                (sum(c.clicked) + {smoothing_k} * {fit_global_ctr}) / (count(*) + {smoothing_k}) AS item_smoothed_ctr
            FROM '{train_cands_p}' c
            JOIN '{events_file.as_posix()}' e ON c.event_id = e.event_id
            WHERE e.split = 'fit'
            GROUP BY c.item_id
        ),
        scored AS (
            SELECT
                v.rowid,
                v.clicked,
                COALESCE(fi.item_smoothed_ctr, {fit_global_ctr}) AS pred
            FROM '{val_path.as_posix()}' v
            LEFT JOIN fit_items fi ON v.item_id = fi.item_id
        )
        SELECT rowid, pred FROM scored
        """
        item_pred_df = con.execute(q_item).df()
        item_res = evaluate_predictions(item_pred_df, ground_truth=gt_df, fold_name=fold_name, slice_type=slice_name)
        item_res["computation_time_seconds"] = round(time.time() - t0, 2)

        # Baseline C: Smoothed User CTR
        # Users computed exclusively from fit split; unseen users imputed with fit_global_ctr
        t0 = time.time()
        q_user = f"""
        WITH fit_users AS (
            SELECT
                te.user_id,
                (sum(c.clicked) + {smoothing_k} * {fit_global_ctr}) / (count(*) + {smoothing_k}) AS user_smoothed_ctr
            FROM '{train_cands_p}' c
            JOIN '{events_file.as_posix()}' fe ON c.event_id = fe.event_id
            JOIN '{train_events_p}' te ON c.event_id = te.event_id
            WHERE fe.split = 'fit'
            GROUP BY te.user_id
        ),
        scored AS (
            SELECT
                v.rowid,
                COALESCE(fu.user_smoothed_ctr, {fit_global_ctr}) AS pred
            FROM '{val_path.as_posix()}' v
            JOIN '{train_events_p}' te ON v.event_id = te.event_id
            LEFT JOIN fit_users fu ON te.user_id = fu.user_id
        )
        SELECT rowid, pred FROM scored
        """
        user_pred_df = con.execute(q_user).df()
        user_res = evaluate_predictions(user_pred_df, ground_truth=gt_df, fold_name=fold_name, slice_type=slice_name)
        user_res["computation_time_seconds"] = round(time.time() - t0, 2)

        benchmarks["results"][slice_name] = {
            "constant_ctr": const_res,
            "item_smoothed_ctr": item_res,
            "user_smoothed_ctr": user_res,
        }

    con.close()
    benchmarks["total_duration_seconds"] = round(time.time() - t_start, 2)
    logger.info("Benchmarks for %s finished in %.2fs", fold_name, benchmarks["total_duration_seconds"])
    return benchmarks


def get_split(
    events_df: Optional[pd.DataFrame] = None,
    fold_name: str = "fold_b",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Backward-compatible interface for obtaining split event IDs.
    If events_df is None, uses precomputed fold artifacts.
    """
    if events_df is None:
        meta = build_fold_artifacts(fold_name=fold_name)
        paths = resolve_data_paths()
        events_file = paths["folds_dir"] / f"{fold_name}_events.parquet"
        con = duckdb.connect()
        fit_ids = [r[0] for r in con.execute(f"SELECT event_id FROM '{events_file.as_posix()}' WHERE split = 'fit'").fetchall()]
        val_all_ids = [r[0] for r in con.execute(f"SELECT event_id FROM '{events_file.as_posix()}' WHERE split IN ('val_first', 'val_later')").fetchall()]
        val_first_ids = [r[0] for r in con.execute(f"SELECT event_id FROM '{events_file.as_posix()}' WHERE split = 'val_first'").fetchall()]
        con.close()
        return fit_ids, val_all_ids, val_first_ids

    # Fallback for in-memory dataframe
    cfg = FOLD_CONFIGS.get(fold_name, FOLD_CONFIGS["fold_b"])
    fit_mask = (events_df["timestamp"] >= cfg["fit_start_ts"]) & (events_df["timestamp"] < cfg["cutoff_ts"])
    fit_event_ids = events_df.loc[fit_mask, "event_id"].unique().tolist()

    val_mask = (events_df["timestamp"] >= cfg["cutoff_ts"]) & (events_df["timestamp"] < cfg["val_end_ts"])
    val_df = events_df[val_mask].sort_values("timestamp")
    val_event_ids = val_df["event_id"].unique().tolist()

    if "user_id" in val_df.columns:
        val_first_df = val_df.groupby("user_id", as_index=False).first()
        val_first_event_ids = val_first_df["event_id"].unique().tolist()
    else:
        val_first_event_ids = val_event_ids

    return fit_event_ids, val_event_ids, val_first_event_ids


def report(
    df: Union[pd.DataFrame, Any],
    val_first_event_ids: Optional[Union[List[str], Set[str]]] = None,
) -> Dict[str, float]:
    """
    Backward-compatible validation reporting interface.
    """
    if hasattr(df, "to_pandas"):
        eval_df = df.to_pandas()
    else:
        eval_df = df.copy()

    # Generate synthetic rowids if caller only supplied event_id, clicked, pred
    if "rowid" not in eval_df.columns:
        eval_df["rowid"] = [f"row_{i}" for i in range(len(eval_df))]

    res_all = evaluate_predictions(eval_df[["rowid", "pred"]], ground_truth=eval_df[["rowid", "clicked"]], slice_type="all")
    if val_first_event_ids is not None:
        first_set = set(val_first_event_ids)
        first_df = eval_df[eval_df["event_id"].isin(first_set)]
        res_first = evaluate_predictions(first_df[["rowid", "pred"]], ground_truth=first_df[["rowid", "clicked"]], slice_type="first")
        return {"auc_first": res_first["auc"], "auc_all": res_all["auc"]}
    return {"auc_first": res_all["auc"], "auc_all": res_all["auc"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal validation framework and leakage-safe benchmarks.")
    parser.add_argument("--fold", choices=["fold_a", "fold_b", "all"], default="all", help="Target validation fold")
    parser.add_argument("--force", action="store_true", help="Rebuild fold artifacts")
    parser.add_argument("--k", type=float, default=50.0, help="Prior smoothing parameter k")
    args = parser.parse_args()

    target_folds = ["fold_a", "fold_b"] if args.fold == "all" else [args.fold]
    for f in target_folds:
        bench = run_baseline_benchmarks(fold_name=f, smoothing_k=args.k, force_build=args.force)
        print(f"\n==================== SUMMARY FOR {f.upper()} ====================")
        print(json.dumps(bench, indent=2))
