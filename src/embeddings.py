"""
Phase 4E: Strictly temporal, leakage-free Truncated SVD latent-factor embeddings and
candidate + within-event latent similarity features for click prediction.

Key design guarantees:
1. Strict temporal boundary (`asof_ts`):
   - Only events with `timestamp < asof_ts` are visible when constructing a historical snapshot.
   - Only positive candidate clicks (`clicked = 1` with `event.timestamp < asof_ts`) and historical
     user click sequences (`hist_ts < event.timestamp < asof_ts`) enter the interaction matrix.
   - On Day 1 (`2025-01-20`, `asof_ts = 1737331200`), zero prior events exist, so an empty
     cold-start snapshot is returned and all candidates receive finite cold-start fallbacks.
2. Spectral delocalization via degree floor (`min_degree = 2.0`):
   - Clamping user/item normalization degrees to `max(deg, 2.0)` suppresses degree-1 leaf-star
     singular vectors that otherwise crowd out global latent factors on sparser cutoffs (Fold A
     and early training days Jan 21-24), ensuring all 32 singular dimensions capture global
     collaborative structure across every training day and both validation folds.
3. Deterministic embeddings:
   - Fixed random seed (`seed = 42`) for the ARPACK starting vector `v0`, descending singular
     value ordering, and deterministic column sign canonicalization.
4. Compact memory footprint:
   - All embeddings and dense feature columns are stored in `np.float32` (ranks in `np.uint8`),
     with snapshots cached in compressed `.npz` archives under `data/processed/features/svd_snapshots/`.
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from src.features import TRAINING_SETS, training_partition_paths
from src.utils import get_logger, get_peak_memory_mb, resolve_data_paths

logger = get_logger("SVDEmbeddings")


def get_svd_feature_cols(k: int = 32, weighting: str = "binary") -> List[str]:
    """Returns the ordered list of 11 SVD feature column names for dimension k and weighting."""
    tag = "_idf" if weighting in ("item_idf", "idf") else ""
    return [
        f"svd_user_item_cos_{k}{tag}",
        f"svd_user_item_dot_{k}{tag}",
        f"svd_history_mean_cos_{k}{tag}",
        f"svd_history_recency_cos_{k}{tag}",
        f"svd_history_last_item_cos_{k}{tag}",
        f"svd_history_max_item_cos_{k}{tag}",
        "we_svd_user_cos_rank",
        "we_svd_user_cos_diff",
        "we_svd_user_cos_ratio",
        "we_svd_history_cos_rank",
        "we_svd_history_cos_diff",
    ]


SVD_FEATURE_COLS: List[str] = get_svd_feature_cols(32, weighting="binary")
SVD_FEATURE_COLS_64: List[str] = get_svd_feature_cols(64, weighting="binary")
SVD_FEATURE_COLS_32_IDF: List[str] = get_svd_feature_cols(32, weighting="item_idf")

WE_SVD_FEATURE_COLS: List[str] = [
    "we_svd_user_cos_rank",
    "we_svd_user_cos_diff",
    "we_svd_user_cos_ratio",
    "we_svd_history_cos_rank",
    "we_svd_history_cos_diff",
]

SVD_FEATURE_TYPES: Dict[str, pa.DataType] = {
    # 32-dim binary features
    "svd_user_item_cos_32": pa.float32(),
    "svd_user_item_dot_32": pa.float32(),
    "svd_history_mean_cos_32": pa.float32(),
    "svd_history_recency_cos_32": pa.float32(),
    "svd_history_last_item_cos_32": pa.float32(),
    "svd_history_max_item_cos_32": pa.float32(),
    # 32-dim item-IDF weighted features
    "svd_user_item_cos_32_idf": pa.float32(),
    "svd_user_item_dot_32_idf": pa.float32(),
    "svd_history_mean_cos_32_idf": pa.float32(),
    "svd_history_recency_cos_32_idf": pa.float32(),
    "svd_history_last_item_cos_32_idf": pa.float32(),
    "svd_history_max_item_cos_32_idf": pa.float32(),
    # 64-dim binary features
    "svd_user_item_cos_64": pa.float32(),
    "svd_user_item_dot_64": pa.float32(),
    "svd_history_mean_cos_64": pa.float32(),
    "svd_history_recency_cos_64": pa.float32(),
    "svd_history_last_item_cos_64": pa.float32(),
    "svd_history_max_item_cos_64": pa.float32(),
    # Within-event features
    "we_svd_user_cos_rank": pa.uint8(),
    "we_svd_user_cos_diff": pa.float32(),
    "we_svd_user_cos_ratio": pa.float32(),
    "we_svd_history_cos_rank": pa.uint8(),
    "we_svd_history_cos_diff": pa.float32(),
}


@dataclass
class SparseSVDSnapshot:
    """
    Holds 32-dimensional user and item latent factor embeddings fitted strictly before `asof_ts`.
    `user_ids` and `item_ids` are sorted 1D string arrays enabling O(log N) vectorized lookup
    via `np.searchsorted`.
    """

    asof_ts: int
    k: int
    min_degree: float
    user_ids: np.ndarray
    item_ids: np.ndarray
    singular_values: np.ndarray
    user_emb: np.ndarray
    item_emb: np.ndarray
    user_emb_norm: np.ndarray
    item_emb_norm: np.ndarray
    stats: Dict[str, Any]
    weighting: str = "binary"

    @property
    def num_users(self) -> int:
        return int(len(self.user_ids))

    @property
    def num_items(self) -> int:
        return int(len(self.item_ids))

    @property
    def is_empty(self) -> bool:
        return self.num_users == 0 or self.num_items == 0


def _empty_svd_snapshot(
    asof_ts: int,
    k: int = 32,
    min_degree: float = 2.0,
    weighting: str = "binary",
) -> SparseSVDSnapshot:
    empty_ids = np.array([], dtype=str)
    empty_s = np.zeros(k, dtype=np.float32)
    empty_emb = np.zeros((0, k), dtype=np.float32)
    stats = {
        "asof_ts": int(asof_ts),
        "k": int(k),
        "effective_k": 0,
        "min_degree": float(min_degree),
        "weighting": weighting,
        "prior_events_count": 0,
        "positive_cands_count": 0,
        "history_clicks_count": 0,
        "num_users": 0,
        "num_items": 0,
        "matrix_nnz": 0,
        "global_dims_ge_2000_users": 0,
        "s_max": 0.0,
        "s_min": 0.0,
        "build_seconds": 0.0,
        "peak_rss_mb": 0.0,
        "cached": False,
    }
    return SparseSVDSnapshot(
        asof_ts=int(asof_ts),
        k=int(k),
        min_degree=float(min_degree),
        user_ids=empty_ids,
        item_ids=empty_ids,
        singular_values=empty_s,
        user_emb=empty_emb,
        item_emb=empty_emb,
        user_emb_norm=empty_emb,
        item_emb_norm=empty_emb,
        stats=stats,
        weighting=weighting,
    )


def _snapshot_cache_paths(
    cache_dir: Path,
    asof_ts: int,
    k: int,
    min_degree: float,
    seed: int,
    weighting: str = "binary",
) -> Tuple[Path, Path]:
    deg_tag = str(min_degree).replace(".", "p")
    w_tag = f"_{weighting}" if weighting not in ("binary", "") else ""
    stem = f"svd_asof_{int(asof_ts)}_k{int(k)}_dmin{deg_tag}_s{int(seed)}{w_tag}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def build_svd_snapshot_asof(
    asof_ts: int,
    train_events_path: Optional[Union[str, Path]] = None,
    train_cands_path: Optional[Union[str, Path]] = None,
    k: int = 32,
    min_degree: float = 2.0,
    seed: int = 42,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    weighting: str = "binary",
) -> SparseSVDSnapshot:
    """
    Constructs (or loads from disk cache) a leakage-free 32-dim Truncated SVD snapshot using
    ONLY positive user-item interactions strictly before `asof_ts`:
    - Events in `train_events.parquet` with `timestamp < asof_ts`
    - Clicked candidates (`clicked = 1`) from events with `timestamp < asof_ts`
    - Historical user clicks (`history_item_ids`, `history_timestamps`) from events with
      `timestamp < asof_ts` where `hist_ts < event_ts < asof_ts`.
    """
    t0 = time.time()
    paths = resolve_data_paths()
    ev_path = Path(train_events_path) if train_events_path is not None else paths["train_events_parquet"]
    cd_path = Path(train_cands_path) if train_cands_path is not None else paths["train_cands_parquet"]

    if cache_dir is not None:
        snap_dir = Path(cache_dir)
    else:
        snap_dir = paths["features_dir"] / "svd_snapshots"

    if use_cache:
        snap_dir.mkdir(parents=True, exist_ok=True)
        npz_path, json_path = _snapshot_cache_paths(snap_dir, asof_ts, k, min_degree, seed, weighting=weighting)
        if npz_path.exists() and json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            snap_weighting = meta.get("weighting", weighting)
            if int(meta.get("num_users", 0)) == 0:
                snap = _empty_svd_snapshot(asof_ts=asof_ts, k=k, min_degree=min_degree, weighting=snap_weighting)
                snap.stats["cached"] = True
                return snap
            with np.load(npz_path, allow_pickle=False) as data:
                user_ids = data["user_ids"].astype(str)
                item_ids = data["item_ids"].astype(str)
                s_svd = data["singular_values"].astype(np.float32)
                user_emb = data["user_emb"].astype(np.float32)
                item_emb = data["item_emb"].astype(np.float32)
            u_norm = np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-8).astype(np.float32)
            i_norm = np.maximum(np.linalg.norm(item_emb, axis=1, keepdims=True), 1e-8).astype(np.float32)
            meta["cached"] = True
            return SparseSVDSnapshot(
                asof_ts=int(asof_ts),
                k=int(k),
                min_degree=float(min_degree),
                user_ids=user_ids,
                item_ids=item_ids,
                singular_values=s_svd,
                user_emb=user_emb,
                item_emb=item_emb,
                user_emb_norm=(user_emb / u_norm).astype(np.float32),
                item_emb_norm=(item_emb / i_norm).astype(np.float32),
                stats=meta,
                weighting=snap_weighting,
            )

    ev_sql = ev_path.as_posix()
    cd_sql = cd_path.as_posix()
    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit = '2GB'; PRAGMA threads = 4;").fetchall()
        n_prior_events = int(
            con.sql(
                f"SELECT count(*) FROM read_parquet('{ev_sql}', hive_partitioning=false) WHERE timestamp < {int(asof_ts)}"
            ).fetchall()[0][0]
        )
        if n_prior_events == 0:
            snap = _empty_svd_snapshot(asof_ts=asof_ts, k=k, min_degree=min_degree, weighting=weighting)
            if use_cache:
                npz_path, json_path = _snapshot_cache_paths(snap_dir, asof_ts, k, min_degree, seed, weighting=weighting)
                json_path.write_text(json.dumps(snap.stats, indent=2), encoding="utf-8")
                np.savez_compressed(
                    npz_path,
                    user_ids=snap.user_ids,
                    item_ids=snap.item_ids,
                    singular_values=snap.singular_values,
                    user_emb=snap.user_emb,
                    item_emb=snap.item_emb,
                )
            return snap

        # 1. All positive clicked candidates from events strictly prior to asof_ts
        pos_df = con.sql(f"""
            SELECT e.user_id, c.item_id
            FROM read_parquet('{cd_sql}', hive_partitioning=false) c
            JOIN read_parquet('{ev_sql}', hive_partitioning=false) e USING (event_id)
            WHERE e.timestamp < {int(asof_ts)} AND c.clicked = 1
        """).df()

        # 2. Earliest and latest event histories per user strictly prior to asof_ts
        ev_hist_rows = con.sql(f"""
            WITH bounds AS (
                SELECT
                    arg_min(event_id, timestamp) AS first_eid,
                    arg_max(event_id, timestamp) AS last_eid
                FROM read_parquet('{ev_sql}', hive_partitioning=false)
                WHERE timestamp < {int(asof_ts)}
                GROUP BY user_id
            ),
            eids AS (
                SELECT first_eid AS event_id FROM bounds
                UNION
                SELECT last_eid AS event_id FROM bounds
            )
            SELECT e.user_id, e.timestamp, e.history_item_ids, e.history_timestamps
            FROM read_parquet('{ev_sql}', hive_partitioning=false) e
            SEMI JOIN eids f ON e.event_id = f.event_id
            WHERE e.timestamp < {int(asof_ts)}
        """).fetchall()
    finally:
        con.close()

    n_pos_cands = len(pos_df)
    u_hist: List[str] = []
    i_hist: List[str] = []
    for uid, ev_ts, hi_s, ht_s in ev_hist_rows:
        if not hi_s or hi_s == "[]":
            continue
        hi = json.loads(hi_s)
        ht = json.loads(ht_s)
        for it, ts in zip(hi, ht):
            if ts < ev_ts and ts < asof_ts:
                u_hist.append(uid)
                i_hist.append(it)
    del ev_hist_rows
    n_hist_clicks = len(u_hist)

    if n_pos_cands + n_hist_clicks == 0:
        return _empty_svd_snapshot(asof_ts=asof_ts, k=k, min_degree=min_degree)

    u_all = np.concatenate([pos_df["user_id"].to_numpy(dtype=str), np.array(u_hist, dtype=str)])
    i_all = np.concatenate([pos_df["item_id"].to_numpy(dtype=str), np.array(i_hist, dtype=str)])
    del pos_df, u_hist, i_hist

    # Deterministic lexicographically sorted vocabularies
    u_uniq, u_idx = np.unique(u_all, return_inverse=True)
    i_uniq, i_idx = np.unique(i_all, return_inverse=True)
    del u_all, i_all

    n_users = len(u_uniq)
    n_items = len(i_uniq)
    eff_k = min(int(k), min(n_users, n_items) - 1)
    if eff_k < 1:
        return _empty_svd_snapshot(asof_ts=asof_ts, k=k, min_degree=min_degree, weighting=weighting)

    R = sp.csr_matrix(
        (np.ones(len(u_idx), dtype=np.float32), (u_idx, i_idx)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    R.sum_duplicates()
    R.data[:] = 1.0
    del u_idx, i_idx

    # Apply optional interaction weighting
    idf_stats: Dict[str, Any] = {}
    if weighting in ("item_idf", "idf"):
        # Predeclared smoothed Item-IDF: idf_i = 1 + log((N + 1) / (df_i + 1))
        # Since R is deduplicated binary, column sum is exactly the distinct users who clicked item i
        df_i = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
        idf_i = (1.0 + np.log((float(n_users) + 1.0) / (df_i + 1.0))).astype(np.float32)
        R.data = idf_i[R.indices]
        idf_stats = {
            "idf_min": round(float(idf_i.min()), 6),
            "idf_max": round(float(idf_i.max()), 6),
            "idf_mean": round(float(idf_i.mean()), 6),
            "idf_median": round(float(np.median(idf_i)), 6),
            "idf_std": round(float(idf_i.std()), 6),
        }

    user_deg = np.asarray(R.sum(axis=1)).ravel()
    item_deg = np.asarray(R.sum(axis=0)).ravel()
    inv_sqrt_u = (1.0 / np.sqrt(np.maximum(user_deg, float(min_degree)))).astype(np.float32)
    inv_sqrt_i = (1.0 / np.sqrt(np.maximum(item_deg, float(min_degree)))).astype(np.float32)
    R_norm = (sp.diags(inv_sqrt_u) @ R @ sp.diags(inv_sqrt_i)).tocsr()

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(R_norm.shape)).astype(np.float32)
    u_svd, s_svd, vt_svd = svds(R_norm, k=eff_k, v0=v0, random_state=seed)
    order = np.argsort(s_svd)[::-1]
    u_svd = u_svd[:, order].astype(np.float32)
    s_svd = s_svd[order].astype(np.float32)
    vt_svd = vt_svd[order, :].astype(np.float32)

    # Deterministic sign canonicalization
    for col_j in range(eff_k):
        idx_max = int(np.argmax(np.abs(vt_svd[col_j, :])))
        if vt_svd[col_j, idx_max] < 0:
            vt_svd[col_j, :] *= -1.0
            u_svd[:, col_j] *= -1.0

    # Pad to k columns with zeros if eff_k < k (e.g. tiny synthetic unit test graphs)
    if eff_k < k:
        u_pad = np.zeros((n_users, k), dtype=np.float32)
        u_pad[:, :eff_k] = u_svd
        u_svd = u_pad
        vt_pad = np.zeros((k, n_items), dtype=np.float32)
        vt_pad[:eff_k, :] = vt_svd
        vt_svd = vt_pad
        s_pad = np.zeros(k, dtype=np.float32)
        s_pad[:eff_k] = s_svd
        s_svd = s_pad

    active_u_per_dim = (np.abs(u_svd) > 1e-3).sum(axis=0)
    n_global_dims = int((active_u_per_dim >= min(2000, max(2, n_users // 5))).sum())

    sqrt_s = np.sqrt(np.maximum(s_svd, 0.0)).astype(np.float32)
    user_emb = (u_svd * sqrt_s[None, :]).astype(np.float32)
    item_emb = (vt_svd.T * sqrt_s[None, :]).astype(np.float32)
    u_norm = np.maximum(np.linalg.norm(user_emb, axis=1, keepdims=True), 1e-8).astype(np.float32)
    i_norm = np.maximum(np.linalg.norm(item_emb, axis=1, keepdims=True), 1e-8).astype(np.float32)
    user_emb_norm = (user_emb / u_norm).astype(np.float32)
    item_emb_norm = (item_emb / i_norm).astype(np.float32)

    build_sec = round(time.time() - t0, 2)
    stats = {
        "asof_ts": int(asof_ts),
        "k": int(k),
        "effective_k": int(eff_k),
        "min_degree": float(min_degree),
        "seed": int(seed),
        "weighting": weighting,
        "prior_events_count": int(n_prior_events),
        "positive_cands_count": int(n_pos_cands),
        "history_clicks_count": int(n_hist_clicks),
        "num_users": int(n_users),
        "num_items": int(n_items),
        "matrix_nnz": int(R.nnz),
        "global_dims_ge_2000_users": int(n_global_dims),
        "s_max": round(float(s_svd[0]), 6),
        "s_min": round(float(s_svd[eff_k - 1]), 6),
        "build_seconds": build_sec,
        "peak_rss_mb": get_peak_memory_mb(),
        "cached": False,
        **idf_stats,
    }

    if use_cache:
        npz_path, json_path = _snapshot_cache_paths(snap_dir, asof_ts, k, min_degree, seed, weighting=weighting)
        tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_npz,
            user_ids=u_uniq,
            item_ids=i_uniq,
            singular_values=s_svd,
            user_emb=user_emb,
            item_emb=item_emb,
        )
        os.replace(tmp_npz, npz_path)
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(
        "Built SparseSVDSnapshot (asof=%d, k=%d, d_min=%.1f, w=%s): users=%d, items=%d, nnz=%d, global_dims=%d/%d in %.2fs",
        asof_ts,
        k,
        min_degree,
        weighting,
        n_users,
        n_items,
        R.nnz,
        n_global_dims,
        k,
        build_sec,
    )
    return SparseSVDSnapshot(
        asof_ts=int(asof_ts),
        k=int(k),
        min_degree=float(min_degree),
        user_ids=u_uniq,
        item_ids=i_uniq,
        singular_values=s_svd,
        user_emb=user_emb,
        item_emb=item_emb,
        user_emb_norm=user_emb_norm,
        item_emb_norm=item_emb_norm,
        stats=stats,
        weighting=weighting,
    )


def compute_within_event_svd_features(
    base_svd_table: Union[pa.Table, pd.DataFrame],
    event_ids: Optional[Union[Sequence[str], np.ndarray, pa.Array, pa.ChunkedArray]] = None,
) -> pa.Table:
    """
    Computes the 5 relative within-event SVD competition features (`we_svd_*`) strictly
    partitioned by `event_id` (10 candidates per event) and returns a complete PyArrow Table
    with `rowid` + all 11 `SVD_FEATURE_COLS`.

    Guarantees:
    - Every `event_id` must have exactly 10 candidates; raises `ValueError` otherwise.
    - Never accesses `clicked`.
    - Canonical ordering inside each `event_id` before window aggregation guarantees row-order
      and `rowid`-renaming invariance.
    - Ties share the same SQL `RANK()` (`1..10`).
    """
    if isinstance(base_svd_table, pd.DataFrame):
        tbl = pa.Table.from_pandas(base_svd_table, preserve_index=False)
    elif isinstance(base_svd_table, pa.Table):
        tbl = base_svd_table
    else:
        raise TypeError("base_svd_table must be a pyarrow.Table or pandas.DataFrame")

    u_col = next((c for c in tbl.column_names if c.startswith("svd_user_item_cos_")), "svd_user_item_cos_32")
    col_suffix = u_col[len("svd_user_item_cos_"):]

    cand_cols = [
        f"svd_user_item_cos_{col_suffix}",
        f"svd_user_item_dot_{col_suffix}",
        f"svd_history_mean_cos_{col_suffix}",
        f"svd_history_recency_cos_{col_suffix}",
        f"svd_history_last_item_cos_{col_suffix}",
        f"svd_history_max_item_cos_{col_suffix}",
    ]
    missing = [c for c in ["rowid"] + cand_cols if c not in tbl.column_names]
    if missing:
        raise ValueError(f"base_svd_table missing required columns: {missing}")

    n_rows = tbl.num_rows
    if n_rows == 0 or n_rows % 10 != 0:
        raise ValueError(f"Within-event SVD requires non-empty table divisible by 10; got {n_rows}")

    if event_ids is not None:
        if isinstance(event_ids, (pa.Array, pa.ChunkedArray)):
            eid_arr = (event_ids.combine_chunks() if isinstance(event_ids, pa.ChunkedArray) else event_ids).cast(
                pa.string()
            )
        else:
            eid_arr = pa.array(event_ids, type=pa.string())
        if len(eid_arr) != n_rows:
            raise ValueError(f"event_ids length ({len(eid_arr)}) != table rows ({n_rows})")
    else:
        if "event_id" not in tbl.column_names:
            raise ValueError("event_ids must be provided when 'event_id' is not in base_svd_table")
        eid_col = tbl.column("event_id")
        eid_arr = (eid_col.combine_chunks() if isinstance(eid_col, pa.ChunkedArray) else eid_col).cast(pa.string())

    rid_col = tbl.column("rowid")
    rid_arr = (rid_col.combine_chunks() if isinstance(rid_col, pa.ChunkedArray) else rid_col).cast(pa.string())
    if rid_arr.null_count > 0 or eid_arr.null_count > 0:
        raise ValueError("Null rowid or event_id in compute_within_event_svd_features")
    if pc.count_distinct(rid_arr).as_py() != n_rows:
        raise ValueError("Duplicate rowids in compute_within_event_svd_features")

    h_col = f"svd_history_mean_cos_{col_suffix}"
    u_cos = tbl.column(u_col).combine_chunks().cast(pa.float32())
    h_cos = tbl.column(h_col).combine_chunks().cast(pa.float32())

    work_tbl = pa.Table.from_arrays(
        [
            pa.array(np.arange(n_rows, dtype=np.int64), type=pa.int64()),
            rid_arr,
            eid_arr,
            u_cos,
            h_cos,
        ],
        names=["row_idx", "rowid", "event_id", "u_cos", "h_cos"],
    )

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit = '2GB'; PRAGMA threads = 4;").fetchall()
        con.register("svd_inp_tbl", work_tbl)
        bad_events = con.sql("""
            SELECT count(*)
            FROM (
                SELECT event_id
                FROM svd_inp_tbl
                GROUP BY event_id
                HAVING count(*) != 10
            )
        """).fetchall()[0][0]
        if int(bad_events) != 0:
            raise ValueError(f"Every event_id must contain 10 candidates; found {bad_events} invalid event(s)")

        we_tbl = con.sql("""
            WITH canonical AS (
                SELECT
                    row_idx,
                    rowid,
                    event_id,
                    u_cos,
                    h_cos
                FROM svd_inp_tbl
                ORDER BY
                    event_id ASC,
                    u_cos DESC,
                    h_cos DESC
            ),
            win AS (
                SELECT
                    row_idx,
                    u_cos,
                    h_cos,
                    min(u_cos) OVER (PARTITION BY event_id) AS ev_min_u,
                    max(u_cos) OVER (PARTITION BY event_id) AS ev_max_u,
                    avg(CAST(u_cos AS DOUBLE)) OVER (PARTITION BY event_id) AS ev_avg_u,
                    min(h_cos) OVER (PARTITION BY event_id) AS ev_min_h,
                    max(h_cos) OVER (PARTITION BY event_id) AS ev_max_h,
                    avg(CAST(h_cos AS DOUBLE)) OVER (PARTITION BY event_id) AS ev_avg_h,
                    RANK() OVER (PARTITION BY event_id ORDER BY u_cos DESC) AS r_u,
                    RANK() OVER (PARTITION BY event_id ORDER BY h_cos DESC) AS r_h
                FROM canonical
            )
            SELECT
                CAST(r_u AS UTINYINT) AS we_svd_user_cos_rank,
                CAST(
                    CASE
                        WHEN ev_min_u = ev_max_u THEN 0.0
                        ELSE CAST(u_cos AS DOUBLE) - ev_avg_u
                    END AS FLOAT
                ) AS we_svd_user_cos_diff,
                CAST(
                    CASE
                        WHEN ev_max_u <= 1e-6 THEN 0.0
                        ELSE least(1.0, greatest(-1.0, CAST(u_cos AS DOUBLE) / CAST(ev_max_u AS DOUBLE)))
                    END AS FLOAT
                ) AS we_svd_user_cos_ratio,
                CAST(r_h AS UTINYINT) AS we_svd_history_cos_rank,
                CAST(
                    CASE
                        WHEN ev_min_h = ev_max_h THEN 0.0
                        ELSE CAST(h_cos AS DOUBLE) - ev_avg_h
                    END AS FLOAT
                ) AS we_svd_history_cos_diff
            FROM win
            ORDER BY row_idx ASC
        """).to_arrow_table()
        con.unregister("svd_inp_tbl")
    finally:
        con.close()

    cand_cols = [c for c in tbl.column_names if c not in ("rowid", "event_id")]
    target_svd_cols = cand_cols + WE_SVD_FEATURE_COLS

    out_fields = [pa.field("rowid", pa.string())] + [
        pa.field(col, SVD_FEATURE_TYPES.get(col, pa.float32() if not col.endswith("_rank") else pa.uint8()))
        for col in target_svd_cols
    ]
    out_arrays = [rid_arr]
    for col in cand_cols:
        col_type = SVD_FEATURE_TYPES.get(col, pa.float32())
        arr = tbl.column(col).combine_chunks().cast(col_type)
        if arr.null_count > 0 or not np.isfinite(arr.to_numpy(zero_copy_only=False)).all():
            raise ValueError(f"Null or non-finite values in column '{col}'")
        out_arrays.append(arr)

    for col in target_svd_cols[len(cand_cols) :]:
        col_type = SVD_FEATURE_TYPES.get(col, pa.float32() if not col.endswith("_rank") else pa.uint8())
        arr = we_tbl.column(col).combine_chunks().cast(col_type)
        if arr.null_count > 0 or not np.isfinite(arr.to_numpy(zero_copy_only=False).astype(np.float64)).all():
            raise ValueError(f"Null or non-finite values in within-event SVD column '{col}'")
        out_arrays.append(arr)

    return pa.Table.from_arrays(out_arrays, schema=pa.schema(out_fields))


def score_candidates_with_svd(
    snapshot: SparseSVDSnapshot,
    rowids: Sequence[str],
    event_ids: Sequence[str],
    item_ids: Sequence[str],
    event_histories: Dict[str, Tuple[str, int, str, str]],
    recency_decay: float = 0.85,
    max_history_items: int = 50,
) -> pa.Table:
    """
    Vectorized computation of all 11 `SVD_FEATURE_COLS` for a batch of candidate rows.

    Args:
        snapshot: `SparseSVDSnapshot` fitted strictly before the scored window (`timestamp < asof_ts`).
        rowids: 1D sequence of candidate `rowid` strings (length N, divisible by 10).
        event_ids: 1D sequence of candidate `event_id` strings (length N).
        item_ids: 1D sequence of candidate `item_id` strings (length N).
        event_histories: Mapping `event_id -> (user_id, event_ts, history_item_ids_json, history_timestamps_json)`.
        recency_decay: Exponential decay factor (default `0.85`) per click rank from most recent (`rank=0`).
        max_history_items: Maximum number of most recent valid history clicks (`hist_ts < event_ts`) inspected per event.

    Returns:
        PyArrow Table with `rowid` + the 11 `SVD_FEATURE_COLS`.
    """
    n_rows = len(rowids)
    if n_rows == 0 or n_rows % 10 != 0:
        raise ValueError(f"score_candidates_with_svd requires non-empty candidate batch divisible by 10; got {n_rows}")
    if len(event_ids) != n_rows or len(item_ids) != n_rows:
        raise ValueError("rowids, event_ids, and item_ids must have identical length")

    u_cos = np.zeros(n_rows, dtype=np.float32)
    u_dot = np.zeros(n_rows, dtype=np.float32)
    h_mean_cos = np.zeros(n_rows, dtype=np.float32)
    h_rec_cos = np.zeros(n_rows, dtype=np.float32)
    h_last_cos = np.zeros(n_rows, dtype=np.float32)
    h_max_cos = np.zeros(n_rows, dtype=np.float32)

    if not snapshot.is_empty:
        u_vocab = snapshot.user_ids
        i_vocab = snapshot.item_ids
        n_u = len(u_vocab)
        n_i = len(i_vocab)
        k_dim = snapshot.k

        # Map unique events to integer indices 0..E-1
        eids_arr = np.asarray(event_ids, dtype=object)
        uniq_eids, cand_ev_idx = np.unique(eids_arr, return_inverse=True)
        n_events = len(uniq_eids)

        # Look up candidate item indices in i_vocab
        c_items_arr = np.asarray(item_ids, dtype=object)
        c_pos = np.searchsorted(i_vocab, c_items_arr)
        c_ok = (c_pos < n_i) & (i_vocab[np.minimum(c_pos, n_i - 1)] == c_items_arr)
        c_idx = np.where(c_ok, c_pos, -1).astype(np.int32)

        # Parse per-event user_id and valid historical item sequences (strictly hist_ts < event_ts)
        ev_users = np.empty(n_events, dtype=object)
        hist_padded = np.full((n_events, max_history_items), -1, dtype=np.int32)
        hist_lengths = np.zeros(n_events, dtype=np.int32)

        # Collect all raw history strings first to batch-lookup item IDs in i_vocab
        flat_hist_items: List[str] = []
        ev_slice_bounds: List[Tuple[int, int, int]] = []
        for e_i, eid in enumerate(uniq_eids):
            info = event_histories.get(eid)
            if info is None:
                ev_users[e_i] = ""
                continue
            uid, ev_ts, hi_s, ht_s = info
            ev_users[e_i] = uid
            if not hi_s or hi_s == "[]":
                continue
            hi = json.loads(hi_s)
            ht = json.loads(ht_s)
            # Filter strictly prior to current event timestamp (`ts < ev_ts`), ordered most-recent-first
            valid_pairs = [(ts, idx_pos, it) for idx_pos, (it, ts) in enumerate(zip(hi, ht)) if ts < ev_ts]
            if not valid_pairs:
                continue
            valid_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if max_history_items and len(valid_pairs) > max_history_items:
                valid_pairs = valid_pairs[:max_history_items]
            start_f = len(flat_hist_items)
            flat_hist_items.extend(x[2] for x in valid_pairs)
            ev_slice_bounds.append((e_i, start_f, len(flat_hist_items)))

        # Vectorized lookup of event user_ids
        ev_u_pos = np.searchsorted(u_vocab, ev_users)
        ev_u_ok = (ev_u_pos < n_u) & (u_vocab[np.minimum(ev_u_pos, n_u - 1)] == ev_users)
        cand_u_pos = ev_u_pos[cand_ev_idx]
        cand_u_ok = ev_u_ok[cand_ev_idx]
        both_ui = cand_u_ok & c_ok
        if np.any(both_ui):
            u_idx_valid = cand_u_pos[both_ui]
            i_idx_valid = c_pos[both_ui]
            u_cos[both_ui] = np.sum(
                snapshot.user_emb_norm[u_idx_valid] * snapshot.item_emb_norm[i_idx_valid],
                axis=1,
            )
            u_dot[both_ui] = np.sum(
                snapshot.user_emb[u_idx_valid] * snapshot.item_emb[i_idx_valid],
                axis=1,
            )

        # Vectorized lookup of all history items and construction of event history vectors
        if flat_hist_items:
            flat_arr = np.asarray(flat_hist_items, dtype=object)
            h_pos = np.searchsorted(i_vocab, flat_arr)
            h_ok = (h_pos < n_i) & (i_vocab[np.minimum(h_pos, n_i - 1)] == flat_arr)
            del flat_arr, flat_hist_items

            max_len_seen = 0
            for e_i, s_f, e_f in ev_slice_bounds:
                valid_h = h_pos[s_f:e_f][h_ok[s_f:e_f]]
                L = len(valid_h)
                if L > 0:
                    if L > max_history_items:
                        valid_h = valid_h[:max_history_items]
                        L = max_history_items
                    hist_padded[e_i, :L] = valid_h
                    hist_lengths[e_i] = L
                    if L > max_len_seen:
                        max_len_seen = L

            if max_len_seen > 0:
                ev_mean_vec = np.zeros((n_events, k_dim), dtype=np.float32)
                ev_rec_vec = np.zeros((n_events, k_dim), dtype=np.float32)
                decay_weights = (recency_decay ** np.arange(max_len_seen, dtype=np.float32)).astype(np.float32)

                # Accumulate event history vectors & per-candidate max cosine across ranks 0..max_len_seen-1
                cand_hist_len = hist_lengths[cand_ev_idx]
                h_max_Tracker = np.full(n_rows, -2.0, dtype=np.float32)

                for r in range(max_len_seen):
                    ev_active = np.where(hist_lengths > r)[0]
                    h_item_r = hist_padded[ev_active, r]
                    emb_r = snapshot.item_emb_norm[h_item_r]
                    ev_mean_vec[ev_active] += emb_r
                    ev_rec_vec[ev_active] += decay_weights[r] * emb_r

                    cand_active = np.where((cand_hist_len > r) & c_ok)[0]
                    if len(cand_active) > 0:
                        c_ev_r = cand_ev_idx[cand_active]
                        c_it_r = c_idx[cand_active]
                        h_it_r = hist_padded[c_ev_r, r]
                        sim_r = np.sum(
                            snapshot.item_emb_norm[h_it_r] * snapshot.item_emb_norm[c_it_r],
                            axis=1,
                        )
                        if r == 0:
                            h_last_cos[cand_active] = sim_r
                        h_max_Tracker[cand_active] = np.maximum(h_max_Tracker[cand_active], sim_r)

                has_hist_and_c = (cand_hist_len > 0) & c_ok
                if np.any(has_hist_and_c):
                    h_max_cos[has_hist_and_c] = h_max_Tracker[has_hist_and_c]

                    ev_mean_norm = ev_mean_vec / np.maximum(
                        np.linalg.norm(ev_mean_vec, axis=1, keepdims=True), 1e-8
                    )
                    ev_rec_norm = ev_rec_vec / np.maximum(
                        np.linalg.norm(ev_rec_vec, axis=1, keepdims=True), 1e-8
                    )
                    ev_idx_valid = cand_ev_idx[has_hist_and_c]
                    c_idx_valid = c_idx[has_hist_and_c]
                    c_emb_valid = snapshot.item_emb_norm[c_idx_valid]
                    h_mean_cos[has_hist_and_c] = np.sum(ev_mean_norm[ev_idx_valid] * c_emb_valid, axis=1)
                    h_rec_cos[has_hist_and_c] = np.sum(ev_rec_norm[ev_idx_valid] * c_emb_valid, axis=1)

    k_dim = 64 if (snapshot is not None and snapshot.k == 64) else 32
    w_tag = "_idf" if (snapshot is not None and getattr(snapshot, "weighting", "binary") in ("item_idf", "idf")) else ""
    col_suffix = f"{k_dim}{w_tag}"
    base_tbl = pa.Table.from_arrays(
        [
            pa.array(rowids, type=pa.string()),
            pa.array(u_cos, type=pa.float32()),
            pa.array(u_dot, type=pa.float32()),
            pa.array(h_mean_cos, type=pa.float32()),
            pa.array(h_rec_cos, type=pa.float32()),
            pa.array(h_last_cos, type=pa.float32()),
            pa.array(h_max_cos, type=pa.float32()),
        ],
        names=[
            "rowid",
            f"svd_user_item_cos_{col_suffix}",
            f"svd_user_item_dot_{col_suffix}",
            f"svd_history_mean_cos_{col_suffix}",
            f"svd_history_recency_cos_{col_suffix}",
            f"svd_history_last_item_cos_{col_suffix}",
            f"svd_history_max_item_cos_{col_suffix}",
        ],
    )
    return compute_within_event_svd_features(base_tbl, event_ids=event_ids)


def _fetch_event_histories_and_items_for_svd(
    ev_path: Path,
    cd_path: Path,
    event_ids: np.ndarray,
) -> Tuple[Dict[str, Tuple[str, int, str, str]], Dict[str, str]]:
    """
    Fetches `event_id -> (user_id, event_ts, history_item_ids, history_timestamps)` and
    `rowid -> item_id` for the requested `event_ids` using a short-lived DuckDB connection.
    Never reads `clicked`.
    """
    ev_sql = ev_path.as_posix()
    cd_sql = cd_path.as_posix()
    eids_df = pd.DataFrame({"event_id": np.unique(np.asarray(event_ids, dtype=object))})
    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit = '1.5GB'; PRAGMA threads = 4;").fetchall()
        con.register("q_svd_eids_tbl", eids_df)
        ev_rows = con.sql(f"""
            SELECT e.event_id, e.user_id, e.timestamp, e.history_item_ids, e.history_timestamps
            FROM read_parquet('{ev_sql}', hive_partitioning=false) e
            SEMI JOIN q_svd_eids_tbl d ON e.event_id = d.event_id
        """).fetchall()
        cand_rows = con.sql(f"""
            SELECT c.rowid, c.item_id
            FROM read_parquet('{cd_sql}', hive_partitioning=false) c
            SEMI JOIN q_svd_eids_tbl d ON c.event_id = d.event_id
        """).fetchall()
        con.unregister("q_svd_eids_tbl")
    finally:
        con.close()
    del eids_df
    ev_map = {eid: (uid, int(ts), hi_s, ht_s) for eid, uid, ts, hi_s, ht_s in ev_rows}
    return ev_map, dict(cand_rows)


def evaluate_svd_on_validation_slice(
    fold_name: str = "fold_b",
    k: int = 32,
    min_degree: float = 2.0,
    seed: int = 42,
    use_cache: bool = True,
    save_val_parquet: bool = False,
    weighting: str = "binary",
) -> Tuple[pa.Table, Dict[str, Any]]:
    """
    Builds (or loads cached) `SparseSVDSnapshot` strictly before `TRAINING_SETS[fold_name]['val_day_start']`,
    scores the primary validation slice (`fold_a_val_features.parquet` or `fold_b_val_features.parquet`),
    and computes standalone ROC AUC diagnostics (Global, Exact-Match, Non-Exact-Match, Cold-User, Warm-User).
    """
    from sklearn.metrics import roc_auc_score

    if fold_name not in TRAINING_SETS:
        raise ValueError(f"Unknown fold '{fold_name}'. Expected one of {list(TRAINING_SETS.keys())}")

    t0 = time.time()
    paths = resolve_data_paths()
    feat_dir = paths["features_dir"]
    ev_path = paths["train_events_parquet"]
    cd_path = paths["train_cands_parquet"]

    val_cutoff_ts = int(TRAINING_SETS[fold_name]["val_day_start"])
    val_slice_path = feat_dir / TRAINING_SETS[fold_name]["val_slice"]
    if not val_slice_path.exists():
        raise FileNotFoundError(f"Validation slice not found: {val_slice_path}")

    snap = build_svd_snapshot_asof(
        asof_ts=val_cutoff_ts,
        train_events_path=ev_path,
        train_cands_path=cd_path,
        k=k,
        min_degree=min_degree,
        seed=seed,
        use_cache=use_cache,
        weighting=weighting,
    )

    val_meta = pq.read_table(
        val_slice_path,
        columns=["rowid", "event_id", "clicked", "h_candidate_in_history", "u_is_unseen_in_fit"],
    )
    val_rids = val_meta.column("rowid").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    val_eids = val_meta.column("event_id").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    y_val = val_meta.column("clicked").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)
    in_hist = val_meta.column("h_candidate_in_history").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)
    cold_user = val_meta.column("u_is_unseen_in_fit").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)
    del val_meta

    t_score = time.time()
    ev_map, rid_to_item = _fetch_event_histories_and_items_for_svd(ev_path, cd_path, val_eids)
    val_items = [rid_to_item[r] for r in val_rids]
    val_svd_tbl = score_candidates_with_svd(
        snapshot=snap,
        rowids=val_rids,
        event_ids=val_eids,
        item_ids=val_items,
        event_histories=ev_map,
    )
    score_sec = round(time.time() - t_score, 2)
    del ev_map, rid_to_item, val_items

    if save_val_parquet:
        w_suffix = f"_{weighting}" if weighting not in ("binary", "") else ""
        if k == 32 and weighting == "binary":
            val_name = f"{fold_name}_val_svd_features.parquet"
        else:
            val_name = f"{fold_name}_val_svd{k}{w_suffix}_features.parquet"
        val_out_path = feat_dir / val_name
        tmp_val = val_out_path.with_name(val_out_path.name + ".tmp")
        pq.write_table(val_svd_tbl, tmp_val, compression="zstd")
        os.replace(tmp_val, val_out_path)

    def _sub_aucs(scores: np.ndarray) -> Dict[str, float]:
        ex = in_hist == 1
        non_ex = in_hist == 0
        cold = cold_user == 1
        warm = cold_user == 0
        return {
            "global_auc": round(float(roc_auc_score(y_val, scores)), 6),
            "exact_match_auc": round(float(roc_auc_score(y_val[ex], scores[ex])), 6) if ex.sum() else 0.5,
            "non_exact_match_auc": round(float(roc_auc_score(y_val[non_ex], scores[non_ex])), 6) if non_ex.sum() else 0.5,
            "cold_user_auc": round(float(roc_auc_score(y_val[cold], scores[cold])), 6) if cold.sum() else 0.5,
            "warm_user_auc": round(float(roc_auc_score(y_val[warm], scores[warm])), 6) if warm.sum() else 0.5,
        }

    col_suffix = f"{k}_idf" if weighting in ("item_idf", "idf") else f"{k}"
    feature_aucs: Dict[str, Any] = {}
    eval_cols = [
        f"svd_user_item_cos_{col_suffix}",
        f"svd_user_item_dot_{col_suffix}",
        f"svd_history_mean_cos_{col_suffix}",
        f"svd_history_recency_cos_{col_suffix}",
        f"svd_history_last_item_cos_{col_suffix}",
        f"svd_history_max_item_cos_{col_suffix}",
        "we_svd_user_cos_diff",
        "we_svd_history_cos_diff",
    ]
    for col in eval_cols:
        arr = val_svd_tbl.column(col).to_numpy(zero_copy_only=False).astype(np.float64)
        feature_aucs[col] = _sub_aucs(arr)

    u_cos_arr = val_svd_tbl.column(f"svd_user_item_cos_{col_suffix}").to_numpy(zero_copy_only=False)
    h_mean_arr = val_svd_tbl.column(f"svd_history_mean_cos_{col_suffix}").to_numpy(zero_copy_only=False)
    h_rec_arr = val_svd_tbl.column(f"svd_history_recency_cos_{col_suffix}").to_numpy(zero_copy_only=False)
    blend_arr = 0.55 * u_cos_arr + 0.45 * h_rec_arr
    feature_aucs[f"blend_0.55_user_cos_0.45_hist_rec_cos"] = _sub_aucs(blend_arr)

    summary = {
        "fold": fold_name,
        "asof_ts": val_cutoff_ts,
        "weighting": weighting,
        "val_rows": int(len(val_rids)),
        "snapshot_stats": snap.stats,
        "coverage": {
            "nonzero_user_item_cos_pct": round(float((u_cos_arr != 0.0).mean() * 100.0), 4),
            "nonzero_history_mean_cos_pct": round(float((h_mean_arr != 0.0).mean() * 100.0), 4),
            "nonzero_either_pct": round(float(((u_cos_arr != 0.0) | (h_mean_arr != 0.0)).mean() * 100.0), 4),
        },
        "standalone_feature_aucs": feature_aucs,
        "score_seconds": score_sec,
        "total_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": get_peak_memory_mb(),
    }
    return val_svd_tbl, summary


def build_fold_svd_artifacts(
    fold_name: str = "fold_b",
    target_train_rows: int = 3_000_000,
    seed: int = 42,
    k: int = 32,
    min_degree: float = 2.0,
    use_cache: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Phase 4E Stage F & G: Builds leakage-free SVD latent feature artifacts for:
    1. The exact sampled training rows of `fold_name` (e.g. 3,000,000 rows / 300,000 events for `fold_b`,
       matching `phase4c` and `phase4d` 1:1 by `rowid`), where each day `D` is scored using ONLY
       the `SparseSVDSnapshot` fitted strictly before `start_of(D)`.
    2. The full primary validation slice (`fold_b_val_features.parquet`, 351,380 rows), scored using
       the `SparseSVDSnapshot` fitted strictly before `2025-01-26 00:00 UTC` (`1737849600`).

    Saves:
    - `data/processed/features/<fold>_train_<suffix>_svd_features.parquet`
    - `data/processed/features/<fold>_val_svd_features.parquet`
    - `data/processed/features/svd_manifest_<fold>_<suffix>.json`
    """
    import shutil
    from src.models.lgbm_model import select_fold_sample_events, training_sample_suffix

    t0 = time.time()
    paths = resolve_data_paths()
    feat_dir = paths["features_dir"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    ev_path = paths["train_events_parquet"]
    cd_path = paths["train_cands_parquet"]

    suffix = training_sample_suffix(target_train_rows)
    train_svd_path = feat_dir / f"{fold_name}_train_{suffix}_svd_features.parquet"
    val_svd_path = feat_dir / f"{fold_name}_val_svd_features.parquet"
    manifest_path = feat_dir / f"svd_manifest_{fold_name}_{suffix}.json"

    if train_svd_path.exists() and val_svd_path.exists() and manifest_path.exists() and not force:
        logger.info("Phase 4E SVD artifacts already exist (%s). Loading manifest.", manifest_path.name)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    sel = select_fold_sample_events(fold_name, target_train_rows, seed, feat_dir)
    day_starts = TRAINING_SETS[fold_name]["day_starts"]
    tmp_dir = feat_dir / f"_tmp_svd_{fold_name}_{suffix}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    day_cache_paths: List[Path] = []
    per_day_stats: List[Dict[str, Any]] = []
    snapshot_stats_by_day: Dict[str, Any] = {}

    for day_idx, ((part_path, day_str, _avail_ev, _avail_rows), day_ts) in enumerate(
        zip(sel["day_totals"], day_starts)
    ):
        day_pq = tmp_dir / f"day_{day_str}.parquet"
        day_meta_json = tmp_dir / f"day_{day_str}.json"
        if day_pq.exists() and day_meta_json.exists() and not force:
            cached = json.loads(day_meta_json.read_text(encoding="utf-8"))
            per_day_stats.append(cached["per_day"])
            snapshot_stats_by_day[day_str] = cached["snapshot_stats"]
            day_cache_paths.append(day_pq)
            logger.info("%s: loaded cached daily SVD partition (%d rows)", day_str, cached["per_day"]["rows"])
            continue

        t_day = time.time()
        chosen_ev = sel["chosen_events"][day_idx]
        meta = pq.read_table(part_path, columns=["rowid", "event_id"])
        if chosen_ev is not None:
            meta = meta.filter(pc.is_in(meta.column("event_id"), value_set=pa.array(chosen_ev, type=pa.string())))
        day_rids = meta.column("rowid").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        day_eids = meta.column("event_id").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
        del meta

        snap = build_svd_snapshot_asof(
            asof_ts=int(day_ts),
            train_events_path=ev_path,
            train_cands_path=cd_path,
            k=k,
            min_degree=min_degree,
            seed=seed,
            use_cache=use_cache,
        )
        snapshot_stats_by_day[day_str] = snap.stats

        if snap.is_empty:
            empty_items = [""] * len(day_rids)
            day_tbl = score_candidates_with_svd(
                snapshot=snap,
                rowids=day_rids,
                event_ids=day_eids,
                item_ids=empty_items,
                event_histories={},
            )
        else:
            ev_map, rid_to_item = _fetch_event_histories_and_items_for_svd(ev_path, cd_path, day_eids)
            day_items = [rid_to_item[r] for r in day_rids]
            day_tbl = score_candidates_with_svd(
                snapshot=snap,
                rowids=day_rids,
                event_ids=day_eids,
                item_ids=day_items,
                event_histories=ev_map,
            )
            del ev_map, rid_to_item, day_items

        u_cos_day = day_tbl.column("svd_user_item_cos_32").to_numpy(zero_copy_only=False)
        h_cos_day = day_tbl.column("svd_history_mean_cos_32").to_numpy(zero_copy_only=False)
        day_info = {
            "day": day_str,
            "asof_ts": int(day_ts),
            "events": int(len(day_rids) // 10),
            "rows": int(len(day_rids)),
            "nonzero_user_item_cos_pct": round(float((u_cos_day != 0.0).mean() * 100.0), 4),
            "nonzero_history_mean_cos_pct": round(float((h_cos_day != 0.0).mean() * 100.0), 4),
            "seconds": round(time.time() - t_day, 2),
            "peak_rss_mb": get_peak_memory_mb(),
        }
        per_day_stats.append(day_info)

        tmp_pq = day_pq.with_name(day_pq.name + ".tmp")
        pq.write_table(day_tbl, tmp_pq, compression="zstd")
        os.replace(tmp_pq, day_pq)
        day_meta_json.write_text(
            json.dumps({"per_day": day_info, "snapshot_stats": snap.stats}, indent=2),
            encoding="utf-8",
        )
        day_cache_paths.append(day_pq)
        logger.info(
            "%s: scored %d SVD rows (user_cos_cov=%.2f%%, hist_cos_cov=%.2f%%) in %.2fs (RSS=%s MB)",
            day_str,
            day_info["rows"],
            day_info["nonzero_user_item_cos_pct"],
            day_info["nonzero_history_mean_cos_pct"],
            day_info["seconds"],
            day_info["peak_rss_mb"],
        )
        del day_rids, day_eids, day_tbl, snap
        gc.collect()

    train_svd_tbl = pa.concat_tables([pq.read_table(p) for p in day_cache_paths])
    if train_svd_tbl.num_rows != target_train_rows:
        raise RuntimeError(f"Train SVD row count ({train_svd_tbl.num_rows}) != expected ({target_train_rows})")
    if pc.count_distinct(train_svd_tbl.column("rowid")).as_py() != target_train_rows:
        raise RuntimeError("Duplicate rowids in train SVD feature table")

    # Verify exact 1:1 rowid order against existing CF artifact if present
    ref_cf_path = feat_dir / f"{fold_name}_train_{suffix}_cf_features.parquet"
    if ref_cf_path.exists():
        ref_rids = pq.read_table(ref_cf_path, columns=["rowid"]).column("rowid").combine_chunks()
        if not train_svd_tbl.column("rowid").combine_chunks().equals(ref_rids):
            raise RuntimeError(f"Train SVD rowid order mismatch vs {ref_cf_path.name}")
        del ref_rids

    for col in SVD_FEATURE_COLS:
        c_arr = train_svd_tbl.column(col)
        if c_arr.null_count > 0 or not np.isfinite(c_arr.to_numpy(zero_copy_only=False).astype(np.float64)).all():
            raise RuntimeError(f"Null or non-finite values in train SVD column '{col}'")

    tmp_tr = train_svd_path.with_name(train_svd_path.name + ".tmp")
    pq.write_table(train_svd_tbl, tmp_tr, compression="zstd")
    os.replace(tmp_tr, train_svd_path)
    del train_svd_tbl
    gc.collect()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Score validation slice and save `<fold>_val_svd_features.parquet`
    val_svd_tbl, val_summary = evaluate_svd_on_validation_slice(
        fold_name=fold_name,
        k=k,
        min_degree=min_degree,
        seed=seed,
        use_cache=use_cache,
        save_val_parquet=True,
    )
    del val_svd_tbl
    gc.collect()

    manifest = {
        "fold": fold_name,
        "seed": int(seed),
        "k": int(k),
        "min_degree": float(min_degree),
        "target_train_rows": int(target_train_rows),
        "svd_feature_cols": SVD_FEATURE_COLS,
        "train_svd_path": str(train_svd_path),
        "train_svd_rows": int(target_train_rows),
        "train_svd_size_mb": round(train_svd_path.stat().st_size / (1024 ** 2), 2),
        "val_svd_path": str(val_svd_path),
        "val_svd_rows": int(val_summary["val_rows"]),
        "val_svd_size_mb": round(val_svd_path.stat().st_size / (1024 ** 2), 2),
        "per_day": per_day_stats,
        "snapshot_stats_by_day": snapshot_stats_by_day,
        "validation_summary": val_summary,
        "total_build_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": get_peak_memory_mb(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Built Phase 4E SVD artifacts (%s: %.2f MB, %s: %.2f MB) in %.2fs (Peak RSS=%s MB)",
        train_svd_path.name,
        manifest["train_svd_size_mb"],
        val_svd_path.name,
        manifest["val_svd_size_mb"],
        manifest["total_build_seconds"],
        manifest["peak_rss_mb"],
    )
    return manifest


def build_eval_svd_artifacts(
    k: int = 32,
    min_degree: float = 2.0,
    seed: int = 42,
    use_cache: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Phase 4E Official Submission: Builds the 11 SVD latent-factor features (`eval_svd_features.parquet`)
    for all 529,390 official evaluation candidates (`eval_features.parquet`):
    - Fits (or loads from disk cache) the 32-dim `SparseSVDSnapshot` (`min_degree=2.0`, `seed=42`)
      strictly before `2025-01-27 00:00:00 UTC` (`asof_ts = 1737936000`), using ONLY legitimate
      Jan 20-26 training period positive interactions and pre-cutoff history clicks.
    - Scores evaluation events (`Jan 27 - Feb 2`) using each event's supplied history ONLY where
      `hist_ts < event_ts`.
    - Never accesses any evaluation labels or future outcomes.
    - Preserves exact 1:1 `rowid` order with `eval_features.parquet` (`529,390` rows).
    """
    from src.features import TS_JAN_26_0000, TS_JAN_27_0000

    t0 = time.time()
    paths = resolve_data_paths()
    feat_dir = paths["features_dir"]
    tr_ev_path = paths["train_events_parquet"]
    tr_cd_path = paths["train_cands_parquet"]
    ev_ev_path = paths["eval_events_parquet"]
    ev_cd_path = paths["eval_cands_parquet"]

    eval_base_path = feat_dir / "eval_features.parquet"
    eval_svd_path = feat_dir / "eval_svd_features.parquet"
    manifest_path = feat_dir / "eval_svd_manifest.json"

    if eval_svd_path.exists() and manifest_path.exists() and not force:
        logger.info("Evaluation SVD artifact already exists (%s). Loading manifest.", manifest_path.name)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if not eval_base_path.exists():
        raise FileNotFoundError(f"Base evaluation feature file not found: {eval_base_path}")

    # 1. Build (or load cached) historical SVD snapshot at 2025-01-27 00:00:00 UTC (asof_ts = 1737936000)
    eval_snap = build_svd_snapshot_asof(
        asof_ts=TS_JAN_27_0000,
        train_events_path=tr_ev_path,
        train_cands_path=tr_cd_path,
        k=k,
        min_degree=min_degree,
        seed=seed,
        use_cache=use_cache,
    )

    # 2. Load evaluation rowid/event_id order and diagnostic flags
    eval_meta_tbl = pq.read_table(
        eval_base_path,
        columns=["rowid", "event_id", "h_candidate_in_history", "u_is_unseen_in_fit"],
    )
    eval_rids = eval_meta_tbl.column("rowid").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    eval_eids = eval_meta_tbl.column("event_id").combine_chunks().to_numpy(zero_copy_only=False).astype(str)
    eval_in_hist = eval_meta_tbl.column("h_candidate_in_history").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)
    eval_cold_user = eval_meta_tbl.column("u_is_unseen_in_fit").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint8)
    del eval_meta_tbl

    n_eval_rows = len(eval_rids)
    if n_eval_rows != 529_390:
        raise RuntimeError(f"Expected 529,390 evaluation rows in eval_features.parquet, got {n_eval_rows}")

    t_score = time.time()
    ev_ev_tbl = pq.read_table(
        ev_ev_path,
        columns=["event_id", "user_id", "timestamp", "history_item_ids", "history_timestamps"],
    )
    ev_eids_col = ev_ev_tbl.column("event_id").to_pylist()
    ev_uids_col = ev_ev_tbl.column("user_id").to_pylist()
    ev_ts_col = ev_ev_tbl.column("timestamp").to_pylist()
    ev_hi_col = ev_ev_tbl.column("history_item_ids").to_pylist()
    ev_ht_col = ev_ev_tbl.column("history_timestamps").to_pylist()
    eval_ev_map = {
        eid: (uid, int(ts), hi_s, ht_s)
        for eid, uid, ts, hi_s, ht_s in zip(ev_eids_col, ev_uids_col, ev_ts_col, ev_hi_col, ev_ht_col)
    }
    del ev_ev_tbl, ev_eids_col, ev_uids_col, ev_ts_col, ev_hi_col, ev_ht_col

    ev_cd_tbl = pq.read_table(ev_cd_path, columns=["rowid", "item_id"])
    eval_rowid_to_item = dict(
        zip(ev_cd_tbl.column("rowid").to_pylist(), ev_cd_tbl.column("item_id").to_pylist())
    )
    del ev_cd_tbl

    if len(eval_rowid_to_item) != n_eval_rows:
        raise RuntimeError(f"eval_cands.parquet row count ({len(eval_rowid_to_item)}) != {n_eval_rows}")

    eval_items = [eval_rowid_to_item[r] for r in eval_rids]
    eval_users = np.array([eval_ev_map[e][0] for e in eval_eids], dtype=object)
    del eval_rowid_to_item

    eval_svd_tbl = score_candidates_with_svd(
        snapshot=eval_snap,
        rowids=eval_rids,
        event_ids=eval_eids,
        item_ids=eval_items,
        event_histories=eval_ev_map,
    )
    score_sec = round(time.time() - t_score, 2)
    del eval_ev_map

    if eval_svd_tbl.num_rows != n_eval_rows:
        raise RuntimeError(f"Eval SVD row count ({eval_svd_tbl.num_rows}) != {n_eval_rows}")
    if not np.array_equal(eval_svd_tbl.column("rowid").to_numpy(zero_copy_only=False).astype(str), eval_rids):
        raise RuntimeError("Eval SVD rowid order mismatch against eval_features.parquet")
    if pc.count_distinct(eval_svd_tbl.column("rowid")).as_py() != n_eval_rows:
        raise RuntimeError("Duplicate rowids in eval_svd_features.parquet")

    for col in SVD_FEATURE_COLS:
        c_arr = eval_svd_tbl.column(col)
        if c_arr.null_count > 0 or not np.isfinite(c_arr.to_numpy(zero_copy_only=False).astype(np.float64)).all():
            raise RuntimeError(f"Null or non-finite values in eval SVD column '{col}'")

    tmp_pq = eval_svd_path.with_name(eval_svd_path.name + ".tmp")
    pq.write_table(eval_svd_tbl, tmp_pq, compression="zstd")
    os.replace(tmp_pq, eval_svd_path)

    # 3. Compute evaluation SVD coverage and compare against Fold B validation coverage (Jan 26)
    u_vocab = eval_snap.user_ids
    i_vocab = eval_snap.item_ids
    u_pos = np.searchsorted(u_vocab, eval_users)
    u_in_vocab = (u_pos < len(u_vocab)) & (u_vocab[np.minimum(u_pos, len(u_vocab) - 1)] == eval_users)
    eval_items_arr = np.asarray(eval_items, dtype=object)
    i_pos = np.searchsorted(i_vocab, eval_items_arr)
    i_in_vocab = (i_pos < len(i_vocab)) & (i_vocab[np.minimum(i_pos, len(i_vocab) - 1)] == eval_items_arr)
    both_in_vocab = u_in_vocab & i_in_vocab

    u_cos_arr = eval_svd_tbl.column("svd_user_item_cos_32").to_numpy(zero_copy_only=False)
    h_mean_arr = eval_svd_tbl.column("svd_history_mean_cos_32").to_numpy(zero_copy_only=False)
    non_exact_mask = eval_in_hist == 0
    cold_mask = eval_cold_user == 1

    val_svd_manifest_path = feat_dir / "svd_manifest_fold_b_3m.json"
    val_summary_ref = {}
    if val_svd_manifest_path.exists():
        val_summary_ref = json.loads(val_svd_manifest_path.read_text(encoding="utf-8")).get("validation_summary", {})

    manifest = {
        "dataset": "evaluation",
        "asof_ts": int(TS_JAN_27_0000),
        "asof_utc": "2025-01-27T00:00:00Z",
        "model_validation_cutoff_ts": int(TS_JAN_26_0000),
        "model_validation_cutoff_utc": "2025-01-26T00:00:00Z",
        "k": int(k),
        "min_degree": float(min_degree),
        "seed": int(seed),
        "svd_feature_cols": SVD_FEATURE_COLS,
        "eval_svd_path": str(eval_svd_path),
        "eval_svd_rows": int(n_eval_rows),
        "eval_svd_events": int(n_eval_rows // 10),
        "eval_svd_size_mb": round(eval_svd_path.stat().st_size / (1024 ** 2), 2),
        "eval_snapshot_stats_jan27": eval_snap.stats,
        "eval_coverage": {
            "user_in_svd_vocab_pct": round(float(u_in_vocab.mean() * 100.0), 4),
            "item_in_svd_vocab_pct": round(float(i_in_vocab.mean() * 100.0), 4),
            "user_and_item_in_svd_vocab_pct": round(float(both_in_vocab.mean() * 100.0), 4),
            "nonzero_user_item_cos_pct": round(float((u_cos_arr != 0.0).mean() * 100.0), 4),
            "nonzero_history_mean_cos_pct": round(float((h_mean_arr != 0.0).mean() * 100.0), 4),
            "nonzero_either_pct": round(float(((u_cos_arr != 0.0) | (h_mean_arr != 0.0)).mean() * 100.0), 4),
            "non_exact_rows_count": int(non_exact_mask.sum()),
            "non_exact_rows_nonzero_either_pct": round(
                float(((u_cos_arr[non_exact_mask] != 0.0) | (h_mean_arr[non_exact_mask] != 0.0)).mean() * 100.0), 4
            ),
            "cold_user_rows_count": int(cold_mask.sum()),
            "cold_user_rows_nonzero_history_mean_cos_pct": round(
                float((h_mean_arr[cold_mask] != 0.0).mean() * 100.0), 4
            ) if cold_mask.sum() else 0.0,
        },
        "fold_b_validation_comparison": {
            "val_asof_ts": val_summary_ref.get("asof_ts"),
            "val_rows": val_summary_ref.get("val_rows"),
            "val_snapshot_stats_jan26": val_summary_ref.get("snapshot_stats", {}),
            "val_coverage": val_summary_ref.get("coverage", {}),
        },
        "snapshot_build_seconds": eval_snap.stats["build_seconds"],
        "svd_score_seconds": score_sec,
        "total_build_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": get_peak_memory_mb(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Built evaluation SVD artifact %s (%d rows, %.2f MB) in %.2fs (user_cos_cov=%.2f%%, hist_cos_cov=%.2f%%, RSS=%s MB)",
        eval_svd_path.name,
        n_eval_rows,
        manifest["eval_svd_size_mb"],
        manifest["total_build_seconds"],
        manifest["eval_coverage"]["nonzero_user_item_cos_pct"],
        manifest["eval_coverage"]["nonzero_history_mean_cos_pct"],
        manifest["peak_rss_mb"],
    )
    return manifest


def build_full_warm_svd_artifacts(
    fold_name: str = "fold_b",
    base_train_rows: int = 3_000_000,
    seed: int = 42,
    k: int = 32,
    min_degree: float = 2.0,
    use_cache: bool = True,
    force: bool = False,
    weighting: str = "binary",
) -> Dict[str, Any]:
    """
    Incrementally generates full warm-window (`2025-01-21`..`2025-01-25`, 4,903,590 rows)
    SVD latent-factor features (`11` cols) for `fold_b`.
    """
    import shutil
    from src.models.lgbm_model import select_fold_sample_events, training_sample_suffix

    t0 = time.time()
    paths = resolve_data_paths()
    feat_dir = paths["features_dir"]
    ev_path = paths["train_events_parquet"]
    cd_path = paths["train_cands_parquet"]

    base_suffix = training_sample_suffix(base_train_rows)
    base_svd_path = feat_dir / f"{fold_name}_train_{base_suffix}_svd_features.parquet"

    w_suffix = f"_{weighting}" if weighting not in ("binary", "") else ""
    if k == 32 and weighting == "binary":
        out_svd_name = f"{fold_name}_train_full_warm_svd_features.parquet"
        manifest_name = f"svd_manifest_{fold_name}_full_warm.json"
        tmp_name = f"_tmp_svd_{fold_name}_full_warm"
    else:
        out_svd_name = f"{fold_name}_train_full_warm_svd{k}{w_suffix}_features.parquet"
        manifest_name = f"svd_manifest_{fold_name}_full_warm_k{k}{w_suffix}.json"
        tmp_name = f"_tmp_svd_{fold_name}_full_warm_k{k}{w_suffix}"

    out_svd_path = feat_dir / out_svd_name
    manifest_path = feat_dir / manifest_name

    if out_svd_path.exists() and manifest_path.exists() and not force:
        logger.info("Full-warm SVD artifact already exists (%s). Loading manifest.", manifest_path.name)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    can_reuse_base = (k == 32 and weighting == "binary" and base_svd_path.exists())
    base_sel = select_fold_sample_events(fold_name, base_train_rows, seed, feat_dir)
    base_rids_all = None
    base_svd_tbl = None
    if can_reuse_base:
        base_svd_tbl = pq.read_table(base_svd_path)
        base_rids_all = base_svd_tbl.column("rowid").combine_chunks()

    day_starts = TRAINING_SETS[fold_name]["day_starts"]
    tmp_dir = feat_dir / tmp_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    day_cache_paths: List[Path] = []
    per_day_stats: List[Dict[str, Any]] = []
    snapshot_stats_by_day: Dict[str, Any] = {}
    total_base_reused = 0
    total_new_scored = 0
    expected_total_rows = 0

    for day_idx, ((part_path, day_str, _avail_ev, avail_rows), day_ts) in enumerate(
        zip(base_sel["day_totals"], day_starts)
    ):
        if day_str == "2025-01-20":
            continue
        expected_total_rows += int(avail_rows)
        day_pq = tmp_dir / f"day_{day_str}.parquet"
        day_meta_json = tmp_dir / f"day_{day_str}.json"
        if day_pq.exists() and day_meta_json.exists() and not force:
            cached = json.loads(day_meta_json.read_text(encoding="utf-8"))
            per_day_stats.append(cached["per_day"])
            snapshot_stats_by_day[day_str] = cached["snapshot_stats"]
            total_base_reused += int(cached["per_day"]["base_rows_reused"])
            total_new_scored += int(cached["per_day"]["new_rows_scored"])
            day_cache_paths.append(day_pq)
            logger.info("%s: loaded cached full-warm SVD partition (%d rows)", day_str, cached["per_day"]["rows"])
            continue

        t_day = time.time()
        snap = build_svd_snapshot_asof(
            asof_ts=int(day_ts),
            train_events_path=ev_path,
            train_cands_path=cd_path,
            k=k,
            min_degree=min_degree,
            seed=seed,
            use_cache=use_cache,
            weighting=weighting,
        )
        snapshot_stats_by_day[day_str] = snap.stats

        meta = pq.read_table(part_path, columns=["rowid", "event_id"])
        day_rids = meta.column("rowid").combine_chunks()
        day_eids = meta.column("event_id").combine_chunks()

        if can_reuse_base:
            base_events = base_sel["chosen_events"][day_idx]
            if base_events is None:
                raise RuntimeError(f"Expected sampled base events on {day_str}")
            base_ev_set = pa.array(base_events, type=pa.string())
            is_base = pc.is_in(day_eids, value_set=base_ev_set).to_numpy(zero_copy_only=False)

            base_day_rids = day_rids.filter(pa.array(is_base))
            idx = pc.index_in(base_day_rids, value_set=base_rids_all)
            if idx.null_count > 0:
                raise RuntimeError(f"{day_str}: {idx.null_count} base rows missing from {base_svd_path.name}")
            base_day_svd = base_svd_tbl.take(idx)

            new_mask = ~is_base
            new_rids = day_rids.filter(pa.array(new_mask)).to_numpy(zero_copy_only=False).astype(str)
            new_eids = day_eids.filter(pa.array(new_mask)).to_numpy(zero_copy_only=False).astype(str)

            ev_map, rid_to_item = _fetch_event_histories_and_items_for_svd(ev_path, cd_path, new_eids)
            new_items = [rid_to_item[r] for r in new_rids]
            new_day_svd = score_candidates_with_svd(
                snapshot=snap,
                rowids=new_rids,
                event_ids=new_eids,
                item_ids=new_items,
                event_histories=ev_map,
            )
            del ev_map, rid_to_item, new_items
            gc.collect()

            combined = pa.concat_tables([base_day_svd, new_day_svd])
            order = pc.index_in(day_rids, value_set=combined.column("rowid").combine_chunks())
            if order.null_count > 0:
                raise RuntimeError(f"{day_str}: SVD rowid assembly failure")
            day_out = combined.take(order)
            del base_day_svd, new_day_svd, combined
            n_base_rows = int(is_base.sum())
            n_new_rows = int(new_mask.sum())
        else:
            day_rids_str = day_rids.to_numpy(zero_copy_only=False).astype(str)
            day_eids_str = day_eids.to_numpy(zero_copy_only=False).astype(str)
            ev_map, rid_to_item = _fetch_event_histories_and_items_for_svd(ev_path, cd_path, day_eids_str)
            day_items = [rid_to_item[r] for r in day_rids_str]
            day_out = score_candidates_with_svd(
                snapshot=snap,
                rowids=day_rids_str,
                event_ids=day_eids_str,
                item_ids=day_items,
                event_histories=ev_map,
            )
            del ev_map, rid_to_item, day_items
            n_base_rows = 0
            n_new_rows = len(day_rids_str)

        if not day_out.column("rowid").combine_chunks().equals(day_rids):
            raise RuntimeError(f"{day_str}: SVD assembled rowid order mismatch")

        col_suffix = f"{k}_idf" if weighting in ("item_idf", "idf") else f"{k}"
        u_col_name = f"svd_user_item_cos_{col_suffix}"
        h_col_name = f"svd_history_mean_cos_{col_suffix}"
        u_cos_day = day_out.column(u_col_name).to_numpy(zero_copy_only=False)
        h_cos_day = day_out.column(h_col_name).to_numpy(zero_copy_only=False)
        total_base_reused += n_base_rows
        total_new_scored += n_new_rows

        day_info = {
            "day": day_str,
            "asof_ts": int(day_ts),
            "events": int(meta.num_rows // 10),
            "rows": int(meta.num_rows),
            "base_rows_reused": n_base_rows,
            "new_rows_scored": n_new_rows,
            "nonzero_user_item_cos_pct": round(float((u_cos_day != 0.0).mean() * 100.0), 4),
            "nonzero_history_mean_cos_pct": round(float((h_cos_day != 0.0).mean() * 100.0), 4),
            "seconds": round(time.time() - t_day, 2),
            "peak_rss_mb": get_peak_memory_mb(),
        }
        per_day_stats.append(day_info)

        tmp_pq = day_pq.with_name(day_pq.name + ".tmp")
        pq.write_table(day_out, tmp_pq, compression="zstd")
        os.replace(tmp_pq, day_pq)
        day_meta_json.write_text(
            json.dumps({"per_day": day_info, "snapshot_stats": snapshot_stats_by_day[day_str]}, indent=2),
            encoding="utf-8",
        )
        day_cache_paths.append(day_pq)
        logger.info(
            "%s: full-warm SVD (k=%d, w=%s) reused %d base + scored %d new -> %d rows (u_cos=%.2f%%, h_cos=%.2f%%) in %.2fs (RSS=%s MB)",
            day_str,
            k,
            weighting,
            n_base_rows,
            n_new_rows,
            day_info["rows"],
            day_info["nonzero_user_item_cos_pct"],
            day_info["nonzero_history_mean_cos_pct"],
            day_info["seconds"],
            day_info["peak_rss_mb"],
        )
        del meta, day_rids, day_eids, day_out, snap
        gc.collect()

    if base_svd_tbl is not None:
        del base_svd_tbl, base_rids_all
        gc.collect()

    out_svd_tbl = pa.concat_tables([pq.read_table(p) for p in day_cache_paths])
    if out_svd_tbl.num_rows != expected_total_rows:
        raise RuntimeError(f"Full-warm SVD row count ({out_svd_tbl.num_rows}) != expected ({expected_total_rows})")
    if pc.count_distinct(out_svd_tbl.column("rowid")).as_py() != expected_total_rows:
        raise RuntimeError("Duplicate rowids in full-warm SVD feature table")

    ref_cf_path = feat_dir / f"{fold_name}_train_full_warm_cf_features.parquet"
    if ref_cf_path.exists():
        ref_rids = pq.read_table(ref_cf_path, columns=["rowid"]).column("rowid").combine_chunks()
        if not out_svd_tbl.column("rowid").combine_chunks().equals(ref_rids):
            raise RuntimeError(f"Full-warm SVD rowid order mismatch vs {ref_cf_path.name}")
        del ref_rids

    expected_cols = get_svd_feature_cols(k, weighting=weighting)
    for col in expected_cols:
        c_arr = out_svd_tbl.column(col)
        if c_arr.null_count > 0 or not np.isfinite(c_arr.to_numpy(zero_copy_only=False).astype(np.float64)).all():
            raise RuntimeError(f"Null or non-finite values in full-warm SVD column '{col}'")

    tmp_tr = out_svd_path.with_name(out_svd_path.name + ".tmp")
    pq.write_table(out_svd_tbl, tmp_tr, compression="zstd")
    os.replace(tmp_tr, out_svd_path)
    del out_svd_tbl
    gc.collect()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = {
        "fold": fold_name,
        "exclude_cold_day": True,
        "target_train_rows": int(expected_total_rows),
        "base_train_rows": int(base_train_rows),
        "base_rows_reused": int(total_base_reused),
        "new_rows_scored": int(total_new_scored),
        "seed": int(seed),
        "k": int(k),
        "min_degree": float(min_degree),
        "weighting": weighting,
        "svd_feature_cols": expected_cols,
        "train_svd_path": str(out_svd_path),
        "train_svd_rows": int(expected_total_rows),
        "train_svd_size_mb": round(out_svd_path.stat().st_size / (1024 ** 2), 2),
        "per_day": per_day_stats,
        "daily_snapshots": snapshot_stats_by_day,
        "total_build_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": get_peak_memory_mb(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Built full-warm SVD artifact %s (%d rows: %d reused + %d new, %.2f MB) in %.2fs (Peak RSS: %s MB)",
        out_svd_path.name,
        expected_total_rows,
        total_base_reused,
        total_new_scored,
        manifest["train_svd_size_mb"],
        manifest["total_build_seconds"],
        manifest["peak_rss_mb"],
    )
    return manifest
    for col in expected_cols:
        c_arr = out_svd_tbl.column(col)
        if c_arr.null_count > 0 or not np.isfinite(c_arr.to_numpy(zero_copy_only=False).astype(np.float64)).all():
            raise RuntimeError(f"Null or non-finite values in full-warm SVD column '{col}'")

    tmp_tr = out_svd_path.with_name(out_svd_path.name + ".tmp")
    pq.write_table(out_svd_tbl, tmp_tr, compression="zstd")
    os.replace(tmp_tr, out_svd_path)
    del out_svd_tbl
    gc.collect()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = {
        "fold": fold_name,
        "exclude_cold_day": True,
        "target_train_rows": int(expected_total_rows),
        "base_train_rows": int(base_train_rows),
        "base_rows_reused": int(total_base_reused),
        "new_rows_scored": int(total_new_scored),
        "seed": int(seed),
        "k": int(k),
        "min_degree": float(min_degree),
        "svd_feature_cols": expected_cols,
        "train_svd_path": str(out_svd_path),
        "train_svd_rows": int(expected_total_rows),
        "train_svd_size_mb": round(out_svd_path.stat().st_size / (1024 ** 2), 2),
        "per_day": per_day_stats,
        "daily_snapshots": snapshot_stats_by_day,
        "total_build_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": get_peak_memory_mb(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Built full-warm SVD artifact %s (%d rows: %d reused + %d new, %.2f MB) in %.2fs (Peak RSS: %s MB)",
        out_svd_path.name,
        expected_total_rows,
        total_base_reused,
        total_new_scored,
        manifest["train_svd_size_mb"],
        manifest["total_build_seconds"],
        manifest["peak_rss_mb"],
    )
    return manifest


def embedding_features(
    rowids: Sequence[str],
    asof_date: str = "2025-01-26",
) -> pd.DataFrame:
    """
    Backward-compatible helper returning a DataFrame keyed by `rowid` with the 11 SVD features
    initialized to safe cold-start defaults when called without event/candidate context.
    """
    n = len(rowids)
    data: Dict[str, Any] = {"rowid": list(rowids)}
    for col in SVD_FEATURE_COLS:
        if col.endswith("_rank"):
            data[col] = np.ones(n, dtype=np.uint8)
        else:
            data[col] = np.zeros(n, dtype=np.float32)
    return pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and evaluate Phase 4E SVD latent feature artifacts.")
    parser.add_argument("--fold", choices=["fold_a", "fold_b"], default="fold_b")
    parser.add_argument("--sample-rows", type=int, default=3_000_000)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--min-degree", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate standalone SVD on validation slice only")
    parser.add_argument("--save-val-parquet", action="store_true", help="Save scored validation features to parquet")
    parser.add_argument("--build-eval-svd", action="store_true",
                        help="Build evaluation SVD features (eval_svd_features.parquet) at Jan 27 00:00 UTC cutoff")
    parser.add_argument("--build-full-warm", action="store_true",
                        help="Phase 4F: build full warm (Jan 21-25, 4,903,590 rows) SVD features")
    parser.add_argument("--force", action="store_true", help="Force rebuild of SVD feature artifacts")
    parser.add_argument("--weighting", choices=["binary", "item_idf"], default="binary",
                        help="SVD interaction weighting: binary (default) or item_idf")
    args = parser.parse_args()

    if args.build_full_warm:
        res = build_full_warm_svd_artifacts(
            fold_name=args.fold,
            base_train_rows=3_000_000,
            seed=args.seed,
            k=args.k,
            min_degree=args.min_degree,
            force=args.force,
            weighting=args.weighting,
        )
        print(json.dumps(res, indent=2))
        raise SystemExit(0)

    if args.build_eval_svd:
        res = build_eval_svd_artifacts(
            k=args.k,
            min_degree=args.min_degree,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps(res, indent=2))
        raise SystemExit(0)

    if args.eval_only:
        _, summary = evaluate_svd_on_validation_slice(
            fold_name=args.fold,
            k=args.k,
            min_degree=args.min_degree,
            seed=args.seed,
            save_val_parquet=args.save_val_parquet,
            weighting=args.weighting,
        )
        print(json.dumps(summary, indent=2))
    else:
        manifest = build_fold_svd_artifacts(
            fold_name=args.fold,
            target_train_rows=args.sample_rows,
            seed=args.seed,
            k=args.k,
            min_degree=args.min_degree,
            force=args.force,
        )
        print(json.dumps(manifest["validation_summary"], indent=2))
