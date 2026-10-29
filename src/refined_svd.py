"""
Phase 7A: Supervised Refinement of Truncated SVD Latent-Factor Embeddings.

Learns a low-rank residual bilinear metric transformation M = I + A B^T (32 x 8)
supervised by candidate click labels, while preserving the underlying collaborative geometry.

Key guarantees:
1. Strictly temporal: For day D, the metric model M is trained ONLY on events strictly
   before day D (timestamp < asof_ts), with zero future label leakage.
2. Geometry preservation: M is anchored to the identity I_32 via strong Frobenius
   regularization, ensuring it perturbs rather than destroys the spectral representation.
3. Procrustes basis alignment: Aligns target snapshot coordinate axes into the training basis
   via an orthogonal rotation Q, guaranteeing that within-snapshot dot products and cosines
   are preserved to machine precision (||cos_orig - cos_aligned|| < 1e-6).
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim

from src.embeddings import SparseSVDSnapshot, build_svd_snapshot_asof
from src.utils import get_logger, resolve_data_paths

logger = get_logger("RefinedSVDEmbeddings")

REFINED_SVD_FEATURE_COLS = [
    "svd_refined_user_affinity",
    "svd_refined_history_affinity",
    "we_svd_refined_history_diff",
    "we_svd_refined_history_rank",
    "svd_refined_vs_orig_residual",
]


class ResidualSVDMetricModel(nn.Module):
    """
    Residual Bilinear Metric Model: M = I_32 + A B^T.
    A in R^(32 x 8), B in R^(32 x 8).
    Initialized with A ~ N(0, 0.01^2), B = 0.
    At step 0: M = I_32 exactly (output equals original SVD cosine).
    """
    def __init__(self, k: int = 32, rank: int = 8, init_scale: float = 10.0, init_bias: float = -2.7):
        super().__init__()
        self.k = k
        self.rank = rank
        self.A = nn.Parameter(torch.randn(k, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(k, rank))
        self.scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_affinity(self, u_norm: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
        base_dot = torch.sum(u_norm * v_norm, dim=1)
        u_proj = torch.matmul(u_norm, self.A)
        v_proj = torch.matmul(v_norm, self.B)
        res_dot = torch.sum(u_proj * v_proj, dim=1)
        return base_dot + res_dot

    def forward(self, u_norm: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
        s = self.forward_affinity(u_norm, v_norm)
        return self.scale * s + self.bias


def solve_orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Finds orthogonal matrix Q (k x k) such that || A Q - B ||_F^2 is minimized.
    Both A and B are (N x k) normalized matrices.
    """
    M = A.T @ B
    U, s, Vt = np.linalg.svd(M)
    return (U @ Vt).astype(np.float32)


def train_supervised_metric_asof(
    asof_ts: int,
    training_days_lookback_ts: Optional[int] = None,
    k: int = 32,
    rank: int = 8,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 8192,
    lr: float = 0.005,
    lambda_reg: float = 0.01,
) -> Tuple[ResidualSVDMetricModel, SparseSVDSnapshot]:
    """
    Trains a ResidualSVDMetricModel using ONLY events strictly prior to asof_ts.
    If training_days_lookback_ts is provided, uses events in [training_days_lookback_ts, asof_ts).
    """
    paths = resolve_data_paths()
    ev_path = str(paths["train_events_parquet"])
    cd_path = str(paths["train_cands_parquet"])
    
    # 1. Load SVD snapshot strictly before asof_ts
    snap = build_svd_snapshot_asof(
        asof_ts=asof_ts,
        k=k,
        seed=seed,
        use_cache=True,
        weighting="binary"
    )
    
    # 2. Query positive and negative candidates strictly prior to asof_ts
    con = duckdb.connect()
    time_filter = f"e.timestamp < {int(asof_ts)}"
    if training_days_lookback_ts is not None:
        time_filter += f" AND e.timestamp >= {int(training_days_lookback_ts)}"
        
    query = f"""
        WITH sample_events AS (
            SELECT event_id, user_id, timestamp, history_item_ids, history_timestamps
            FROM '{ev_path}' e
            WHERE {time_filter}
            USING SAMPLE 50000 (reservoir, {seed})
        )
        SELECT c.rowid, c.event_id, c.item_id, c.clicked, e.user_id, e.timestamp, e.history_item_ids, e.history_timestamps
        FROM '{cd_path}' c
        JOIN sample_events e ON c.event_id = e.event_id
    """
    df_train = con.execute(query).df()
    con.close()
    
    # Build vector mappings
    zero_vec = np.zeros(k, dtype=np.float32)
    i_map = {iid: snap.item_emb_norm[idx] for idx, iid in enumerate(snap.item_ids)}
    
    ev_dict = {}
    for eid, grp in df_train.groupby("event_id"):
        row = grp.iloc[0]
        ev_ts = row["timestamp"]
        hi = json.loads(row["history_item_ids"]) if row["history_item_ids"] else []
        ht = json.loads(row["history_timestamps"]) if row["history_timestamps"] else []
        valid_items = [it for it, ts in zip(hi, ht) if ts < ev_ts and it in i_map]
        if valid_items:
            h_mat = np.array([i_map[it] for it in valid_items], dtype=np.float32)
            h_mean = h_mat.mean(axis=0)
            norm = np.linalg.norm(h_mean)
            h_norm = (h_mean / norm).astype(np.float32) if norm > 1e-8 else zero_vec
        else:
            h_norm = zero_vec
        ev_dict[eid] = h_norm
        
    v_arr = np.array([i_map.get(iid, zero_vec) for iid in df_train["item_id"]], dtype=np.float32)
    h_arr = np.array([ev_dict.get(eid, zero_vec) for eid in df_train["event_id"]], dtype=np.float32)
    y_arr = df_train["clicked"].values.astype(np.float32)
    
    # Filter active vectors
    active_mask = (np.abs(h_arr).sum(axis=1) > 0) & (np.abs(v_arr).sum(axis=1) > 0)
    h_train_t = torch.tensor(h_arr[active_mask], dtype=torch.float32)
    v_train_t = torch.tensor(v_arr[active_mask], dtype=torch.float32)
    y_train_t = torch.tensor(y_arr[active_mask], dtype=torch.float32)
    
    model = ResidualSVDMetricModel(k=k, rank=rank)
    pos_weight = torch.tensor([(1.0 - 0.0634) / 0.0634], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW([
        {"params": [model.A, model.B], "lr": lr, "weight_decay": 1e-4},
        {"params": [model.scale, model.bias], "lr": 0.01, "weight_decay": 0.0}
    ])
    
    n_batches = math.ceil(len(h_train_t) / batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(len(h_train_t))
        for b in range(n_batches):
            idx_b = perm[b*batch_size : (b+1)*batch_size]
            hb = h_train_t[idx_b]
            vb = v_train_t[idx_b]
            yb = y_train_t[idx_b]
            
            optimizer.zero_grad()
            logits = model(hb, vb)
            loss_bce = criterion(logits, yb)
            
            AtA = torch.matmul(model.A.T, model.A)
            BtB = torch.matmul(model.B.T, model.B)
            reg_loss = lambda_reg * torch.sum(AtA * BtB)
            
            loss = loss_bce + reg_loss
            loss.backward()
            optimizer.step()
            
    return model, snap


def compute_refined_svd_features_for_partition(
    target_df: Any,
    snap_target: SparseSVDSnapshot,
    metric_model: ResidualSVDMetricModel,
    snap_canon: Optional[SparseSVDSnapshot] = None,
) -> pa.Table:
    """
    Given a dataframe of target candidate impressions (with rowid, event_id, user_id, item_id,
    timestamp, history_item_ids, history_timestamps):
    Computes 5 refined SVD features with Procrustes rotation if snap_canon is supplied.
    """
    k = snap_target.k
    zero_vec = np.zeros(k, dtype=np.float32)
    
    # 1. Procrustes Alignment Q if snap_canon is provided
    if snap_canon is not None and snap_canon.asof_ts != snap_target.asof_ts:
        common_items = sorted(list(set(snap_canon.item_ids) & set(snap_target.item_ids)))
        canon_idx_map = {it: i for i, it in enumerate(snap_canon.item_ids)}
        tgt_idx_map = {it: i for i, it in enumerate(snap_target.item_ids)}
        
        Vc = snap_canon.item_emb_norm[[canon_idx_map[it] for it in common_items]]
        Vt = snap_target.item_emb_norm[[tgt_idx_map[it] for it in common_items]]
        Q = solve_orthogonal_procrustes(Vt, Vc)
    else:
        Q = np.eye(k, dtype=np.float32)
        
    # Rotate target embeddings into canonical basis
    tgt_user_aligned = snap_target.user_emb_norm @ Q
    tgt_item_aligned = snap_target.item_emb_norm @ Q
    
    u_map = {uid: tgt_user_aligned[i] for i, uid in enumerate(snap_target.user_ids)}
    i_map = {iid: tgt_item_aligned[i] for i, iid in enumerate(snap_target.item_ids)}
    
    # Pre-parse event histories once per event
    ev_dict = {}
    for eid, grp in target_df.groupby("event_id"):
        row = grp.iloc[0]
        ev_ts = row["timestamp"]
        hi = json.loads(row["history_item_ids"]) if row["history_item_ids"] else []
        ht = json.loads(row["history_timestamps"]) if row["history_timestamps"] else []
        valid_items = [it for it, ts in zip(hi, ht) if ts < ev_ts and it in i_map]
        if valid_items:
            h_mat = np.array([i_map[it] for it in valid_items], dtype=np.float32)
            h_mean = h_mat.mean(axis=0)
            norm = np.linalg.norm(h_mean)
            h_norm = (h_mean / norm).astype(np.float32) if norm > 1e-8 else zero_vec
        else:
            h_norm = zero_vec
        ev_dict[eid] = h_norm
        
    u_arr = np.array([u_map.get(uid, zero_vec) for uid in target_df["user_id"]], dtype=np.float32)
    v_arr = np.array([i_map.get(iid, zero_vec) for iid in target_df["item_id"]], dtype=np.float32)
    h_arr = np.array([ev_dict.get(eid, zero_vec) for eid in target_df["event_id"]], dtype=np.float32)
    
    # Compute refined affinities using PyTorch model
    with torch.no_grad():
        u_t = torch.tensor(u_arr, dtype=torch.float32)
        v_t = torch.tensor(v_arr, dtype=torch.float32)
        h_t = torch.tensor(h_arr, dtype=torch.float32)
        
        orig_h_cos = torch.sum(h_t * v_t, dim=1).numpy()
        ref_u_aff = metric_model.forward_affinity(u_t, v_t).numpy()
        ref_h_aff = metric_model.forward_affinity(h_t, v_t).numpy()
        
    res_df = pd.DataFrame({
        "rowid": target_df["rowid"].values,
        "event_id": target_df["event_id"].values,
        "svd_refined_user_affinity": ref_u_aff.astype(np.float32),
        "svd_refined_history_affinity": ref_h_aff.astype(np.float32),
        "svd_refined_vs_orig_residual": (ref_h_aff - orig_h_cos).astype(np.float32),
    })
    
    # Within-event difference and rank
    res_df["we_svd_refined_history_diff"] = (
        res_df["svd_refined_history_affinity"] - res_df.groupby("event_id")["svd_refined_history_affinity"].transform("mean")
    ).astype(np.float32)
    
    res_df["we_svd_refined_history_rank"] = (
        res_df.groupby("event_id")["svd_refined_history_affinity"].rank(ascending=False, method="min")
    ).astype(np.uint8)
    
    res_df.drop(columns=["event_id"], inplace=True)
    return pa.Table.from_pandas(res_df)
