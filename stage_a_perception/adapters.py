# -*- coding: utf-8 -*-
# stage_a_perception/adapters.py
"""
Adapters for Stage-A Perception inputs.

This module provides two entry points:
1) `from_demo_dir(path)`       – load a single demo case from a folder.
2) `from_dataset(root_dir)`    – load multiple service instances from the
                                 office_mini_storm_dataset-style directory.

Outputs (unified format):
    logs    : List[str]                       # length = N
    metrics : torch.FloatTensor [N, T, C]     # z-score normalized
    spans   : torch.FloatTensor [N, L, 4]     # (svc_id, op_id, duration_z, error_flag)
    lengths : torch.LongTensor  [N]           # each item is L for the sample
    Q       : torch.FloatTensor [N, 3]        # quality scores (logs, metrics, traces) in [0,1]
    Y       : torch.LongTensor  [N]           # labels if provided (default 0)
    meta    : List[dict] or dict              # metadata loaded from meta.json

Notes
-----
- This file contains no sensitive information and is ready for public repositories.
- Resampling to a fixed Δ is typically handled by data preparation; here we assume
  files are already aligned. Light normalization and quality scoring are included.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

__all__ = [
    "from_demo_dir",
    "from_dataset",
    "zscore_normalize",
    "quality_indicators",
    "quality_score",
]


# --------------------------------------------------------------------------
# Basic utilities
# --------------------------------------------------------------------------

def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """
    Per-feature z-score with NaN safety.
    x: [T, C] or [L, D]
    """
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def _safe_hist_entropy(values: np.ndarray, bins: int = 10) -> float:
    """
    Histogram-based Shannon entropy normalized to [0,1] by log2(bins).
    """
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(np.float32)
    p = p / (p.sum() + 1e-6)
    ent = -np.sum(p * np.log2(p + 1e-6))
    return float(ent / (math.log2(bins) + 1e-6))


def quality_indicators(
    arr: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> Tuple[float, float, float, float]:
    """
    Compute quality indicators for a 1D/2D signal array.

    Returns:
        missing_rate  : float in [0,1]   – fraction of NaNs or missing entries
        entropy_norm  : float in [0,1]   – normalized entropy
        staleness_norm: float in [0,1]   – higher means more stale
        noise_ratio   : float in [0,1]   – fraction of >3σ outliers
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return 1.0, 0.0, 1.0, 0.0

    # Collapse to 1D for statistics if needed
    if a.ndim > 1:
        a_flat = a.reshape(-1)
    else:
        a_flat = a

    # Missing rate
    valid_mask = ~np.isnan(a_flat)
    missing_rate = 1.0 - float(np.mean(valid_mask)) if a_flat.size > 0 else 1.0

    # Entropy on valid values
    vals = a_flat[valid_mask]
    entropy_norm = _safe_hist_entropy(vals) if vals.size > 0 else 0.0

    # Staleness: relative distance from last valid index to end
    staleness_norm = 1.0
    if a.ndim >= 1 and a.shape[0] > 0:
        # Approximate staleness along the first axis
        first_dim = a.shape[0]
        if np.any(valid_mask):
            # Find last valid along the flattened view, map to first axis approx.
            last_idx = int(np.max(np.where(valid_mask)[0]))
            # Normalize by length
            staleness_norm = (a_flat.size - 1 - last_idx) / max(1, a_flat.size - 1)
            staleness_norm = float(np.clip(staleness_norm, 0.0, 1.0))
        else:
            staleness_norm = 1.0

    # Noise ratio: > 3*std on valid values
    if vals.size > 0:
        mu, sd = float(np.mean(vals)), float(np.std(vals) + 1e-6)
        noise_ratio = float(np.mean(np.abs(vals - mu) > 3.0 * sd))
    else:
        noise_ratio = 0.0

    return float(missing_rate), float(entropy_norm), float(staleness_norm), float(noise_ratio)


def quality_score(
    indicators: Tuple[float, float, float, float],
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Convert indicators into a single quality score in [0,1].

    By default:
        score = mean([1 - missing_rate, entropy_norm,
                      1 - staleness_norm, 1 - noise_ratio])

    This is a lightweight proxy for modality reliability (q^(m)).
    """
    mr, ent, stale, noise = indicators
    w1, w2, w3, w4 = weights
    comp = (
        w1 * (1.0 - mr)
        + w2 * ent
        + w3 * (1.0 - stale)
        + w4 * (1.0 - noise)
    )
    # Normalize by sum of weights (assumed 1.0 if default)
    wsum = float(sum(weights)) + 1e-6
    return float(np.clip(comp / wsum, 0.0, 1.0))


# --------------------------------------------------------------------------
# Demo loader (single case)
# --------------------------------------------------------------------------

def from_demo_dir(
    dir_path: Union[str, Path],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Load a single case from a demo directory:
        dir_path/
            log.txt
            metrics.npy        # [T, C]
            traces.npy         # [L, 4] -> (svc_id, op_id, duration, error_flag)
            meta.json          # {"y": int, ...}

    Returns:
        logs, metrics, spans, lengths, Q, Y, meta
    """
    d = Path(dir_path).expanduser().resolve()
    log_path = d / "log.txt"
    metrics_path = d / "metrics.npy"
    traces_path = d / "traces.npy"
    meta_path = d / "meta.json"

    if not log_path.exists():
        raise FileNotFoundError(f"Missing: {log_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing: {metrics_path}")
    if not traces_path.exists():
        raise FileNotFoundError(f"Missing: {traces_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    # Logs (as raw text)
    logs: List[str]
    logs = [log_path.read_text(encoding="utf-8").strip()]

    # Metrics [T, C] -> normalize -> [1, T, C]
    M = np.load(metrics_path)
    if M.ndim != 2:
        raise ValueError(f"`metrics.npy` must be 2-D [T, C], got {M.shape}")
    Mz = zscore_normalize(M)
    metrics = torch.from_numpy(Mz).float().unsqueeze(0)

    # Traces [L, 4] -> duration z-score on column 2 -> [1, L, 4]
    S = np.load(traces_path)
    if S.ndim != 2 or S.shape[1] != 4:
        raise ValueError(f"`traces.npy` must be 2-D [L, 4], got {S.shape}")
    S = S.astype(np.float32)
    if S.shape[0] > 0:
        # z-score the duration column (index 2)
        dur = S[:, 2:3]
        S[:, 2:3] = zscore_normalize(dur)
    spans = torch.from_numpy(S).float().unsqueeze(0)
    lengths = torch.tensor([S.shape[0]], dtype=torch.long)

    # Meta, label, and modality quality
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    y = int(meta.get("y", 0))
    Y = torch.tensor([y], dtype=torch.long)

    # Quality scores (logs, metrics, traces)
    # Logs proxy: length of text as a simple signal container
    logs_len = np.array([len(logs[0])], dtype=np.float32)
    q_logs = quality_score(quality_indicators(logs_len))
    q_mets = quality_score(quality_indicators(M))
    q_trcs = quality_score(quality_indicators(S))
    Q = torch.tensor([[q_logs, q_mets, q_trcs]], dtype=torch.float32)

    return logs, metrics, spans, lengths, Q, Y, meta


# --------------------------------------------------------------------------
# Dataset loader (multiple services)
# --------------------------------------------------------------------------

def from_dataset(
    root_dir: Union[str, Path],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    """
    Load multiple service instances from a dataset directory:

        root_dir/
          svc_A/
            log.txt
            metrics.npy
            traces.npy
            meta.json
          svc_B/
            ...

    Returns:
        logs, metrics, spans, lengths, Q, Y, meta_list
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Invalid dataset directory: {root}")

    svc_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    logs_all: List[str] = []
    metrics_all: List[torch.Tensor] = []
    spans_all: List[torch.Tensor] = []
    lengths_all: List[torch.Tensor] = []
    Q_all: List[torch.Tensor] = []
    Y_all: List[torch.Tensor] = []
    meta_all: List[dict] = []

    for svc in svc_dirs:
        log_file = svc / "log.txt"
        met_file = svc / "metrics.npy"
        trc_file = svc / "traces.npy"
        meta_file = svc / "meta.json"
        if not (log_file.exists() and met_file.exists() and trc_file.exists() and meta_file.exists()):
            # Skip incomplete samples
            continue

        # Logs
        log_text = log_file.read_text(encoding="utf-8").strip()
        logs_all.append(log_text)

        # Metrics
        M = np.load(met_file)
        if M.ndim != 2:
            raise ValueError(f"[{svc.name}] metrics.npy must be 2-D [T, C], got {M.shape}")
        Mz = zscore_normalize(M)
        metrics_all.append(torch.from_numpy(Mz).float().unsqueeze(0))

        # Traces
        S = np.load(trc_file).astype(np.float32)
        if S.ndim != 2 or S.shape[1] != 4:
            raise ValueError(f"[{svc.name}] traces.npy must be 2-D [L, 4], got {S.shape}")
        if S.shape[0] > 0:
            S[:, 2:3] = zscore_normalize(S[:, 2:3])  # z-score duration
        spans_all.append(torch.from_numpy(S).float().unsqueeze(0))
        lengths_all.append(torch.tensor([S.shape[0]], dtype=torch.long))

        # Meta and label
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta_all.append(meta)
        y = int(meta.get("y", 0))
        Y_all.append(torch.tensor([y], dtype=torch.long))

        # Quality scores
        q_logs = quality_score(quality_indicators(np.array([len(log_text)], dtype=np.float32)))
        q_mets = quality_score(quality_indicators(M))
        q_trcs = quality_score(quality_indicators(S))
        Q_all.append(torch.tensor([[q_logs, q_mets, q_trcs]], dtype=torch.float32))

    if len(logs_all) == 0:
        raise RuntimeError(f"No valid samples found under {root}")

    # Concatenate along batch dimension
    metrics_cat = torch.cat(metrics_all, dim=0)   # [N, T, C]
    spans_cat = torch.cat(spans_all, dim=0)       # [N, L, 4]  (ragged L handled via `lengths`)
    lengths_cat = torch.cat(lengths_all, dim=0)   # [N]
    Q_cat = torch.cat(Q_all, dim=0)               # [N, 3]
    Y_cat = torch.cat(Y_all, dim=0)               # [N]

    return logs_all, metrics_cat, spans_cat, lengths_cat, Q_cat, Y_cat, meta_all
