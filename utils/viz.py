# -*- coding: utf-8 -*-
"""
utils/viz.py

最小可用可视化：
- plot_topk_roots: 根因候选分数条形图
- plot_alpha_summary: Stage-A alpha(模态贡献) 的均值 + IQR 误差棒
"""
from __future__ import annotations
import os, json
from typing import Sequence, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # 后端无窗口环境安全
import matplotlib.pyplot as plt

def plot_topk_roots(topk_roots: Sequence[int], topk_scores: Sequence[float], save_path: str):
    plt.figure(figsize=(6, 4))
    xs = range(len(topk_roots))
    plt.bar(xs, topk_scores)
    plt.xticks(xs, [str(r) for r in topk_roots])
    plt.xlabel("Root candidates (node id)")
    plt.ylabel("Score")
    plt.title("Top-K Root Candidates")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_alpha_summary(alpha_stats: Dict[str, Dict[str, float]], save_path: str):
    """
    alpha_stats 形如：
    {
      "mean": {"logs":0.1,"metrics":0.6,"traces":0.3},
      "p25": {...}, "p50": {...}, "p75": {...}
    }
    用均值作为柱高，(p75 - p25) / 2 作为误差棒。
    """
    if not alpha_stats or "mean" not in alpha_stats:
        raise ValueError("alpha_stats is empty or missing 'mean'.")

    keys = ["logs", "metrics", "traces"]
    means = [alpha_stats["mean"][k] for k in keys]
    iqr   = [ (alpha_stats["p75"][k] - alpha_stats["p25"][k]) / 2.0 for k in keys ]

    plt.figure(figsize=(5, 4))
    xs = range(len(keys))
    plt.bar(xs, means, yerr=iqr, capsize=6)
    plt.xticks(xs, keys)
    plt.ylim(0, 1.0)
    plt.ylabel("Alpha weight")
    plt.title("Modal Contributions (mean ± IQR/2)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def quick_from_reasoning_json(reason_json_path: str, alpha_pt_path: Optional[str], out_dir: str):
    """从 single_case.json 与 mm_node_embeddings.pt 生成两张图"""
    import torch
    with open(reason_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = obj.get("result", {})
    topk_roots = res.get("topk_roots", [])
    topk_scores = res.get("topk_scores", [])
    os.makedirs(out_dir, exist_ok=True)
    plot_topk_roots(topk_roots, topk_scores, os.path.join(out_dir, "topk_roots.png"))

    alpha_stats = None
    if alpha_pt_path and os.path.exists(alpha_pt_path):
        try:
            # 加 weights_only=True 可避免未来 PyTorch 警告（旧版不支持就回退）
            try:
                pt = torch.load(alpha_pt_path, map_location="cpu", weights_only=True)
            except TypeError:
                pt = torch.load(alpha_pt_path, map_location="cpu")
            A = pt.get("alpha", None)  # [B,3]
            if A is not None:
                A = A.detach().cpu().float()
                mean = A.mean(0).tolist()
                p25 = A.quantile(0.25, 0).tolist()
                p50 = A.quantile(0.50, 0).tolist()
                p75 = A.quantile(0.75, 0).tolist()
                alpha_stats = {
                    "mean": {"logs":mean[0], "metrics":mean[1], "traces":mean[2]},
                    "p25":  {"logs":p25[0],  "metrics":p25[1],  "traces":p25[2]},
                    "p50":  {"logs":p50[0],  "metrics":p50[1],  "traces":p50[2]},
                    "p75":  {"logs":p75[0],  "metrics":p75[1],  "traces":p75[2]},
                }
        except Exception:
            alpha_stats = None

    if alpha_stats:
        plot_alpha_summary(alpha_stats, os.path.join(out_dir, "alpha_summary.png"))
    return True
