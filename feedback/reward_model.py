# -*- coding: utf-8 -*-
# feedback/reward_model.py
from typing import Dict, Any

"""
最小可用“奖励模型”：
- 规则项：命中/一致性/margin 提升 → 奖励；人工评分 → 线性叠加
- 占位项：可替换为线性回归/小 MLP（未来）
"""

def rule_reward(feedback: Dict[str, Any]) -> float:
    """
    feedback: {
      "score": 0~1,
      "context": {
         "chain_jaccard": float,
         "agree_ratio": float,   # Verifier 反向一致性
         "confidence_margin": float, # top1-top2
      }
    }
    """
    base = float(feedback.get("score", 0.0))
    ctx = feedback.get("context", {}) or {}

    chain = float(ctx.get("chain_jaccard", 0.0))
    agree = float(ctx.get("agree_ratio", 0.0))
    margin = float(ctx.get("confidence_margin", 0.0))

    # 规则加权：数值可根据实验再调
    reward = (
        0.6 * base +
        0.2 * chain +
        0.15 * agree +
        0.05 * max(0.0, margin)
    )
    # 归一到 [0,1]
    reward = max(0.0, min(1.0, reward))
    return reward
