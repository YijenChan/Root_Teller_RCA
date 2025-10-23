# -*- coding: utf-8 -*-
# agents/feedback_agent.py

"""
FeedbackAgent:
- 接收人类反馈（FEEDBACK_SUBMIT / FEEDBACK_SUGGEST）
- 存入 memory/feedback_store.jsonl
- 基于反馈生成一个简易的 profile patch
"""

import os, sys, json, time
from typing import Any, Dict

# --- 兼容包内 / 单文件两种运行方式 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))     # .../agents
ROOT = os.path.dirname(THIS_DIR)                         # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .base import BaseAgent, Message
except ImportError:
    from agents.base import BaseAgent, Message

try:
    from feedback.schema import parse_feedback, FeedbackRecord
except ImportError:
    # 单文件直跑时 fallback
    from ..feedback.schema import parse_feedback, FeedbackRecord


# --- 存储路径 ---
MEM_DIR = os.path.join(THIS_DIR, "memory")
MEM_FILE = os.path.join(MEM_DIR, "feedback_store.jsonl")
PROFILE_FILE = os.path.join(MEM_DIR, "profile.yaml")


def ensure_mem():
    os.makedirs(MEM_DIR, exist_ok=True)
    if not os.path.exists(MEM_FILE):
        open(MEM_FILE, "a", encoding="utf-8").close()


def append_feedback(rec: FeedbackRecord):
    ensure_mem()
    with open(MEM_FILE, "a", encoding="utf-8") as f:
        f.write(rec.to_json() + "\n")


def build_profile_patch(rec: FeedbackRecord) -> Dict[str, Any]:
    """
    根据反馈生成一个极简“profile patch”：
    - 分数低于阈值时，增加 explain_more & evidence_min_score
    - edits 中的路径按 section 聚合
    """
    patch: Dict[str, Any] = {"style": {}, "thresholds": {}, "sections": {}}

    if rec.score < 0.6:
        patch["style"]["explain_more"] = True
        patch["thresholds"]["evidence_min_score"] = 0.5

    for e in rec.edits:
        sec = e.path.split(".")[0] if e.path else "general"
        patch["sections"].setdefault(sec, []).append(
            {"path": e.path, "reason": e.reason}
        )

    return patch


class FeedbackAgent(BaseAgent):
    name = "feedback"

    def handle(self, msg: Message):
        if msg.type not in ("FEEDBACK_SUBMIT", "FEEDBACK_SUGGEST"):
            return self.reply_err(f"unknown message type: {msg.type}")

        try:
            payload = msg.payload or {}
            rec = parse_feedback(payload)
            append_feedback(rec)
            patch = build_profile_patch(rec)

            # 可选：广播给 reporter / predictor
            # self.bus.publish("reporter", "PROFILE_PATCH", {"patch": patch})

            return self.reply_ok({"stored": True, "patch": patch, "ts": rec.ts})
        except Exception as e:
            return self.reply_err(str(e))


# --- 允许单文件测试 ---
if __name__ == "__main__":
    # 构造一个最简消息模拟测试
    from agents.bus import EventBus

    bus = EventBus()
    fb = FeedbackAgent("feedback", bus=bus)

    payload = {
        "case_id": "demo_case",
        "score": 0.4,
        "comment": "证据不足，需要更详细的日志支持。",
        "edits": [{"path": "recommendations.steps[0]", "new": "补充日志"}],
        "context": {"demo": True},
    }

    msg = type("M", (), {"type": "FEEDBACK_SUBMIT", "payload": payload})
    print(fb.handle(msg))
