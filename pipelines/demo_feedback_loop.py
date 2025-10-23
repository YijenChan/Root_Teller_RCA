# -*- coding: utf-8 -*-
# pipelines/demo_feedback_loop.py
"""
演示：读取 Stage-B/C 产物 -> 构造一条人类反馈 -> 提交到 FeedbackAgent
打印返回的 reward 与合并后的 profile，模拟“记忆→再生成”的准备动作。
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.bus import EventBus
from agents.feedback_agent import FeedbackAgent

REASON_JSON = os.path.join(ROOT, "outputs", "reasoning", "single_case.json")
REPORT_RAW  = os.path.join(ROOT, "outputs", "reports",   "llm_report_raw.json")

def _load_reasoning_summary():
    if not os.path.exists(REASON_JSON):
        return {}
    with open(REASON_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("eval_summary", {}) or {}

def _load_report_raw():
    if not os.path.exists(REPORT_RAW):
        return {}
    try:
        with open(REPORT_RAW, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # 兼容：你的 llm_report_raw.json 结构目前是 {ok, error, messages}，可能没有 pred_root
        return obj or {}
    except Exception:
        return {}

def main():
    bus = EventBus()
    fb = FeedbackAgent("feedback", bus=bus)

    ev = _load_reasoning_summary()
    rep = _load_report_raw()

    # 从评测结果提炼一部分 context
    topk = ev.get("topk_scores") or []
    if len(topk) >= 2:
        margin = float(topk[0]) - float(topk[1])
    else:
        margin = 0.0

    context = {
        "chain_jaccard": float(ev.get("chain_jaccard", 0.0)),
        "confidence_margin": float(margin),
        # 你的 llm_report_raw.json 当前未携带 pred_root/pred_chain，这里兼容留空
        "pred_root": ev.get("pred_root"),
        "pred_chain": ev.get("chain_pred"),
    }

    # 构造一条“人类反馈”
    payload = {
        "case_id": "single_case",
        "score": 0.72,  # 示例分数
        "comment": "请把证据引用的节点ID和时间窗列清楚，处置建议更保守些。",
        "edits": [
            {"path": "recommendations.steps[0]", "old": None, "new": "先对受影响主机做网络隔离", "reason": "降低外溢风险"},
            {"path": "evidence.items", "old": None, "new": "展示触发告警的原始日志窗口", "reason": "可溯源性"}
        ],
        "context": context
    }

    res = fb.handle(type("M", (), {"type": "FEEDBACK_SUBMIT", "payload": payload}))
    print("Feedback submit result:", res)

if __name__ == "__main__":
    main()
