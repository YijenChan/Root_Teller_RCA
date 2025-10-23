# -*- coding: utf-8 -*-
# agents/collector_agent.py
"""
CollectorAgent
- 通过 EventBus 接收采集请求（COLLECT_REQUEST）
- 调用 tools/collectors 下的 fs_* 采集器（本地模拟）
- 将原始证据落盘到 outputs/perception/raw_inputs/ 下，并返回索引
- 兼容包内/单文件直跑
"""

import os, sys, json, time
from typing import Any, Dict, List

# --- 兼容包内/单文件 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../agents
ROOT = os.path.dirname(THIS_DIR)                         # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .base import BaseAgent, Message
except ImportError:
    from agents.base import BaseAgent, Message

# 尝试相对导入 tools/collectors
try:
    from tools.collectors.fs_logs import collect_logs
    from tools.collectors.fs_metrics import collect_metrics
    from tools.collectors.fs_traces import collect_traces
except Exception:
    # 单文件直跑兜底（若包结构不同可按需调整）
    from ..tools.collectors.fs_logs import collect_logs
    from ..tools.collectors.fs_metrics import collect_metrics
    from ..tools.collectors.fs_traces import collect_traces


OUT_RAW_DIR = os.path.join(ROOT, "outputs", "perception", "raw_inputs")
os.makedirs(OUT_RAW_DIR, exist_ok=True)


def _write_json(obj: Dict[str, Any], name: str) -> str:
    path = os.path.join(OUT_RAW_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


class CollectorAgent(BaseAgent):
    name = "collector"

    def handle(self, msg: Message):
        """
        COLLECT_REQUEST payload 约定：
        {
          "case_id": "single_case",
          "need": ["logs","metrics","traces"],   # 可任意子集
          "time_window": {"start": "...", "end": "..."}  # 可选
          "filters": {...}                       # 可选，采集器自解释
        }
        """
        if msg.type != "COLLECT_REQUEST":
            return self.reply_err(f"unknown message type: {msg.type}")

        payload = msg.payload or {}
        case_id = payload.get("case_id", "single_case")
        need: List[str] = payload.get("need", ["logs", "metrics", "traces"])
        t_win = payload.get("time_window", None)
        filters = payload.get("filters", {})

        try:
            results: Dict[str, Any] = {"case_id": case_id, "ts": time.time(), "artifacts": {}}

            if "logs" in need:
                logs = collect_logs(time_window=t_win, filters=filters.get("logs"))
                p = _write_json({"case_id": case_id, "kind": "logs", "data": logs}, f"{case_id}_logs.json")
                results["artifacts"]["logs"] = {"path": p, "count": len(logs)}

            if "metrics" in need:
                mets = collect_metrics(time_window=t_win, filters=filters.get("metrics"))
                p = _write_json({"case_id": case_id, "kind": "metrics", "data": mets}, f"{case_id}_metrics.json")
                results["artifacts"]["metrics"] = {"path": p, "count": len(mets)}

            if "traces" in need:
                trs = collect_traces(time_window=t_win, filters=filters.get("traces"))
                p = _write_json({"case_id": case_id, "kind": "traces", "data": trs}, f"{case_id}_traces.json")
                results["artifacts"]["traces"] = {"path": p, "count": len(trs)}

            return self.reply_ok(results)

        except Exception as e:
            return self.reply_err(str(e))


# --- 单文件测试 ---
if __name__ == "__main__":
    from agents.bus import EventBus
    bus = EventBus()
    c = CollectorAgent("collector", bus=bus)

    demo_msg = type("M", (), {
        "type": "COLLECT_REQUEST",
        "payload": {
            "case_id": "single_case",
            "need": ["logs","metrics","traces"],
            "time_window": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T02:00:00Z"},
            "filters": {"logs": {"level": ["ERROR","WARN"]}}
        }
    })
    print(c.handle(demo_msg))
