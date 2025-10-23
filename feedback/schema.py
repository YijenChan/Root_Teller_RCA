# -*- coding: utf-8 -*-
# feedback/schema.py
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import time, json

@dataclass
class EditItem:
    path: str                   # JSONPath 或 "section.key" 之类定位（占位即可）
    old: Optional[Any]          # 原内容（可选）
    new: Any                    # 新内容
    reason: str = ""            # 修改原因（可选）

@dataclass
class FeedbackRecord:
    # 最小可用的人类反馈结构
    case_id: str                                # 一次 RCA 运行的标识（文件名/时间戳等）
    score: float                                # 人类评分 [0,1]
    comment: str                                # 文字评价
    edits: List[EditItem] = field(default_factory=list)     # 结构化修订（可空）
    context: Dict[str, Any] = field(default_factory=dict)   # 附带上下文（可空）
    ts: float = field(default_factory=lambda: time.time())  # 记录时间

    def to_json(self) -> str:
        # 保证 score 在 [0,1] 内
        s = max(0.0, min(1.0, float(self.score)))
        d = asdict(self)
        d["score"] = s
        return json.dumps(d, ensure_ascii=False)

def parse_feedback(payload: Dict[str, Any]) -> FeedbackRecord:
    """将任意 dict 整理为 FeedbackRecord（宽松解析、带裁剪）"""
    payload = payload or {}
    edits_raw = payload.get("edits") or []
    edits: List[EditItem] = []
    for e in edits_raw:
        e = e or {}
        edits.append(EditItem(
            path=str(e.get("path", "")),
            old=e.get("old"),
            new=e.get("new"),
            reason=str(e.get("reason", "")),
        ))

    # 分数裁剪
    try:
        score = float(payload.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    return FeedbackRecord(
        case_id=str(payload.get("case_id", "single_case")),
        score=score,
        comment=str(payload.get("comment", "")),
        edits=edits,
        context=payload.get("context", {}) or {}
    )
