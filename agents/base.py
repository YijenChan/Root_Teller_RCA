# -*- coding: utf-8 -*-
"""
agents/base.py
通用 Agent 抽象基类与消息结构。
- Message：标准消息对象（sender/recipient/type/payload）
- BaseAgent：所有代理需继承，提供 on_message 分发、send/ask/reply 便捷方法
- 仅依赖“总线接口”最小约定（见下），不直接实现总线：
  总线应至少实现：
    - register(agent)                     -> None
    - publish(message: Message)           -> None            # fire-and-forget
    - ask(message: Message, timeout=None) -> Any/Dict/...    # request-reply
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
import time
import traceback


# -------------------- 消息结构 --------------------
@dataclass
class Message:
    """统一的消息格式"""
    sender: str
    recipient: str
    type: str                    # 例如: "train_perception", "generate_report"
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())
    # 可选：相关上下文 / 追踪 id
    meta: Dict[str, Any] = field(default_factory=dict)


# -------------------- 异常类型 --------------------
class AgentError(Exception):
    pass


# -------------------- 基类 --------------------
class BaseAgent:
    """
    代理基类：
    - 子类实现 `handle(self, msg: Message) -> Any` 或者定义 `on_{type}(...)` 精细化回调
    - 通过 self.bus 与其他代理通信（send/ask）
    """
    def __init__(self, name: str, bus: Optional[Any] = None, logger: Optional[Callable[[str], None]] = None):
        self.name = name
        self.bus = None
        self.logger = logger or (lambda s: print(f"[{self.name}] {s}"))
        if bus is not None:
            self.bind(bus)

    # --------- 生命周期/绑定 ---------
    def bind(self, bus: Any):
        """绑定到总线并注册自己"""
        self.bus = bus
        if hasattr(self.bus, "register"):
            self.bus.register(self)
        self.log("bound to bus.")

    # --------- 日志便捷 ---------
    def log(self, msg: str):
        self.logger(str(msg))

    # --------- 消息入口 ---------
    def on_message(self, msg: Message) -> Any:
        """
        总线调用此入口。默认分发策略：
        1) 如果子类实现了 `on_{msg.type}(msg)`，优先调用
        2) 否则回退到 `handle(msg)`
        """
        try:
            # on_{type} 精确分发
            meth_name = f"on_{msg.type}"
            if hasattr(self, meth_name) and callable(getattr(self, meth_name)):
                return getattr(self, meth_name)(msg)

            # 通用处理
            return self.handle(msg)
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"ERROR while handling '{msg.type}': {e}\n{tb}")
            raise

    # --------- 需要子类实现 ---------
    def handle(self, msg: Message) -> Any:
        """子类可重写；或实现 on_{type} 变体。"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement handle() or on_<type>()")

    # --------- 与总线交互（便捷） ---------
    def send(self, recipient: str, mtype: str, payload: Dict[str, Any] | None = None):
        """fire-and-forget 推送"""
        if self.bus is None:
            raise AgentError("Agent is not bound to a bus.")
        message = Message(sender=self.name, recipient=recipient, type=mtype, payload=payload or {})
        if not hasattr(self.bus, "publish"):
            raise AgentError("Bus has no 'publish' method.")
        self.bus.publish(message)

    def ask(self, recipient: str, mtype: str, payload: Dict[str, Any] | None = None, timeout: Optional[float] = None) -> Any:
        """request-reply 同步请求，返回对方结果"""
        if self.bus is None:
            raise AgentError("Agent is not bound to a bus.")
        if not hasattr(self.bus, "ask"):
            raise AgentError("Bus has no 'ask' method.")
        message = Message(sender=self.name, recipient=recipient, type=mtype, payload=payload or {})
        return self.bus.ask(message, timeout=timeout)

    def reply_ok(self, data: Any = None) -> Dict[str, Any]:
        """统一成功返回结构，可由上层总线透传"""
        return {"ok": True, "data": data}

    def reply_err(self, err: str, data: Any = None) -> Dict[str, Any]:
        """统一失败返回结构"""
        return {"ok": False, "error": err, "data": data}
