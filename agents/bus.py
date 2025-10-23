# -*- coding: utf-8 -*-
from typing import Dict, Optional
from .base import Message

class EventBus:
    """
    极简同步事件总线：
    - register(agent): 注册代理实例（不额外打印日志，避免重复）
    - publish(message): fire-and-forget，忽略返回值与异常
    - ask(message, timeout=None): request-reply，同步返回对方处理结果
    """
    def __init__(self):
        self.registry: Dict[str, object] = {}

    def register(self, agent) -> None:
        self.registry[agent.name] = agent
        # 不在这里打印日志，避免和 BaseAgent.bind() 的日志重复
        # 如果想保留，可以改成：
        # agent.log("registered on bus.")

    def publish(self, msg: Message) -> None:
        """单向投递（忽略返回值）；若收件人不存在则静默丢弃。"""
        agent = self.registry.get(msg.recipient)
        if agent is None:
            # 也可选择打印告警：print(f"[bus] recipient not found: {msg.recipient}")
            return
        try:
            agent.on_message(msg)
        except Exception as e:
            # fire-and-forget 场景忽略异常（必要时可记录日志）
            # print(f"[bus] publish error to {msg.recipient}: {e}")
            pass

    def ask(self, msg: Message, timeout: Optional[float] = None):
        """
        同步请求（request-reply）。目前未实现真正的超时机制，
        仅为了兼容 BaseAgent.ask(...) 的参数签名。
        """
        agent = self.registry.get(msg.recipient)
        if agent is None:
            return {"ok": False, "error": f"Recipient not found: {msg.recipient}"}
        # 这里直接同步调用；如需超时控制，可在未来接入线程/协程或队列实现
        return agent.on_message(msg)
