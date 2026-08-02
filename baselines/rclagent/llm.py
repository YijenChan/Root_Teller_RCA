"""
llm.py – OpenAI-compatible LLM client for RCLAgent.

Wraps an OpenAI-compatible chat completion endpoint with tool-calling
support. Each call returns a ``(content, tool_calls)`` pair; ``tool_calls``
is always a list (possibly empty). Transient network errors are retried
with exponential back-off. A text-based tool-call parser is included as a
fallback for ReAct-style models that emit tool calls inside the assistant
message body rather than in the structured ``tool_calls`` field.

API URL, key, and model are read from ``config.py`` so they can be set
via environment variables in one place.
"""

import json
import warnings

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import config

warnings.filterwarnings("ignore")

model = config.LLM_MODEL


# ── helpers ───────────────────────────────────────────────────────────────────

def get_tool_names(tools: list) -> list:
    return [t["function"]["name"] for t in (tools or [])]


def get_tool_from_content(content: str, tool_names: list) -> list:
    """
    Parse tool calls embedded in plain text (Qwen / ReAct format).
    Returns a list of tool-call dicts compatible with the OpenAI schema.
    """
    result = []
    lines  = content.split("\n")
    name   = None
    for line in lines:
        if line.startswith("Action: "):
            name = line[len("Action: "):].strip()
        elif line.startswith("✿FUNCTION✿: "):
            name = line[len("✿FUNCTION✿: "):].strip()
        elif line.startswith("✿ARGS✿: "):
            arguments = line[len("✿ARGS✿: "):].strip()
            if name and name in tool_names:
                result.append({
                    "id": f"text_{len(result)}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            name = None
        elif line.startswith("Action Input: "):
            arguments = line[len("Action Input: "):].strip()
            if name and name in tool_names:
                result.append({
                    "id": f"text_{len(result)}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            name = None
    return result


# ── public API ────────────────────────────────────────────────────────────────

def chat_api(prompts: list, tools) -> tuple:
    """
    Call the configured LLM and return (content: str, tool_calls: list).
    Always returns a 2-tuple; never None.
    """
    force_stream = getattr(config, "LLM_FORCE_STREAM", False)
    client = LLMClient(api_url=config.LLM_API_URL, api_key=config.LLM_API_KEY,
                        force_stream=force_stream)
    return client.generate(prompts, tools)


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """OpenAI-compatible chat completions client. Supports both regular and streaming APIs."""

    def __init__(self, api_url: str, api_key: str, force_stream: bool = False):
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.force_stream = force_stream

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompts: list, tools) -> tuple:
        """
        Call the API once.  Returns (content: str, tool_calls: list).
        """
        payload: dict = {"model": model, "messages": prompts}
        if tools:
            payload["tools"] = tools
        # Disable thinking mode for Qwen3/3.5 models (speeds up tool-call heavy workloads)
        if "qwen3" in model.lower():
            payload["enable_thinking"] = False

        if self.force_stream:
            return self._generate_stream(payload)

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            verify=False,
            timeout=180,
        )
        response.raise_for_status()

        message    = response.json()["choices"][0]["message"]
        content    = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        # Fallback: detect inline tool calls for models that use plain text.
        if not tool_calls and tools and content:
            tool_names = get_tool_names(tools)
            tool_calls = get_tool_from_content(content, tool_names)

        return content, tool_calls

    def _generate_stream(self, payload: dict) -> tuple:
        """SSE streaming fallback for APIs that require stream=true."""
        import json as _json
        payload["stream"] = True

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            verify=False,
            timeout=180,
            stream=True,
        )
        response.raise_for_status()

        content = ""
        tool_calls_map: dict = {}  # index → {id, function: {name, arguments}}

        for line in response.iter_lines():
            line = line.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = _json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})

                # Content
                if "content" in delta and delta["content"]:
                    content += delta["content"]

                # Tool calls (streamed incrementally)
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", f"stream_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        fn = tc.get("function", {})
                        if "name" in fn and fn["name"]:
                            tool_calls_map[idx]["function"]["name"] = fn["name"]
                        if "arguments" in fn and fn["arguments"]:
                            tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]
            except Exception:
                continue

        tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]

        # Fallback: detect inline tool calls
        if not tool_calls and payload.get("tools") and content:
            tool_names = get_tool_names(payload["tools"])
            tool_calls = get_tool_from_content(content, tool_names)

        return content, tool_calls
