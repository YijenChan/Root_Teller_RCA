from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests


class InferenceError(RuntimeError):
    pass


def credentials(workspace: Path, model: str = "gpt-4o-mini") -> tuple[str, str]:
    key = os.environ.get("ROOTTELLER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("ROOTTELLER_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not key:
        raise InferenceError("Set ROOTTELLER_API_KEY or OPENAI_API_KEY before explicitly selecting live mode")
    return key, base.rstrip("/")


class LLM:
    def __init__(self, workspace: Path, cache: Path, model: str, mode: str,
                 max_calls: int = 200, timeout: int = 90, seed: int = 42,
                 max_output_tokens: int = 1800, max_total_tokens: int = 100000):
        self.workspace, self.cache, self.model, self.mode = workspace, cache, model, mode
        self.max_calls, self.timeout, self.seed = max_calls, timeout, seed
        self.calls, self.cache_hits, self.tokens = 0, 0, 0
        self.max_output_tokens, self.max_total_tokens = max_output_tokens, max_total_tokens
        self.requests = []

    def complete(self, role: str, messages: list[dict], temperature: float = 0, nonce: str = "") -> str:
        limit = min(self.max_output_tokens, 700) if role == "copilot_summary" else self.max_output_tokens
        body = {"model": self.model, "messages": messages}
        if role not in {"copilot_summary", "jsonregen_yaml"}:
            body["response_format"] = {"type": "json_object"}
        if self.model.startswith("gpt-5") and "chat" not in self.model:
            body.update({"max_completion_tokens": limit, "reasoning_effort": "minimal"})
        else:
            body.update({"temperature": temperature, "max_tokens": limit})
        identity = hashlib.sha256(json.dumps({"body": body, "role": role, "seed": self.seed,
                                             "nonce": nonce}, sort_keys=True).encode()).hexdigest()
        path = self.cache / "llm" / f"{identity}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            self.requests.append({"role": role, "cache_key": identity, "cached": True,
                                  "usage": saved.get("usage", {}), "model": saved.get("model")})
            return saved["content"]
        if self.mode != "live":
            raise InferenceError("Cache miss in replay; no model inference performed")
        if self.calls >= self.max_calls:
            raise InferenceError("Physical API call budget exhausted")
        estimated = len(json.dumps(messages, ensure_ascii=False)) // 3 + limit
        if self.tokens + estimated > self.max_total_tokens:
            raise InferenceError("Registered token budget would be exceeded")
        key, base = credentials(self.workspace, self.model)
        self.calls += 1
        begin = time.perf_counter()
        try:
            response = requests.post(base + "/chat/completions", json=body,
                                     headers={"Authorization": "Bearer " + key}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise InferenceError("API transport error: " + type(exc).__name__) from None
        if not response.ok:
            error_code = ""
            try:
                error = response.json().get("error", {})
                if isinstance(error, dict):
                    error_code = re.sub(r"[^A-Za-z0-9_ -]", "", str(error.get("code", "")))[:60]
            except (ValueError, AttributeError):
                pass
            raise InferenceError(f"API HTTP {response.status_code}" + (": " + error_code if error_code else ""))
        result = response.json()
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise InferenceError("API returned no assistant message") from None
        usage = result.get("usage", {})
        self.tokens += int(usage.get("total_tokens", 0))
        record = {"role": role, "cache_key": identity, "cached": False, "usage": usage,
                  "model": result.get("model"), "latency_seconds": time.perf_counter()-begin}
        record.update({"requested_temperature": temperature, "temperature_sent": "temperature" in body,
                       "output_token_limit": limit, "reasoning_effort": body.get("reasoning_effort"),
                       "finish_reason": result["choices"][0].get("finish_reason")})
        self.requests.append(record)
        if not isinstance(content, str) or not content.strip():
            record["status"] = "empty_output"
            failure_path = self.cache / "failed_llm" / f"{identity}.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            raise InferenceError("Empty model output; inspect completion/reasoning token limit")
        saved = {"content": content, "usage": usage, "model": result.get("model"),
                 "request_sha256": identity, "role": role, "messages": messages}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        return content

    def json(self, role: str, system: str, payload: dict, temperature: float = 0, nonce: str = "") -> dict:
        raw = self.complete(role, [{"role": "system", "content": system},
                                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], temperature, nonce)
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            # JsonRegen: YAML intermediary followed by JSON regeneration.
            yaml = self.complete("jsonregen_yaml", [{"role": "system", "content": "Convert the supplied structured answer to YAML without changing its information."},
                                                       {"role": "user", "content": raw}], nonce=nonce)
            repaired = self.complete("jsonregen_json", [{"role": "system", "content": "Convert this YAML to one valid JSON object, preserving its fields. Return JSON only."},
                                                           {"role": "user", "content": yaml}], nonce=nonce)
            return parse_json(repaired)


def parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    first, last = raw.find("{"), raw.rfind("}")
    result = json.loads(raw[first:last+1])
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result


def validate_ranking(value: dict, services: list[str]) -> list[str]:
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("Missing ordered ranking")
    size = min(5, len(services))
    if len(ranking) != size or len(set(ranking)) != len(ranking) or not set(ranking).issubset(services):
        unknown = sorted(set(ranking)-set(services))
        raise ValueError(f"Ranking must be a JSON array of {size} DISTINCT allowed services; unknown: {unknown}")
    return ranking
