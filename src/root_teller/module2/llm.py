from __future__ import annotations

import json
import http.client
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import canonical_json, stable_id


@dataclass(frozen=True)
class APISettings:
    api_key: str
    base_url: str


def load_api_settings(path: Path | None = None) -> APISettings:
    """Load an OpenAI-compatible endpoint without embedding credentials.

    Environment variables take precedence. A local configuration file is
    supported for backward-compatible private runs and must never be committed.
    """

    environment_key = os.environ.get("ROOTTELLER_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    environment_url = os.environ.get("ROOTTELLER_API_BASE") or os.environ.get(
        "OPENAI_BASE_URL"
    )
    if environment_key and environment_url:
        return APISettings(environment_key, environment_url.rstrip("/"))
    if path is None or not path.exists():
        raise ValueError(
            "Set ROOTTELLER_API_KEY and ROOTTELLER_API_BASE, or provide a "
            "local untracked API configuration file."
        )
    text = path.read_text(encoding="utf-8")
    key_match = re.search(r"sk-[A-Za-z0-9]+", text)
    url_match = re.search(r"https?://[^\s'\"]+", text)
    if not key_match or not url_match:
        raise ValueError(f"Could not parse API key/base URL from {path}")
    return APISettings(key_match.group(0), url_match.group(0).rstrip("/"))


class CachedJSONClient:
    def __init__(
        self,
        settings: APISettings,
        cache_dir: Path,
        model: str,
        temperature: float,
        timeout: int,
        max_retries: int,
    ) -> None:
        self.settings = settings
        self.cache_dir = cache_dir
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "retries": 0,
            "api_errors": 0,
            "last_api_error": None,
            "schema_errors": 0,
            "last_schema_error": None,
        }

    @staticmethod
    def _extract_json(text: str) -> object:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return json.loads(stripped)

    def complete(
        self,
        *,
        role: str,
        prompt_version: str,
        system_prompt: str,
        payload: dict[str, Any],
        validator,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        identity = stable_id(
            "llm",
            self.model,
            role,
            prompt_version,
            system_prompt,
            canonical_json(payload),
            length=32,
        )
        cache_file = self.cache_dir / f"{identity}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            result = validator(cached["validated_response"])
            self.stats["cache_hits"] += 1
            return result, {"cache_key": identity, "cached": True, "fallback": False}

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": canonical_json(
                        {
                            "prompt_version": prompt_version,
                            "role": role,
                            "input": payload,
                        }
                    ),
                },
            ],
        }
        encoded = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.stats["requests"] += 1
            request = urllib.request.Request(
                self.settings.base_url + "/chat/completions",
                data=encoded,
                method="POST",
                headers={
                    "Authorization": "Bearer " + self.settings.api_key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.load(response)
                text = raw["choices"][0]["message"]["content"]
                validated = validator(self._extract_json(text))
                cache_file.write_text(
                    json.dumps(
                        {
                            "cache_key": identity,
                            "model": self.model,
                            "role": role,
                            "prompt_version": prompt_version,
                            "validated_response": validated,
                            "usage": raw.get("usage", {}),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return validated, {
                    "cache_key": identity,
                    "cached": False,
                    "fallback": False,
                    "usage": raw.get("usage", {}),
                }
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.stats["schema_errors"] += 1
                self.stats["last_schema_error"] = str(error)[:240]
                last_error = error
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.RemoteDisconnected,
            ) as error:
                self.stats["api_errors"] += 1
                self.stats["last_api_error"] = (
                    f"{type(error).__name__}: {str(error)[:240]}"
                )
                last_error = error
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"{role} LLM call failed after retries: {type(last_error).__name__}")
