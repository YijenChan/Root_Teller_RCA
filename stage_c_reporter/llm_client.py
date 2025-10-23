# -*- coding: utf-8 -*-
"""
stage_c_reporter/llm_client.py

Lightweight OpenAI-compatible LLM client:
- Loads configuration from YAML (supports ${VAR} environment placeholders)
- Automatically loads .env if present
- Ensures HTTPS scheme for api_base when missing
"""

import os
import sys
import re
import yaml
import requests
from typing import List, Dict, Any

# Allow both "script" and "package" execution modes
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../stage_c_reporter
ROOT = os.path.dirname(THIS_DIR)                        # .../agentic_rca
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Attempt to auto-load environment variables
try:
    import utils.env  # utils/env.py internally calls load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _expand_env(s: Any) -> str:
    """
    Expand ${VAR} and $VAR environment variables manually.
    Windows `os.path.expandvars` has inconsistent ${VAR} behavior,
    so this function provides a unified implementation.
    """
    if s is None:
        return ""
    s = str(s)

    # Expand ${VAR} placeholders
    def repl(m):
        key = m.group(1)
        return os.environ.get(key, m.group(0))
    s = re.sub(r"\$\{([^}]+)\}", repl, s)

    # Expand $VAR / %VAR%
    s = os.path.expandvars(s)
    return s


def _ensure_scheme(url: str) -> str:
    """Ensure the given URL starts with a valid scheme (defaults to https://)."""
    if not url:
        return url
    url = url.strip()
    if "://" not in url:
        return "https://" + url.lstrip("/")
    return url


# ---------------------------------------------------------------------
# Main LLM client
# ---------------------------------------------------------------------
class OpenAICompatibleClient:
    """
    Minimal OpenAI-compatible REST client.
    - POST {api_base}/chat/completions
    - Headers: Authorization: Bearer <api_key>
    - Body: {model, messages, temperature, max_tokens}
    """

    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = _ensure_scheme(_expand_env(api_base)).rstrip("/")
        self.api_key = _expand_env(api_key)
        self.model = _expand_env(model or "gpt-4o-mini")

        if not self.api_base:
            raise ValueError("api_base not configured or empty")
        if not self.api_key:
            raise ValueError("api_key not configured or empty")

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request to the LLM API."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        response = self.session.post(url, json=payload, timeout=60)
        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"LLM HTTP {response.status_code} - {response.text[:500]}"
            ) from e

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response: {data}") from e


# ---------------------------------------------------------------------
# YAML-based builder
# ---------------------------------------------------------------------
def build_client_from_cfg(cfg_path: str) -> OpenAICompatibleClient:
    """
    Build an LLM client from a YAML configuration file.

    Expected YAML structure:
    llm:
      api_base: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}
      model: gpt-4o-mini
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    llm = cfg.get("llm", {}) or {}

    # Prefer YAML fields (with ${ENV} expansion),
    # fallback to environment variables if missing.
    api_base = _expand_env(llm.get("api_base", "")) or os.getenv("OPENAI_API_BASE", "")
    api_key = _expand_env(llm.get("api_key", "")) or os.getenv("OPENAI_API_KEY", "")
    model = _expand_env(llm.get("model", "gpt-4o-mini"))

    return OpenAICompatibleClient(api_base=api_base, api_key=api_key, model=model)
