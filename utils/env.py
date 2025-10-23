# utils/env.py
from dotenv import load_dotenv
load_dotenv()

import os

def getenv_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")
