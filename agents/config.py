"""
Environment configuration for REX AI.

Secrets (TAVILY_API_KEY, REX_ADMIN_PASSCODE) are read from the process
environment, optionally populated from a .env file at the project root
(gitignored). Values are never logged or returned to the UI.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_loaded = False


def load_env() -> None:
    """Load PROJECT_ROOT/.env once; real environment variables win."""
    global _loaded
    if not _loaded:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        _loaded = True


def get_secret(name: str) -> str | None:
    """A configured, non-empty secret, or None."""
    load_env()
    value = os.environ.get(name, "").strip()
    return value or None


def get_setting(name: str, default: str) -> str:
    """A non-secret setting from the environment or .env, else the default."""
    return get_secret(name) or default


# REX Assistant's chat model (profile extraction and answer writing).
# qwen2.5:3b: in live tests on the project laptop (RTX 3060, 6 GB) it was
# as accurate as qwen2.5:7b, 2-3x faster, and fits fully on the GPU.
# Try another model with REX_LLM_MODEL in .env (e.g. REX_LLM_MODEL=qwen2.5:7b).
REX_DEFAULT_MODEL = "qwen2.5:3b"
REX_MODEL_SETTING = "REX_LLM_MODEL"


def rex_model() -> str:
    return get_setting(REX_MODEL_SETTING, REX_DEFAULT_MODEL)
