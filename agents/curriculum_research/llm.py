"""
The language model for Agent 3 (LangChain ChatOllama) and one helper for
structured calls.

The model is configurable with CURRICULUM_LLM_MODEL; the default is REX's
model (agents/config.rex_model(), qwen2.5:3b) until a measured comparison
justifies another (docs/CURRICULUM_AGENT_ARCHITECTURE.md §17).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from functools import lru_cache
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from agents.config import get_setting, rex_model

logger = logging.getLogger(__name__)

MODEL_SETTING = "CURRICULUM_LLM_MODEL"
# Curriculum prompts carry evidence and a partial curriculum: more room
# than REX's 8192 tokens.
CONTEXT_TOKENS = 12288
# One retry when the model's answer does not fit the schema.
STRUCTURED_ATTEMPTS = 2
# Limits so one call can never hang a request: a small model can loop
# while writing structured output (seen with qwen2.5:3b on an Arabic
# outline: no answer after 34 minutes). An answer cut at the token limit
# fails schema validation and is reported as a model error.
MAX_OUTPUT_TOKENS = 4096
CALL_TIMEOUT_SECONDS = 240

T = TypeVar("T", bound=BaseModel)


class CurriculumModelError(RuntimeError):
    """The model could not be reached or kept returning unusable output."""


def curriculum_model() -> str:
    return get_setting(MODEL_SETTING, "") or rex_model()


def _supports_thinking(model: str) -> bool:
    try:
        request = urllib.request.Request("http://localhost:11434/api/show", data=json.dumps({"model": model}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return "thinking" in (json.load(response).get("capabilities") or [])
    except Exception:
        return False


@lru_cache(maxsize=4)
def get_llm(model: str):
    """One ChatOllama client per model; thinking models answer directly."""
    from langchain_ollama import ChatOllama

    options: dict[str, Any] = {"model": model, "temperature": 0, "num_ctx": CONTEXT_TOKENS,
                               "num_predict": MAX_OUTPUT_TOKENS,
                               "client_kwargs": {"timeout": CALL_TIMEOUT_SECONDS}}
    if _supports_thinking(model):
        options["reasoning"] = False
    return ChatOllama(**options)


def structured(llm: Any, schema: Type[T], messages: list[Any], step: str) -> T:
    """
    Ask ``llm`` for an answer in ``schema`` (LangChain structured output),
    retrying once. Raises CurriculumModelError instead of returning junk.
    """
    runnable = llm.with_structured_output(schema)
    last_error: Exception | None = None
    for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
        try:
            result = runnable.invoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            last_error = ValueError(f"unexpected answer type {type(result).__name__}")
        except Exception as exc:  # parse errors, connection errors
            last_error = exc
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                break  # a second try would only wait as long again
        logger.warning("Curriculum step %s attempt %d failed: %s", step, attempt, last_error)
    raise CurriculumModelError(f"The language model failed at the {step} step ({type(last_error).__name__}).")
