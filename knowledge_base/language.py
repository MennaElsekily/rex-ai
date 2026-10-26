"""
Generic Arabic/English language-handling utilities shared by the RAG
answer generator and Agent 1.

This module keeps four concepts explicitly separate, per the REX AI
architecture:
    - the original user query (never mutated)
    - the detected language of that query
    - the preferred response language (may differ from the detected one)
    - the retrieval query (translated to English only when required)
"""

import re
from typing import Optional

from knowledge_base.translator import QueryTranslator


_ARABIC_RANGES = (
    ("؀", "ۿ"),
    ("ݐ", "ݿ"),
    ("ࢠ", "ࣿ"),
)

_LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")

_ARABIC_LABELS = {"ar", "arabic", "العربية"}
_ENGLISH_LABELS = {"en", "english"}


def contains_arabic(text: str) -> bool:
    """Detect whether a string contains Arabic-script characters."""
    return any(
        any(start <= character <= end for start, end in _ARABIC_RANGES)
        for character in text
    )


def detect_language(text: str) -> str:
    """
    Classify text as "ar", "en", "mixed", "other", or "empty".

    This is a lightweight script-based classifier rather than a general
    language identifier: the project requires distinguishing Arabic from
    English only, so no extra model is loaded to make this decision.
    """
    if not isinstance(text, str) or not text.strip():
        return "empty"

    has_arabic = contains_arabic(text)
    has_latin = bool(_LATIN_LETTER_PATTERN.search(text))

    if has_arabic and has_latin:
        return "mixed"
    if has_arabic:
        return "ar"
    if has_latin:
        return "en"
    return "other"


_translator_singleton: Optional[QueryTranslator] = None


def _get_translator() -> QueryTranslator:
    """Reuse one QueryTranslator instance; the model itself loads lazily."""
    global _translator_singleton

    if _translator_singleton is None:
        _translator_singleton = QueryTranslator()

    return _translator_singleton


def build_retrieval_query(original_query: str) -> tuple[str, str]:
    """
    Decide the text used for retrieval and report the detected language.

    Returns (retrieval_query, detected_language).

    - Arabic or mixed-language text is translated to English, since the
      embedding model is English-oriented. Mixed text is translated too,
      because an untranslated Arabic portion would not match the
      English-only corpus.
    - English or non-Arabic text is passed through unchanged, avoiding
      unnecessary translation calls.
    - If translation fails (model unavailable, runtime error), the
      original text is used for retrieval rather than raising, so a
      translation outage degrades retrieval quality instead of breaking
      the request entirely.
    """
    detected = detect_language(original_query)

    if detected in ("ar", "mixed"):
        try:
            translated = _get_translator().translate_to_english(
                original_query
            )
        except Exception:
            return original_query, detected

        return (translated.strip() if translated else original_query), detected

    return original_query, detected


def resolve_response_language(
    detected_language: str,
    preferred_language: str | None,
) -> str | None:
    """
    Resolve which language the final answer should be written in.

    An explicit preferred_language always takes precedence over the
    detected query language, since a user may ask in one language while
    having chosen to receive responses in another. Returns "ar", "en",
    or None when neither can be determined (e.g. "other"/"empty"
    detected language with no preference supplied).
    """
    if preferred_language:
        normalized = preferred_language.strip().lower()

        if normalized in _ARABIC_LABELS:
            return "ar"
        if normalized in _ENGLISH_LABELS:
            return "en"

    if detected_language in ("ar", "en"):
        return detected_language

    return None
