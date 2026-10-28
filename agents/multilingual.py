"""
Compatibility wrapper: language handling lives in knowledge_base/language.py
(the single implementation). This module keeps the orchestrator's
detect_language() contract ("ar" or "en", never None) and the Arabizi
helpers' original import path.
"""

from __future__ import annotations

from knowledge_base.language import (  # noqa: F401  (re-exported)
    contains_arabic as contains_arabic_script,
    detect_conversation_language,
    looks_like_arabizi,
    transliterate_arabizi,
)


def detect_language(text: str) -> str:
    """
    "ar" for Arabic script or confident Arabizi, otherwise "en".
    Short or neutral messages fall back to "en" here; REX Assistant keeps
    the conversation's previous language for those instead.
    """
    return detect_conversation_language(text) or "en"
