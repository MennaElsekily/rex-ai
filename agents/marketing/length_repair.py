"""
Deterministic repair for a channel's or the strategy's LENGTH-only
problems: this module only ever touches text whose sole remaining
problem (after the model's own one-shot repair in generator.py) is
being too long or too short for its field's limit (validation.py's
"length" code). A claim, structure, repeated-text or any other kind of
problem is never touched here -- that stays the model's job, or the
campaign is refused, exactly as before.

Too long: trimmed at the last sentence or word boundary that still fits,
never mid-word, so the meaning of what remains is unchanged.
Too short: extended only with a fact already in the approved course
(currently its stated duration), never invented text; if no such fact
is available, or adding it still does not reach the minimum, the field
is left as it was and the campaign still fails the recheck -- the same
outcome as today, not a worse one.
"""

from __future__ import annotations

import re
from typing import Any

from agents.marketing import schemas
from agents.marketing.facts import Allowed

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟،,،])\s+")
_TRAILING_PUNCTUATION = " ,،.!؟?؛;:"


def _trim_to_fit(text: str, limit: int) -> str:
    """The longest prefix of whole sentences (else whole words) that fits within limit characters."""
    text = text.strip()
    if len(text) <= limit:
        return text

    kept = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        candidate = f"{kept} {sentence}".strip() if kept else sentence
        if len(candidate) <= limit:
            kept = candidate
        else:
            break
    if kept and len(kept) >= limit * 0.5:
        return kept.rstrip(_TRAILING_PUNCTUATION)

    words = text[: limit + 1].split(" ")
    kept_words = words[:-1] if len(words) > 1 else []
    # A cut must not end on a word that only joins to what was cut off ("... JavaScript و", "... SQL and").
    while len(kept_words) > 1 and kept_words[-1].strip(_TRAILING_PUNCTUATION).casefold() in _DANGLING:
        kept_words.pop()
    trimmed = " ".join(kept_words) if kept_words else text[:limit]
    return trimmed.rstrip(_TRAILING_PUNCTUATION) or text[:limit]


# Words a trimmed text must not end on: they only join to the words that were cut off.
_DANGLING = frozenset("""
and or with for to of in on by the a an as at from your our
و أو او ب ل في من على إلى الى مع عن ثم أن ان التي الذي
""".split())


def _hours_phrase(allowed: Allowed, language: str) -> str:
    if not allowed.hours:
        return ""
    hours = sorted(allowed.hours)[0]
    return f"{hours:g} hours" if language == "en" else f"{hours:g} ساعة"


def _extend_to_fit(text: str, low: int, high: int, allowed: Allowed, language: str) -> str:
    """Appends one already-verified course fact (never invented) if that reaches the minimum without passing the maximum."""
    text = text.strip()
    if len(text) >= low:
        return text
    phrase = _hours_phrase(allowed, language)
    if not phrase or phrase.casefold() in text.casefold():
        return text
    candidate = f"{text} ({phrase})" if language == "en" else f"{text} ({phrase})"
    return candidate if low <= len(candidate) <= high else text


def _only_length_problems(problems: list[dict[str, str]]) -> bool:
    return bool(problems) and all(p.get("code") == "length" for p in problems)


def repair_variation_lengths(channel: str, variation: dict[str, Any], allowed: Allowed, language: str) -> dict[str, Any]:
    """A copy of variation with every text field mechanically brought inside its length limit."""
    fixed = dict(variation)
    for name, (low, high) in schemas.LIMITS[channel].items():
        value = fixed.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if len(value) > high:
            value = _trim_to_fit(value, high)
        elif len(value) < low:
            value = _extend_to_fit(value, low, high, allowed, language)
        fixed[name] = value
    return fixed


def repair_channel_lengths(
    channel: str, variations: list[dict[str, Any]], problems: list[dict[str, str]], allowed: Allowed, language: str,
) -> list[dict[str, Any]]:
    """Repairs every variation's lengths when the channel's only remaining problems are length problems; else unchanged."""
    if not _only_length_problems(problems):
        return variations
    return [repair_variation_lengths(channel, v, allowed, language) if isinstance(v, dict) else v for v in variations]


def repair_strategy_lengths(strategy: dict[str, Any], problems: list[dict[str, str]]) -> dict[str, Any]:
    """The strategy's long-text fields trimmed to MAX_TEXT_FIELD when that is the only remaining problem."""
    from agents.marketing.validation import MAX_TEXT_FIELD

    if not _only_length_problems(problems):
        return strategy
    fixed = dict(strategy)
    for key in ("objective", "main_message", "call_to_action"):
        value = fixed.get(key)
        if isinstance(value, str) and len(value) > MAX_TEXT_FIELD:
            fixed[key] = _trim_to_fit(value, MAX_TEXT_FIELD)
    return fixed
