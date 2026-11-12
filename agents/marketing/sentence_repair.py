"""
Deterministic repair that REMOVES the sentences the checks flag -- an unsupported claim ("No prerequisites, just a solid
foundation in HTML." for a course that lists prerequisites), characters of another writing system, or English glued to
Arabic letters -- when the model's own one-shot repair (generator.py) left them in. Before that, a one-letter Arabic
prefix glued to an English term ("وNode.js") only gets the missing space ("و Node.js").

Nothing is added or reworded: a flagged sentence is dropped and every other sentence is kept exactly as written. The
result is checked again with the unchanged rules (validation.py); if it is now too short, empty, too similar to another
variation or still wrong in any way, the campaign is refused exactly as before. A text with nothing left after the
removal is not changed at all.

Real runs (qwen2.5:3b, 2026-11-12): ad copy ended with "..., no prerequisites." or "No prior experience needed, just
...", on every attempt, for courses whose curriculum lists prerequisites; Arabic e-mails contained Chinese characters.
"""

from __future__ import annotations

import re
from typing import Any

from agents.marketing import schemas
from agents.marketing.facts import Allowed
from agents.marketing.validation import _RISKY, text_problems

# The problems a sentence can be removed for: claims the approved course does not support, and text in the wrong script.
REMOVABLE = frozenset({code for code, _, _ in _RISKY} | {
    "accreditation_claim", "hours_mismatch", "number_claim", "foreign_script", "foreign_word"})

# A sentence with its closing punctuation and following space ("." "!" "?" "؟", or a line break).
_SENTENCE = re.compile(r"[^.!?؟\n]+(?:[.!?؟]+|\n|$)\s*")


def _flagged(text: str, allowed: Allowed, **options: Any) -> bool:
    return any(p.get("code") in REMOVABLE for p in text_problems(text, allowed, **options))


# A one-letter Arabic prefix (و "and", ب "with", ل "for", ف, ك) written against an English technical term: "وNode.js".
# Only a standalone prefix (start of text or after a space or punctuation) is separated; nothing else changes.
_GLUED_PREFIX = re.compile(r"(?<![ء-ي])([وبلفك])(?=[A-Za-z])")


def spaced_prefixes(text: Any) -> Any:
    """ "وNode.js" -> "و Node.js": the same words, with the space the checks require between Arabic and an English term."""
    return _GLUED_PREFIX.sub(lambda m: m.group(1) + " ", text) if isinstance(text, str) else text


def without_flagged_sentences(text: str, allowed: Allowed) -> str:
    """text without the sentences that have a removable problem; unchanged if none does, or if nothing would be left."""
    if not isinstance(text, str) or not _flagged(text, allowed):
        return text
    sentences = [s for s in _SENTENCE.findall(text) if s.strip()]
    kept = [s for s in sentences if not _flagged(s, allowed)]
    if not kept or len(kept) == len(sentences):
        return text
    return "".join(kept).strip()


def repair_channel_sentences(channel: str, variations: list[Any], allowed: Allowed) -> list[Any]:
    """A copy of the variations with the flagged sentences (and flagged hashtags) removed."""
    fixed = []
    for variation in variations:
        if not isinstance(variation, dict):
            fixed.append(variation)
            continue
        variation = dict(variation)
        for name in schemas.LIMITS[channel]:
            variation[name] = without_flagged_sentences(spaced_prefixes(variation.get(name)), allowed)
        if channel == "social_captions" and isinstance(variation.get("hashtags"), list):
            variation["hashtags"] = [t for t in variation["hashtags"]
                                     if not (isinstance(t, str) and _flagged(t, allowed, hashtag=True))]
        fixed.append(variation)
    return fixed


def repair_strategy_sentences(strategy: dict[str, Any], allowed: Allowed) -> dict[str, Any]:
    """A copy of the strategy with flagged sentences removed, and flagged steps dropped while three or more remain."""
    fixed = dict(strategy)
    for key in ("objective", "main_message", "call_to_action"):
        fixed[key] = without_flagged_sentences(spaced_prefixes(fixed.get(key)), allowed)
    steps = fixed.get("content_sequence")
    if isinstance(steps, list):
        steps = [without_flagged_sentences(spaced_prefixes(s), allowed) for s in steps]
        clean = [s for s in steps if not (isinstance(s, str) and _flagged(s, allowed, enumerated=True))]
        fixed["content_sequence"] = clean if len(clean) >= 3 else steps
    return fixed
