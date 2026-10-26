"""
Small text helpers shared by keyword-based routing and request parsing.

normalize_for_matching() is a character-for-character mapping (same
length as its input after diacritics are removed), so a position found
in the normalized text can be used to slice the original text. It is
meant for MATCHING only: show users the original text, not this one.
"""

from __future__ import annotations

import re

# Arabic diacritics (tashkeel), Quranic marks and tatweel.
_ARABIC_MARKS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۭـ]"
)

_CHAR_MAP = str.maketrans({
    # Alef variants -> bare alef.
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    # Alef maksura -> yeh; teh marbuta -> heh.
    "ى": "ي",
    "ة": "ه",
    # Arabic-Indic and Eastern Arabic-Indic digits -> ASCII digits.
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
})


def strip_arabic_marks(text: str) -> str:
    """Remove Arabic diacritics and tatweel."""
    return _ARABIC_MARKS.sub("", text)


def normalize_for_matching(text: str) -> str:
    """Lower-case and normalize Arabic letters/digits for keyword matching."""
    return strip_arabic_marks(text).lower().translate(_CHAR_MAP)


def prepare(text: str) -> tuple[str, str]:
    """
    Return (clean, norm): the text with Arabic marks removed, and its
    normalized form. Both have the same length, so indexes found in
    `norm` are valid in `clean`.
    """
    clean = strip_arabic_marks(text)
    norm = clean.lower().translate(_CHAR_MAP)

    if len(norm) != len(clean):
        # Extremely rare (a few exotic characters change length when
        # lower-cased). Fall back to one consistent string.
        clean = norm

    return clean, norm
