"""
The single language-handling module for REX AI (orchestrator, REX
Assistant and the RAG pipeline all use it; agents/multilingual.py only
re-exports it for older imports).

Two different questions are answered here, on purpose:

- Which LANGUAGE is the user speaking? detect_conversation_language():
  "ar" (Arabic script, Egyptian Arabic, mixed Arabic/English, or
  confident Arabizi), "en", or None when the message is too short or
  neutral to tell ("ok", "yes", "5000"). Callers keep the conversation's
  previous language in that case (resolve_conversation_language()).
- Which SCRIPT does a text use, for retrieval? detect_language():
  "ar"/"en"/"mixed"/"other"/"empty". prepare_retrieval_query() uses it to
  decide whether a query must be translated to English for the
  English-only embedding model, and reports the translation outcome.

Kept separate: the original user query (never mutated), the detected
language, an explicit language preference (only set when the user asks
for one), and the retrieval query.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_ARABIC_RANGES = (
    ("؀", "ۿ"),  # Arabic
    ("ݐ", "ݿ"),  # Arabic Supplement
    ("ࢠ", "ࣿ"),  # Arabic Extended-A
    ("ﭐ", "﷿"),  # Arabic Presentation Forms-A
    ("ﹰ", "﻿"),  # Arabic Presentation Forms-B
)

_LATIN_LETTER_PATTERN = re.compile(r"[A-Za-z]")

# Runs of Latin-script words and numbers inside an Arabic query
# ("Python Basics", "PY-101", "3000 EGP").
_LATIN_SPAN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9.+#&'/_-]*(?:[ \t]+[A-Za-z0-9][A-Za-z0-9.+#&'/_-]*)*"
)
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9']*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_ARABIC_LABELS = {"ar", "arabic", "العربية"}
_ENGLISH_LABELS = {"en", "english"}

# Latin-only messages shorter than this ("ok", "yes please", "Python
# course") do not say which language the conversation is in.
_MIN_CONFIDENT_LATIN_WORDS = 3

# ---------------------------------------------------------------------
# Arabizi (Arabic written with Latin letters and digits)
# ---------------------------------------------------------------------

# Words that are distinctively Arabizi (not English): one is enough.
_STRONG_ARABIZI = frozenset({
    "3ayez", "3ayz", "3awez", "3awz", "3ayza", "3awza", "3ayzeen", "3awzeen",
    "akhod", "akhd", "a5od", "a5d",
    "at3alem", "at3allem", "et3alem", "adros",
    "tensa7ny", "tensa7ni", "ensa7ny", "ensa7ni",
    "3shan", "3ashan", "3alashan", "3lshan",
    "ezay", "ezzay", "ezai",
    "me7tag", "mo7tag", "m7tag", "e7na", "ya3ni", "keda", "dawra", "dawrat",
})

# Short words that are common in Arabizi but can also appear in English
# text (e.g. the name "Ana"): two weak signals are needed.
_WEAK_ARABIZI = frozenset({"ana", "enta", "enti", "ehna", "fel", "fi", "fe", "eh", "eih"})

# Letters with an embedded Arabizi digit (3=ع, 7=ح, 5=خ, 2=ء), e.g.
# "sa7eb". Weak, because tech tokens like "s3bucket" exist. Digits must
# be followed by a letter, so "web3", "2nd", "b2b" and plain numbers
# never count.
_ARABIZI_DIGIT_WORD = re.compile(r"^[a-z]*(?:[2357][a-z]+)+$")

_ARABIZI_TO_ARABIC = {
    "ana": "انا", "enta": "انت", "enti": "انتي", "ehna": "احنا", "e7na": "احنا",
    "3ayez": "عايز", "3ayz": "عايز", "3awez": "عايز", "3awz": "عايز",
    "3ayza": "عايزه", "3awza": "عايزه",
    "3ayzeen": "عايزين", "3awzeen": "عايزين",
    "akhod": "اخد", "akhd": "اخد", "a5od": "اخد", "a5d": "اخد",
    "at3alem": "اتعلم", "at3allem": "اتعلم", "et3alem": "اتعلم",
    "adros": "ادرس",
    "tensa7ny": "تنصحني", "tensa7ni": "تنصحني",
    "ensa7ny": "انصحني", "ensa7ni": "انصحني",
    "3shan": "عشان", "3ashan": "عشان", "3alashan": "عشان", "3lshan": "عشان",
    "ezay": "ازاي", "ezzay": "ازاي", "ezai": "ازاي",
    "eh": "ايه", "eih": "ايه",
    "fel": "في", "fi": "في", "fe": "في",
    "me7tag": "محتاج", "mo7tag": "محتاج", "m7tag": "محتاج",
    "course": "كورس", "kors": "كورس", "kours": "كورس", "courses": "كورسات",
    "dawra": "دوره", "dawrat": "دورات",
}

# The Arabic->English translation model is trained on Modern Standard
# Arabic and mistranslates common Egyptian words (live run: "كورس" ->
# "choir", "إيه" untranslated). Before translating, these whole words
# are replaced with their MSA equivalents. General dialect vocabulary,
# not dataset-specific; used for the translator's input only.
_EGYPTIAN_TO_MSA = {
    # "دورة" alone is ambiguous (cycle / tournament); "تدريبية" pins it.
    "كورس": "دورة تدريبية", "الكورس": "الدورة التدريبية",
    "كورسات": "دورات تدريبية", "الكورسات": "الدورات التدريبية",
    "إيه": "ماذا", "ايه": "ماذا", "عايز": "أريد", "عاوز": "أريد", "عايزة": "أريد",
    "عايزه": "أريد", "كام": "كم", "بكام": "بكم", "ازاي": "كيف", "إزاي": "كيف",
    "فين": "أين", "امتى": "متى", "إمتى": "متى", "ليه": "لماذا", "دلوقتي": "الآن",
    "ده": "هذا", "دي": "هذه", "اللي": "الذي", "بيغطي": "يغطي", "بيشرح": "يشرح",
    "بيعلم": "يعلم", "هتعلم": "سأتعلم", "هاتعلم": "سأتعلم",
}
# Letters only: the Arabic block also contains punctuation such as "؟".
_ARABIC_WORD_RE = re.compile(r"[^\W\d_]+")

# ---------------------------------------------------------------------
# Explicit language requests ("answer in English", "كلمني عربي")
# ---------------------------------------------------------------------

_EXPLICIT_REQUEST_EN = re.compile(
    r"\b(?:in|speak|use|switch to|reply in|answer in|respond in|write in)"
    r"\s+(english|arabic)\b",
    re.IGNORECASE,
)
_EXPLICIT_REQUEST_AR = (
    ("ar", ("بالعربي", "بالعربيه", "باللغه العربيه", "كلمني عربي")),
    ("en", ("بالانجليزي", "بالانجليزيه", "بالانقليزي", "باللغه الانجليزيه",
            "كلمني انجليزي")),
)
_ARABIC_NORMALIZE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def contains_arabic(text: str) -> bool:
    """Detect whether a string contains Arabic-script characters."""
    return any(
        any(start <= character <= end for start, end in _ARABIC_RANGES)
        for character in text
    )


def looks_like_arabizi(text: str) -> bool:
    """True when Latin-script text is confidently Arabizi."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]

    if any(token in _STRONG_ARABIZI for token in tokens):
        return True

    weak_signals = sum(
        1
        for token in tokens
        if token in _WEAK_ARABIZI
        or (len(token) >= 4 and _ARABIZI_DIGIT_WORD.match(token))
    )
    return weak_signals >= 2


def transliterate_arabizi(text: str) -> str:
    """Replace known Arabizi words with Arabic script, for matching only."""
    return _TOKEN_RE.sub(
        lambda match: _ARABIZI_TO_ARABIC.get(match.group(0).lower(), match.group(0)),
        text,
    )


# Common technology names written in Arabic letters. The translation
# model renders them as ordinary words ("فلاتر" -> "Flatter"), so they
# are written back in Latin script and then kept verbatim. General tech
# vocabulary, not tied to any dataset.
_TECH_TERMS = {
    "فلاتر": "Flutter", "بايثون": "Python", "جافا": "Java", "جافاسكريبت": "JavaScript",
    "جافا سكريبت": "JavaScript", "رياكت": "React", "ريأكت": "React", "نود": "Node.js",
    "دجانجو": "Django", "جانجو": "Django", "فايربيز": "Firebase", "اكسل": "Excel",
    "إكسل": "Excel", "باور بي اي": "Power BI", "باور بي آي": "Power BI", "سيكوال": "SQL",
    "اس كيو ال": "SQL", "دوكر": "Docker", "تيك توك": "TikTok", "فيسبوك": "Facebook",
    "انستجرام": "Instagram", "لينكدان": "LinkedIn", "جوجل": "Google",
}
_TECH_TERM_RE = re.compile(
    # Boundaries are Arabic letters only, so "فلاتر؟" still matches.
    r"(?<![ء-ي])(?:ال)?("
    + "|".join(sorted(map(re.escape, _TECH_TERMS), key=len, reverse=True))
    + r")(?![ء-ي])"
)


def egyptian_to_msa(text: str) -> str:
    """Replace common Egyptian words with MSA ones, for translation input only."""
    text = _TECH_TERM_RE.sub(lambda match: _TECH_TERMS[match.group(1)], text)
    return _ARABIC_WORD_RE.sub(
        lambda match: _EGYPTIAN_TO_MSA.get(match.group(0), match.group(0)),
        text,
    )


def detect_language(text: str) -> str:
    """
    Classify the SCRIPT of a text as "ar", "en", "mixed", "other", or
    "empty". Used for retrieval-query preparation; for the language a
    user is speaking, use detect_conversation_language().
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


def detect_conversation_language(text: str) -> Optional[str]:
    """
    The language the user is writing in: "ar", "en", or None when the
    message gives no confident signal (empty, numbers, or a short
    Latin-only reply such as "ok" or "yes").
    """
    if not isinstance(text, str) or not text.strip():
        return None

    if contains_arabic(text):
        return "ar"

    if looks_like_arabizi(text):
        return "ar"

    if len(_LATIN_WORD_PATTERN.findall(text)) >= _MIN_CONFIDENT_LATIN_WORDS:
        return "en"

    return None


def detect_explicit_language_request(text: str) -> Optional[str]:
    """"ar"/"en" when the user explicitly asks for a reply language."""
    if not isinstance(text, str):
        return None

    match = _EXPLICIT_REQUEST_EN.search(text)
    if match:
        return "en" if match.group(1).lower() == "english" else "ar"

    normalized = text.translate(_ARABIC_NORMALIZE)
    for language, phrases in _EXPLICIT_REQUEST_AR:
        if any(phrase in normalized for phrase in phrases):
            return language

    return None


def _normalize_label(language: Optional[str]) -> Optional[str]:
    if not language:
        return None

    normalized = language.strip().lower()
    if normalized in _ARABIC_LABELS:
        return "ar"
    if normalized in _ENGLISH_LABELS:
        return "en"
    return None


def resolve_conversation_language(
    text: str,
    previous_language: Optional[str] = None,
    preferred_language: Optional[str] = None,
) -> str:
    """
    The language to reply in: an explicit preference first, then a
    confident detection of this message, then the conversation's
    previous language, then English.
    """
    return (
        _normalize_label(preferred_language)
        or detect_conversation_language(text)
        or _normalize_label(previous_language)
        or "en"
    )


# ---------------------------------------------------------------------
# Retrieval-query preparation (translation for the English-only embedder)
# ---------------------------------------------------------------------

_translator_singleton = None


def _get_translator():
    """Reuse one QueryTranslator; imported lazily (it loads PyTorch)."""
    global _translator_singleton

    if _translator_singleton is None:
        from knowledge_base.translator import QueryTranslator

        _translator_singleton = QueryTranslator()

    return _translator_singleton


@dataclass(frozen=True)
class RetrievalQuery:
    """
    original: the text as given (never modified).
    query: the text to embed (English when translation succeeded).
    language: "ar", "en", "mixed", "arabizi", "other" or "empty".
    translation: "not_needed", "translated" or "failed".
    error: why translation failed, when it did.
    """

    original: str
    query: str
    language: str
    translation: str
    error: Optional[str] = None


def prepare_retrieval_query(original_query: str) -> RetrievalQuery:
    """
    Make the query used for embedding English, without touching the
    original. English text is passed through untranslated; Arabic and
    mixed text are translated; Arabizi is first converted to Arabic
    script for its known words, then translated. On failure the
    original text is used for retrieval and the failure is reported.
    """
    language = detect_language(original_query)

    if language == "en" and looks_like_arabizi(original_query):
        language = "arabizi"

    if language not in ("ar", "mixed", "arabizi"):
        return RetrievalQuery(original_query, original_query, language, "not_needed")

    source_text = egyptian_to_msa(
        transliterate_arabizi(original_query)
        if language == "arabizi"
        else original_query
    ).translate(_ARABIC_INDIC_DIGITS)

    # Names, codes and numbers written in Latin script ("Flutter",
    # "PY-101", "3000") are kept verbatim: only the Arabic part is
    # translated, then the Latin parts are added back unchanged.
    kept = _LATIN_SPAN.findall(source_text)
    arabic_part = re.sub(r"\s+", " ", _LATIN_SPAN.sub(" ", source_text)).strip()

    if not contains_arabic(arabic_part):
        return RetrievalQuery(original_query, " ".join(kept) or original_query, language, "not_needed")

    try:
        translated = _get_translator().translate_to_english(arabic_part)
    except Exception as exc:
        logger.warning("Query translation failed: %s", exc)
        return RetrievalQuery(
            original_query, original_query, language, "failed",
            f"{type(exc).__name__}: {exc}",
        )

    translated = (translated or "").strip()

    if not translated or contains_arabic(translated):
        return RetrievalQuery(
            original_query, original_query, language, "failed",
            "The translator returned an empty or untranslated result.",
        )

    for span in kept:
        if span.casefold() not in translated.casefold():
            translated = f"{translated} {span}"

    return RetrievalQuery(original_query, translated, language, "translated")
