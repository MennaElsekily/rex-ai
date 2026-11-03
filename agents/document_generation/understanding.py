"""
Rule-based reading of a chat request for Agent 4 (English and Arabic):
the document type, the curriculum id, the language, and which approved
curriculum a name such as "the SQL curriculum" refers to. No language
model is involved. When the Streamlit form already chose the document
type and curriculum, none of this is used (see graph.py).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.text_utils import normalize_for_matching

CURRICULUM_ID = re.compile(r"\bCUR-[0-9a-f]{12}\b", re.IGNORECASE)
_BAD_REFERENCE = re.compile(r"\b(?:curriculum|curricula|id)\s*(?:no\.?|number|#)?\s*#?(\w{1,12})\b")

# Phrases are matched on normalize_for_matching() text (Arabic letters
# normalised), most specific first.
DOCUMENT_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("outcomes_assessment_plan", ("learning outcomes and assessment", "assessment plan", "خطه التقييم", "خطه تقييم")),
    ("capstone_brief", ("capstone project brief", "capstone brief", "ملخص المشروع الختامي")),
    ("course_description", ("course description", "وصف الدوره", "وصف الكورس")),
    ("course_proposal", ("course proposal", "training proposal", "مقترح دوره", "مقترح تدريبي")),
    ("participant_guide", ("participant guide", "participant handbook", "learner guide", "student guide",
                           "trainee guide", "دليل المتدرب", "دليل المشارك", "دليل الطالب")),
    ("instructor_guide", ("instructor guide", "facilitator guide", "trainer guide", "دليل المدرب", "دليل الميسر")),
    ("presentation_outline", ("presentation outline", "slide deck", "slides", "عرض تقديمي", "شرائح")),
    ("question_bank", ("question bank", "بنك اسئله")),
    ("assessment_rubric", ("assessment rubric", "grading rubric", "rubrics", "rubric", "معايير التقييم")),
    ("curriculum_pdf", ("curriculum pdf", "pdf of the curriculum", "curriculum as a pdf")),
)
_VERBS = re.compile(
    r"\b(?:generate|create|make|produce|write|prepare|draft|build|compose|export|need|want|give|get|design)\b"
    r"|(?<!\w)(?:انشئ|انشاء|اكتب|جهز|حضر|اعد|اعمل|اريد|انتج|ولد)(?!\w)"
)
_LANGUAGE_AR = re.compile(r"\b(?:in|into)\s+arabic\b|arabic\s+version|بالعربي|باللغه العربي|عربي")
_LANGUAGE_EN = re.compile(r"\b(?:in|into)\s+english\b|english\s+version|بالانجليزي|بالانكليزي|باللغه الانجليزي")

_STOP = frozenset("""a an the for of on in to and or with using use from me us our my please that this it its is are be
approved selected chosen current existing new curriculum curricula course courses document documents pdf generate
create make produce write prepare draft build compose export need want give get design description proposal guide
participant instructor facilitator capstone brief assessment plan outcomes learning training""".split())


def find_document_type(text: str) -> Optional[str]:
    normalized = normalize_for_matching(text)
    for doc_type, phrases in DOCUMENT_PHRASES:
        if any(phrase in normalized for phrase in phrases):
            return doc_type
    return None


# "create a (40-hour) curriculum ..." -- the curriculum itself is what is asked for (Agent 3), even if the
# sentence also names a document ("... with an assessment plan"). A document phrase inside the matched
# span ("generate a course description for curriculum X") means the curriculum is only the source.
_CREATES_CURRICULUM = re.compile(
    r"\b(?:generate|create|make|build|design|develop|draft|write|prepare|produce|compose)\s+(?:me\s+|us\s+)?"
    r"(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,3}?(?:curriculum|curricula|syllabus|course outline)\b"
    r"|(?<!\w)(?:انشئ|صمم|اعد|اكتب|جهز|اعمل|طور)\s+(?:\S+\s+){0,2}?(?:منهج|مناهج|منهاج)"
)


def _creates_a_curriculum(normalized: str) -> bool:
    return any(find_document_type(match.group(0)) is None for match in _CREATES_CURRICULUM.finditer(normalized))


def is_document_request(text: str) -> bool:
    """A request to produce one of the listed documents (supported or not), but not a curriculum PDF or a new curriculum."""
    normalized = normalize_for_matching(text)
    doc_type = find_document_type(text)
    return (doc_type not in (None, "curriculum_pdf") and bool(_VERBS.search(normalized))
            and not _creates_a_curriculum(normalized))


def find_curriculum_id(text: str) -> Optional[str]:
    match = CURRICULUM_ID.search(text)
    return "CUR-" + match.group(0)[4:].lower() if match else None


def find_bad_reference(text: str) -> Optional[str]:
    """Something like "curriculum 123" that is not a valid curriculum id."""
    if CURRICULUM_ID.search(text):
        return None
    for match in _BAD_REFERENCE.finditer(normalize_for_matching(text)):
        value = match.group(1)
        if any(ch.isdigit() for ch in value):
            return value
    return None


def find_language(text: str) -> Optional[str]:
    normalized = normalize_for_matching(text)
    if _LANGUAGE_AR.search(normalized):
        return "ar"
    if _LANGUAGE_EN.search(normalized):
        return "en"
    return None


def choose_from_options(text: str, option_ids: list[str]) -> Optional[str]:
    """A reply such as "2" or "number 2" picks the n-th curriculum offered."""
    match = re.fullmatch(r"\s*(?:number\s*|no\.?\s*|#)?(\d{1,2})\s*[.)]?\s*", normalize_for_matching(text))
    if match and 1 <= int(match.group(1)) <= len(option_ids):
        return option_ids[int(match.group(1)) - 1]
    return None


def match_curricula(text: str, curricula: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The approved curricula whose title shares the most words with the request (ties kept)."""
    words = {w for w in re.findall(r"[^\W_]+", normalize_for_matching(text)) if len(w) > 1 and w not in _STOP}
    scored = []
    for row in curricula:
        title_words = set(re.findall(r"[^\W_]+", normalize_for_matching(str(row.get("title") or ""))))
        score = len(words & title_words)
        if score:
            scored.append((score, row))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [row for score, row in scored if score == best]
