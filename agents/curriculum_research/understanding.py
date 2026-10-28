"""
Rule-based understanding of an administrator's message (English and
Arabic): which workflow it asks for, and the values it states.

Built on request_parser.py (hours, audience, topic, creation intent and
draft edits are extracted there and tested) -- this module only adds the
intents and values the rebuilt agent needs: catalogue suggestions,
framework alignment, show/list, approval attempts, level, course ids,
output language and answers to a pending question. Pure functions; the
graph's LLM fallback is used only when these find no topic.
"""

from __future__ import annotations

import re
from typing import Optional

from agents.curriculum_research.request_parser import (
    has_creation_intent,
    is_accreditation_request,
    parse_request,
)
from agents.text_utils import normalize_for_matching
from knowledge_base.language import detect_explicit_language_request

COURSE_ID = re.compile(r"\b([A-Za-z]{2,4}-\d{3})\b")
CURRICULUM_ID = re.compile(r"\b(CUR-[0-9a-f]{12})\b", re.IGNORECASE)

_LEVELS = (
    # Only explicit level words: "Python Fundamentals" names a topic, not a level.
    ("Beginner", r"\b(?:beginners?|introductory|entry[\s-]level)\b|مبتدئ|مستوي\s+اساسي|تمهيدي"),
    ("Intermediate", r"\bintermediate\b|متوسط"),
    ("Advanced", r"\badvanced\b|متقدم"),
)
_LEVEL_RES = [(level, re.compile(pattern)) for level, pattern in _LEVELS]

_NO_CAPSTONE = re.compile(r"\b(?:no|without)\s+(?:a\s+|any\s+)?capstone\b|بدون\s+(?:ال)?مشروع\s+(?:ال)?ختامي")

# Catalogue planning (Workflow B).
_SUGGEST = [re.compile(p) for p in (
    r"\b(?:suggest|recommend|propose|identify|find|list|what)\b.{0,50}\b(?:new|missing|additional|more|future)\b"
    r".{0,30}\b(?:courses?|curricula|programs?|programmes?|learning\s+paths?|tracks?|offerings?)\b",
    r"\b(?:gaps?|missing)\b.{0,40}\b(?:catalogue|catalog|courses?|learning\s+paths?|offerings?|portfolio|tracks?)\b",
    r"\b(?:catalogue|catalog)\b.{0,40}\b(?:gaps?|review|analysis|analy[sz]e|missing)\b",
    r"\b(?:review|analy[sz]e|audit)\b.{0,30}\b(?:catalogue|catalog|course\s+offerings?)\b",
    r"\bwhat\s+(?:new\s+)?courses?\s+should\s+(?:we|the\s+academy)\b",
    r"(?:اقترح|رشح|حدد|ما\s+هي|ماهي|ايه).{0,30}(?:دورات|مناهج|كورسات|مسارات|برامج)\s+(?:جديده|ناقصه|اضافيه|مستقبليه)",
    r"(?:فجوات|فجوه|نواقص|الناقصه).{0,40}(?:الكتالوج|الدورات|المسارات|البرامج|الاكاديميه|الكورسات)",
    r"(?:الكتالوج|كتالوج).{0,30}(?:فجوات|فجوه|مراجعه|تحليل)",
    r"(?:راجع|حلل).{0,20}(?:الكتالوج|كتالوج|الدورات\s+الحاليه)",
)]
_SUGGESTION_CHOICE = re.compile(
    r"\b(?:suggestion|option|idea|proposal)\s*(?:no\.?|number|#)?\s*(\d)\b"
    r"|\b(first|second|third|fourth|fifth)\s+(?:suggestion|option|idea|proposal|one)\b"
    r"|(?:الاقتراح|المقترح|الخيار)\s*(?:رقم\s*)?(\d)"
    r"|(?:الاقتراح|المقترح|الخيار)\s+(الاول|الثاني|الثالث|الرابع|الخامس)"
)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "الاول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5}
_CHOICE_VERB = re.compile(r"\b(?:generate|create|build|develop|design|use|go\s+with|take|pick|choose|select|"
                          r"make|draft)\b|انشئ|اعد|صمم|طور|اختار|اختر|استخدم|نفذ|ابن|اعمل")

# Framework alignment.
_FRAMEWORK_WORD = re.compile(
    r"\b(?:framework|standards?|competenc(?:y|ies)\s+model|nqf|ncaaa|naqaae|masar|esco|o\*?net|nice|swebok|"
    r"cs2023|accreditation)\b|اطار|معايير|معيار|الاطار\s+الوطني|اعتماد\s+اكاديمي"
)
_ALIGN_VERB = re.compile(r"\b(?:align|compare|map|check|assess|evaluate|match|benchmark)\w*\b|قارن|مقارنه|طابق|"
                         r"مطابقه|مواءمه|موائمه|قيم")

# Approval belongs to administrators in the Curricula tab only.
_APPROVE = re.compile(
    r"^\s*(?:please\s+|now\s+|ok\s+|okay\s+)?(?:approve|publish|finali[sz]e|sign\s+off|accept)\b"
    r"|\b(?:approve|publish)\s+(?:the|this|my|it|curriculum|draft|version)\b"
    r"|^\s*(?:من\s+فضلك\s+)?(?:اعتمد|وافق|انشر)(?:ه|ها)?(?!\w)"
    r"|(?:اعتماد|الموافقه\s+علي|نشر)\s+(?:ال)?(?:منهج|مسوده|نسخه)"
)

_SHOW = re.compile(
    r"\b(?:show|view|display|open|see|get|print|read)\b.{0,25}\b(?:curriculum|curricula|draft|drafts|version|"
    r"versions|it|history)\b"
    r"|\b(?:list|show)\b.{0,15}\b(?:curricula|drafts|my\s+curricula|saved)\b"
    r"|\b(?:download|export)\b|\bpdf\b|\bstatus\s+of\b"
    r"|(?:اعرض|اظهر|ورني|وريني|عرض|افتح).{0,25}(?:المنهج|المسوده|المناهج|النسخه|المسودات)"
    r"|تحميل|نزل|\bpdf\b"
)
_LIST = re.compile(r"\b(?:list|all|my)\b.{0,15}\b(?:curricula|drafts)\b|(?:كل|قائمه)\s+(?:ال)?(?:مناهج|مسودات)")

_SEND = re.compile(r"^\s*(?:please\s+)?(?:e-?mail|send|share|forward|post)\b|^\s*(?:من\s+فضلك\s+)?(?:ارسل|شارك)")

_DEFAULTS = re.compile(r"\b(?:use|go\s+with|keep)\s+(?:the\s+)?defaults?\b|\bdefaults?\s+(?:is|are)?\s*(?:fine|ok)\b"
                       r"|^\s*defaults?\s*$|(?:استخدم|اعتمد)\s+(?:ال)?(?:افتراضي|افتراضيات|القيم\s+الافتراضيه)")


def find_level(text: str) -> Optional[str]:
    norm = normalize_for_matching(text)
    found = [level for level, pattern in _LEVEL_RES if pattern.search(norm)]
    return found[0] if len(found) == 1 else None


def find_course_id(text: str) -> Optional[str]:
    match = COURSE_ID.search(text or "")
    if not match or CURRICULUM_ID.search(match.group(0)):
        return None
    return match.group(1).upper()


def find_curriculum_id(text: str) -> Optional[str]:
    match = CURRICULUM_ID.search(text or "")
    return ("CUR-" + match.group(1)[4:].lower()) if match else None


def output_language(text: str, reply_language: str) -> str:
    """The curriculum's language: an explicit "in Arabic"/"بالإنجليزية", else the reply language."""
    explicit = detect_explicit_language_request(text or "")
    if explicit:
        return explicit
    norm = normalize_for_matching(text or "")
    match = re.search(r"\b(?:translate|convert|rewrite|switch|change)\b.{0,40}\b(?:to|into|in)\s+(arabic|english)\b", norm)
    if match:
        return "ar" if match.group(1) == "arabic" else "en"
    if re.search(r"(?:باللغه|بال|الي|لل)\s*(?:ال)?(?:لغه\s+)?(?:ال)?عربي", norm):
        return "ar"
    if re.search(r"(?:باللغه|بال|الي|لل)\s*(?:ال)?(?:لغه\s+)?(?:ال)?(?:انجليزي|انقليزي|انكليزي)", norm):
        return "en"
    return reply_language


def wants_no_capstone(text: str) -> bool:
    return bool(_NO_CAPSTONE.search(normalize_for_matching(text)))


def is_suggest_request(text: str) -> bool:
    norm = normalize_for_matching(text)
    return any(pattern.search(norm) for pattern in _SUGGEST)


def suggestion_choice(text: str) -> Optional[int]:
    """1-based number of a stored suggestion the admin picks ("generate suggestion 2")."""
    norm = normalize_for_matching(text)
    match = _SUGGESTION_CHOICE.search(norm)
    if not match:
        return None
    if not _CHOICE_VERB.search(norm) and len(norm.split()) > 4:
        return None
    for group in match.groups():
        if group:
            return int(group) if group.isdigit() else _ORDINALS.get(group)
    return None


def is_alignment_request(text: str) -> bool:
    norm = normalize_for_matching(text)
    if is_suggest_request(text) and not _FRAMEWORK_WORD.search(norm):
        return False  # "gap analysis of our catalogue" is Workflow B
    return is_accreditation_request(text) or bool(_FRAMEWORK_WORD.search(norm) and _ALIGN_VERB.search(norm))


def is_approve_attempt(text: str) -> bool:
    return bool(_APPROVE.search(normalize_for_matching(text)))


def is_show_request(text: str) -> bool:
    return bool(_SHOW.search(normalize_for_matching(text)))


def is_list_request(text: str) -> bool:
    return bool(_LIST.search(normalize_for_matching(text)))


def is_send_request(text: str) -> bool:
    return bool(_SEND.search(normalize_for_matching(text)))


def accepts_defaults(text: str) -> bool:
    return bool(_DEFAULTS.search(normalize_for_matching(text)))


def is_creation_request(text: str) -> bool:
    return has_creation_intent(text)


def request_slots(text: str) -> dict:
    """Values a generation request states (None when not stated)."""
    no_capstone = wants_no_capstone(text)
    if no_capstone:  # "..., no capstone" is an instruction, not part of the topic
        text = re.sub(r"(?i)\b(?:no|without)\s+(?:a\s+|any\s+)?capstone\b|بدون\s+(?:ال)?مشروع\s+(?:ال)?ختامي",
                      " ", text)
    parsed = parse_request(text)
    level = find_level(text)
    audience = parsed.target_audience
    if audience and find_level(audience) and normalize_for_matching(audience).strip() in (
            "beginners", "beginner", "المبتدئين", "مبتدئين"):
        audience = None  # "for beginners" states the level, not a distinct audience
    return {
        "topic": parsed.topic,
        "course_ref": find_course_id(text),
        "total_hours": parsed.training_hours,
        "level": level,
        "target_audience": audience,
        "capstone": False if no_capstone else None,
        "hours_rate_ignored": parsed.ignored_hours_rate,
    }


def answer_slots(text: str) -> dict:
    """
    Values in a reply to the agent's question ("40 hours, intermediate").
    A bare number counts as hours; any other short text without a stated
    value is taken as the topic.
    """
    slots = request_slots(text)
    stripped = (text or "").strip()
    if slots["total_hours"] is None:
        bare = re.fullmatch(r"\s*(\d{1,4})\s*", normalize_for_matching(stripped))
        if bare:
            slots["total_hours"] = int(bare.group(1))
    if not slots["topic"] and slots["total_hours"] is None and not slots["level"] and not accepts_defaults(text):
        if 0 < len(stripped) <= 120 and not re.search(r"[?؟]", stripped):
            slots["topic"] = stripped.strip(" .!،,")
    return slots
