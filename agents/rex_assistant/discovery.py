"""
Learner discovery for REX Assistant's personalized-recommendation mode.

The six required fields come from the official brief (section 10, "REX
Assistant" discovery questions). Each has a fixed English and Arabic
question; the reply to a pending question is mapped to that field,
validated, and normalized where the answer has a closed set of values.

Pure functions only: no LLM and no I/O, so every rule here is testable.
Free-form answers that need understanding (several facts in one message)
are extracted separately by the graph's LLM step, then validated here.
"""

from __future__ import annotations

import re
from typing import Optional

from agents.text_utils import normalize_for_matching

REQUIRED_FIELDS = (
    "background",
    "career_goal",
    "skill_level",
    "training_preference",
    "budget",
    "available_time",
)
OPTIONAL_FIELDS = ("interests",)
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# A field may be answered at most this many times with an unclear reply
# before it is skipped automatically.
MAX_ATTEMPTS_PER_FIELD = 2

QUESTIONS = {
    "background": {
        "en": "What is your educational or professional background?",
        "ar": "ما هو مؤهلك الدراسي أو خلفيتك المهنية؟",
    },
    "career_goal": {
        "en": "What career path are you interested in?",
        "ar": "ما هو المسار المهني الذي ترغب في العمل فيه؟",
    },
    "skill_level": {
        "en": "What is your current skill level?",
        "ar": "ما مستوى مهاراتك الحالي؟",
    },
    "training_preference": {
        "en": "Do you prefer online or offline training?",
        "ar": "هل تفضل التدريب أونلاين أم حضوريًا؟",
    },
    "budget": {
        "en": "What is your available budget?",
        "ar": "ما الميزانية المتاحة لديك للتدريب؟",
    },
    "available_time": {
        "en": "How much time can you dedicate to learning?",
        "ar": "كم من الوقت يمكنك تخصيصه للتعلم؟",
    },
}

DISCOVERY_INTRO = {
    "en": (
        "I'll ask a few short questions so I can recommend the right "
        "courses. You can say \"skip\" for any question, or \"recommend "
        "now\" to get recommendations with what you've shared so far."
    ),
    "ar": (
        "سأطرح عليك بعض الأسئلة القصيرة لأرشح لك الدورات المناسبة. يمكنك "
        "قول «تخطي» لأي سؤال، أو «رشحلي دلوقتي» للحصول على ترشيحات بما "
        "شاركته حتى الآن."
    ),
}

CLARIFY_PREFIX = {
    "en": "Sorry, I didn't quite get that. ",
    "ar": "عذرًا، لم أفهم إجابتك تمامًا. ",
}

FIELD_LABELS = {
    "background": {"en": "background", "ar": "الخلفية"},
    "career_goal": {"en": "career goal", "ar": "الهدف المهني"},
    "skill_level": {"en": "skill level", "ar": "مستوى المهارة"},
    "training_preference": {"en": "training preference", "ar": "طريقة التدريب"},
    "budget": {"en": "budget", "ar": "الميزانية"},
    "available_time": {"en": "available time", "ar": "الوقت المتاح"},
}

# Normalized values for the closed-set fields.
PREFERENCE_VALUES = ("online", "offline", "hybrid", "no_preference")
SKILL_VALUES = ("beginner", "intermediate", "advanced")

_SKIP = (
    "skip", "pass", "next question", "prefer not to say", "rather not",
    "don't want to say", "dont want to say", "i don't know", "i dont know",
    "no idea", "not sure",
    "تخطي", "تخطى", "عدي", "بلاش", "مش عايز اقول", "مش عايزه اقول",
    "لا اريد الاجابه", "مش عارف", "مش عارفه", "لا اعرف", "معرفش",
)
_RECOMMEND_NOW = (
    "recommend now", "just recommend", "recommend anyway", "recommend me now",
    "skip the questions", "enough questions", "show me courses now",
    "رشحلي دلوقتي", "رشحلي على طول", "رشح دلوقتي", "كفايه اسئله",
    "رشحلي مباشره", "رشحلي الان",
)

_PREFERENCE_PATTERNS = (
    ("hybrid", ("hybrid", "blended", "both", "mix", "مدمج", "الاتنين", "الاثنين", "هجين")),
    ("no_preference", (
        "no preference", "either", "any", "doesn't matter", "does not matter",
        "whatever", "مش فارقه", "مش فارق", "اي حاجه", "اي واحد", "لا يهم",
    )),
    ("online", (
        "online", "remote", "virtual", "from home", "اونلاين", "اون لاين",
        "عن بعد", "اونلاين", "من البيت",
    )),
    ("offline", (
        "offline", "onsite", "on-site", "on site", "in person", "in-person",
        "classroom", "حضوري", "حضوريا", "حضورى", "اوفلاين", "في المقر",
        "في القاعه",
    )),
)
_SKILL_PATTERNS = (
    ("advanced", ("advanced", "expert", "senior", "professional", "متقدم", "خبير", "محترف")),
    ("intermediate", (
        "intermediate", "some experience", "medium", "average", "متوسط",
        "عندي خبره بسيطه", "نص نص",
    )),
    ("beginner", (
        "beginner", "basic", "novice", "no experience", "zero", "from scratch",
        "just starting", "starting out", "مبتدئ", "مبتدا", "من الصفر",
        "مفيش خبره", "بدون خبره", "لسه ببدا",
    )),
)
_BUDGET_WORDS = (
    "free", "no budget", "limited", "flexible", "cheap", "affordable",
    "egp", "usd", "sar", "pound", "dollar", "riyal", "per month",
    "مجاني", "ببلاش", "محدوده", "مفتوحه", "مرنه", "جنيه", "دولار", "ريال",
    "على قدي", "على القد", "مش مهم",
    "no limit", "any budget", "not specific", "not sure", "مش محدد", "مش عارف",
)
_TIME_WORDS = (
    "hour", "day", "week", "month", "weekend", "evening", "morning",
    "full time", "full-time", "part time", "part-time", "flexible",
    # Arabic units match inside words: prefixes attach ("اليوم") and the
    # dual is a different form ("ساعتين").
    "ساعه", "ساعات", "ساعت", "يوم", "ايام", "اسبوع", "شهر", "ويك اند",
    "بالليل", "الصبح", "وقت كامل", "جزئي", "مرن",
    # No fixed amount ("not specific time", "any time") is still an answer.
    "not specific", "no specific", "any time", "anytime", "whenever", "depends",
    "اي وقت", "مش محدد", "حسب",
)
_BARE_REPLIES = frozenset({
    "yes", "no", "ok", "okay", "sure", "yeah", "nope", "thanks", "thank you",
    "نعم", "لا", "تمام", "ايوه", "ماشي", "شكرا", "اوك",
})
_PLACEHOLDER_PREFIXES = (
    "no information", "no info", "not mentioned", "not provided",
    "not specified", "not stated", "unknown", "n/a", "none", "null",
    "لا يوجد", "لا توجد", "غير معروف", "غير محدد", "غير مذكور", "لم يذكر",
    "لم يتم",
)
_DIGITS = re.compile(r"\d")
_WORDS = re.compile(r"[^\W_]+")


def _norm(text: str) -> str:
    return normalize_for_matching(text).strip()


def _has_phrase(norm_text: str, phrases: tuple[str, ...]) -> bool:
    """Whole-word / whole-phrase match (so "عدي" is not found in "بعدين")."""
    return any(
        re.search(rf"(?<![^\W_]){re.escape(_norm(phrase))}(?![^\W_])", norm_text)
        for phrase in phrases
    )


def _has_unit(norm_text: str, phrases: tuple[str, ...]) -> bool:
    """English units as whole words, Arabic units inside words."""
    latin = tuple(p for p in phrases if p.isascii())
    arabic = tuple(_norm(p) for p in phrases if not p.isascii())
    return _has_phrase(norm_text, latin) or any(p in norm_text for p in arabic)


def wants_to_skip(text: str) -> bool:
    return _has_phrase(_norm(text), _SKIP)


def wants_recommendation_now(text: str) -> bool:
    return _has_phrase(_norm(text), _RECOMMEND_NOW)


def is_placeholder(value: str) -> bool:
    normalized = _norm(value).strip(" .،,")
    return any(
        normalized.startswith(_norm(prefix)) for prefix in _PLACEHOLDER_PREFIXES
    )


def _match_closed_set(text: str, patterns) -> Optional[str]:
    normalized = _norm(text)
    for value, phrases in patterns:
        if _has_phrase(normalized, phrases):
            return value
    return None


def normalize_answer(field: str, text: str) -> Optional[str]:
    """
    The value to store for `field` from a reply, or None when the reply
    does not answer it clearly. Closed-set fields are normalized
    (training_preference, skill_level); the others keep the user's words.
    """
    if not isinstance(text, str):
        return None

    value = text.strip()
    normalized = _norm(value)

    if not normalized or is_placeholder(value):
        return None

    if field == "training_preference":
        return _match_closed_set(value, _PREFERENCE_PATTERNS)

    if field == "skill_level":
        return _match_closed_set(value, _SKILL_PATTERNS)

    if normalized in _BARE_REPLIES or len(_WORDS.findall(normalized)) == 0:
        return None

    if field == "budget":
        if _DIGITS.search(normalized) or _has_unit(normalized, _BUDGET_WORDS):
            return value
        return None

    if field == "available_time":
        if _DIGITS.search(normalized) or _has_unit(normalized, _TIME_WORDS):
            return value
        return None

    # background, career_goal, interests: free text of at least two
    # letters. A question back ("what do you mean?") is not an answer.
    if value.endswith(("?", "؟")):
        return None
    return value if len(normalized) >= 2 else None


def missing_fields(profile: dict, declined: list[str] | tuple[str, ...]) -> list[str]:
    """Required fields with neither a value nor an explicit skip."""
    return [
        field
        for field in REQUIRED_FIELDS
        if not profile.get(field) and field not in declined
    ]


def question_for(field: str, language: str) -> str:
    return QUESTIONS[field]["ar" if language == "ar" else "en"]


def field_label(field: str, language: str) -> str:
    return FIELD_LABELS.get(field, {}).get("ar" if language == "ar" else "en", field)
