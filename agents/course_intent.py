"""
What does a course-related message want? One shared classifier, used by
the orchestrator (should REX get this message?) and by REX Assistant
(which mode: personalized discovery or a direct course question?).

- RECOMMENDATION: the user wants courses picked FOR THEM ("recommend a
  course for me", "I want to become a data analyst, what should I
  study?", "تنصحني بإيه؟", "رشحلي كورسات مناسبة ليا").
- COURSE_QUESTION: the user wants facts about courses ("what courses are
  available?", "how long does the Python course take?", "مدة الكورس
  كام؟"). Weak cues such as "cost" or "online" only count next to a
  course word, unless the caller says a course is already being
  discussed (course_context=True), e.g. a follow-up "Is it online?".
- GENERAL: anything else (greetings, small talk, unrelated requests).

A course word alone is never enough, so "Make a good decision about the
course" is GENERAL. "hours" is deliberately not a cue: "make the course
40 hours" is a curriculum edit, not a question for REX. A learning goal
("I want to learn AI", "عايز أتعلم الذكاء الاصطناعي") is a RECOMMENDATION:
the user needs course guidance.
"""

from __future__ import annotations

import re

from agents.text_utils import normalize_for_matching
from knowledge_base.language import transliterate_arabizi

RECOMMENDATION = "recommendation"
COURSE_QUESTION = "course_question"
GENERAL = "general"

_COURSE_WORD = r"(?:course|program|programme|catalog|catalogue|class|bootcamp|diploma)s?"

_RECOMMENDATION_EN = re.compile("|".join((
    r"\brecommend",
    r"\bsuggest",
    r"\badvi[cs]e\b",
    r"\bhelp me (?:choose|pick|decide)\b",
    rf"\b(?:which|what) {_COURSE_WORD} should i\b",
    r"\bwhat should i (?:study|learn|take)\b",
    rf"\b(?:best|right|suitable|good) {_COURSE_WORD} for me\b",
    r"\b(?:best|right|suitable|good|ideal) for me\b",
    rf"\b{_COURSE_WORD} (?:that )?(?:suits?|fits?) me\b",
    r"\bbased on my\b",
    r"\bi want to (?:become|be a|work as|get into|switch to|start)\b",
    r"\bmy career\b",
    r"\bwhere (?:should|do) i start\b",
    rf"\b(?:want|like|need) to (?:take|join|start) an? {_COURSE_WORD}\b",
    rf"\blooking for an? {_COURSE_WORD}\b",
    # A learning goal asks for course guidance ("I want to learn AI").
    r"\b(?:want|would like|need|plan|hope) to (?:learn|study)\b",
    r"\binterested in learning\b",
)))

_RECOMMENDATION_AR = (
    "رشح", "اقترح", "نصح", "نصيحه", "انسب", "مناسب لي", "مناسبه لي",
    # The noun "a recommendation" ("عايز ترشيح كورس"): "رشح" is not inside
    # "ترشيح" (the extra ي), so it needs its own cue.
    "ترشيح",
    "اتعلم ايه", "ادرس ايه", "ابدا منين", "ابدا في", "عايز ابدا", "عاوز ابدا",
    "عايز اشتغل", "عايز ابقي", "عايز اكون", "اخد كورس", "اخد دوره", "اخذ دوره",
    "اخذ كورس", "حسب خلفيتي", "مساري المهني",
    # "Help me choose a course" ("ساعدني أختار دورة").
    "ساعدني اختار", "ساعدني في اختيار", "ساعدني اني اختار", "اختارلي",
    "اختار كورس", "اختار دوره", "اختيار كورس", "اختيار دوره", "اختيار الدوره",
    # Learning goals ("عايز أتعلم الذكاء الاصطناعي").
    "عايز اتعلم", "عاوز اتعلم", "حابب اتعلم", "نفسي اتعلم", "اريد ان اتعلم",
    "اريد تعلم", "اود تعلم", "ارغب في تعلم",
)

# "I need course python", "I want a Python course", "looking for a course
# in data analysis": the learner wants a course found for them. Checked
# only after the course-question rules, so "I need the Python course
# schedule" stays a question; "want to know which courses..." never matches.
_NEEDS_A_COURSE_EN = re.compile(
    r"\b(?:need|want|looking for|interested in|searching for)\s+(?!to\b)"
    r"(?:(?!(?:to|which|what|about|know|if|whether)\b)[\w+#.-]+\s+){0,3}?"
    rf"{_COURSE_WORD}\b"
)
_NEEDS_A_COURSE_AR = (
    "عايز كورس", "عاوز كورس", "عايزه كورس", "محتاج كورس", "محتاجه كورس", "اريد كورس",
    "عايز دوره", "عاوز دوره", "عايزه دوره", "محتاج دوره", "محتاجه دوره", "اريد دوره",
    "ابحث عن دوره", "ابحث عن كورس", "بدور علي كورس", "بدور علي دوره",
)

_COURSE_NOUN_EN = re.compile(rf"\b{_COURSE_WORD}\b")
_COURSE_NOUN_AR = ("كورس", "دوره", "دورات", "برنامج", "برامج", "كتالوج")

# Enough on their own: unmistakably about course details.
_STRONG_INFO_EN = re.compile("|".join((
    r"\bprerequisites?\b",
    r"\bentry requirements?\b",
    r"\btuition\b",
    r"\binstructors?\b",
    r"\bhow long does (?:it|that|this) (?:take|last)\b",
    r"\bwhat (?:skills )?will i learn\b",
    r"\bwhat skills\b",
    r"\btell me (?:more )?about (?!yoursel|you\b)",
    r"\bmore (?:info|information|details) (?:about|on)\b",
)))
_STRONG_INFO_AR = ("متطلبات", "هتعلم ايه", "هاتعلم ايه", "مدته قد ايه", "مدتها قد ايه")

# Academy information REX answers from the knowledge base (fees and
# offers, admission, certificates, instructors, schedules -- the brief's
# required REX topics), even when no course is named.
_ACADEMY_INFO_EN = re.compile("|".join((
    r"\brefunds?\b", r"\bcertificates?\b", r"\baccredit\w*", r"\bdiscounts?\b",
    r"\b(?:offers?|promotions?|promo codes?)\b", r"\binstal+ments?\b",
    r"\bpayment options?\b", r"\bhow (?:can|do) i pay\b",
    r"\bregist(?:er|ration)\b", r"\benrol+(?:ment)?\b", r"\badmission\b",
    r"\bdocuments? (?:do i need|required|needed)\b", r"\bplacement test\b",
    r"\b(?:instructors?|teachers?|trainers?|lecturers?)\b", r"\bwho teach(?:es)?\b",
    r"\b(?:schedules?|timings?|intakes?)\b", r"\bwhen does\b.*\bstart\b", r"\bstart date\b",
    r"\bcampus\b", r"\bonline classes\b", r"\brecord(?:ed|ings?)\b", r"\blaptop\b",
    r"\bjob placement\b", r"\bmiss(?:ed)? a session\b", r"\bcorporate\b",
)))
_ACADEMY_INFO_AR = (
    "استرداد", "استرجاع", "شهاده", "شهادات", "معتمد", "اعتماد", "خصم", "خصومات", "عروض",
    "تقسيط", "اقساط", "الدفع", "تسجيل", "اسجل", "التحاق", "القبول", "المستندات",
    "الاوراق المطلوبه", "اختبار تحديد المستوي", "المدرب", "المدربين", "المحاضر", "المدرس",
    "مين بيدرس", "مواعيد", "يبدا امتي", "امتي يبدا", "يبدا متي", "متي يبدا", "الفرع",
    "المقر", "لابتوب", "تسجيلات", "المحاضرات المسجله",
)
_AVAILABILITY_EN = re.compile(r"\b(?:do you (?:have|offer)|is there|are there|can i (?:take|study|learn))\b")

_QUESTION_START_EN = re.compile(
    r"^\s*(?:what|how|when|where|who|whom|which|why|is|are|do|does|did|can|could|will|would|should|may)\b"
)
_QUESTION_WORDS_AR = ("هل", "ما", "ماذا", "كيف", "متي", "امتي", "اين", "فين", "مين", "كم",
                      "كام", "ايه", "ازاي", "ليه", "لماذا", "بكام")


def _looks_like_question(text: str, arabic: str) -> bool:
    """A question, as opposed to a statement such as an answer to REX."""
    if text.rstrip().endswith(("?", "؟")) or _QUESTION_START_EN.match(text.casefold()):
        return True
    words = set(arabic.split())
    return any(word in words for word in _QUESTION_WORDS_AR)


def looks_like_question(text: str) -> bool:
    """'How experienced is Karim Nabil' / 'مين المدرب' -- phrased as a question."""
    if not isinstance(text, str) or not text.strip():
        return False
    return _looks_like_question(text, normalize_for_matching(transliterate_arabizi(text)))


# Questions about REX itself or social chat, answered with the greeting
# rather than a knowledge-base search.
_SMALL_TALK_EN = re.compile(
    r"^\s*(?:how are you|how's it going|what'?s up|who are you|what are you|"
    r"what can you do|how can you help|are you (?:a )?(?:bot|robot|human|ai))\b"
)
_SMALL_TALK_AR = ("ازيك", "عامل ايه", "اخبارك", "انت مين", "مين انت", "من انت", "كيف حالك",
                  "تقدر تعمل ايه", "بتعمل ايه")


_GREETING_EN = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|greetings|good (?:morning|afternoon|evening)|thanks?|thank you|bye|goodbye)\b"
)
_GREETING_AR = ("اهلا", "مرحبا", "السلام عليكم", "صباح الخير", "مساء الخير", "شكرا", "هاي", "ازيك")


def is_greeting(text: str) -> bool:
    """'Hello', 'Thanks!', 'أهلاً', 'مرحبا' -- answered with REX's greeting."""
    if not isinstance(text, str):
        return False
    arabic = normalize_for_matching(text).strip()
    return bool(_GREETING_EN.match(text.casefold())) or arabic.startswith(_GREETING_AR)


def is_small_talk(text: str) -> bool:
    if not isinstance(text, str):
        return False
    arabic = normalize_for_matching(text)
    return bool(_SMALL_TALK_EN.match(text.casefold())) or any(cue in arabic for cue in _SMALL_TALK_AR)

# Need a course word nearby (or a course already under discussion).
_WEAK_INFO_EN = re.compile("|".join((
    r"\bavailable\b", r"\bcover(?:s|ed)?\b", r"\bteach(?:es)?\b", r"\blearn\b",
    r"\bhow long\b", r"\bduration\b", r"\bcosts?\b", r"\bprice\b", r"\bfees?\b",
    r"\bhow much\b", r"\bonline\b", r"\boffline\b", r"\bin[- ]person\b",
    r"\bformat\b", r"\bschedule\b", r"\bstart date\b", r"\bdifference\b",
    r"\bcompare\b", r"\bsuitable for\b", r"\bwho is\b", r"\binclude[sd]?\b",
    r"\boutcomes?\b", r"\bcontent\b", r"\blevel\b", r"\bcertificate\b",
    r"\btell me about\b", r"\bdetails?\b", r"\bwhat (?:does|is|are)\b",
)))
_WEAK_INFO_AR = (
    "متاح", "يغطي", "بيغطي", "محتوي", "يشرح", "بيشرح", "هتعلم", "هاتعلم",
    "مده", "قد ايه", "كام", "سعر", "بكام", "تكلفه", "رسوم", "اونلاين",
    "حضوري", "اوفلاين", "شروط", "الفرق", "مناسب ل", "تفاصيل", "مستوي",
    "شهاده", "ما هو", "ما هي", "ايه هو", "ايه هي",
)


_LISTING_EN = re.compile(
    rf"\b(?:which|what|list|show)\b.{{0,30}}\b{_COURSE_WORD}\b.{{0,30}}"
    rf"\b(?:available|offer(?:s|ed)?|have|there|provide[sd]?|run)\b"
    rf"|\b(?:list|show)(?: me)?(?: all)?(?: the| your)?(?: available)? (?:training )?{_COURSE_WORD}\b"
    rf"|\ball (?:the |your |of your )?(?:available )?(?:training )?{_COURSE_WORD}\b"
    rf"|\bdo you (?:have|offer|provide|run) (?:any )?(?:training )?{_COURSE_WORD}\b"
)
_LISTING_AR = ("الكورسات المتاحه", "الدورات المتاحه", "ايه الكورسات", "ايه الدورات",
               "كل الكورسات", "كل الدورات", "ما هي الدورات", "ما هي الكورسات", "الدورات الموجوده",
               "جميع الدورات", "جميع الكورسات", "الدورات التي تقدمها", "الدورات اللي بتقدمها",
               "عندكم دورات", "عندكم كورسات", "هل لديكم دورات", "ما الدورات")

# Added to the retrieval query of catalogue-listing questions, so the
# search favours course descriptions over generic headings.
CATALOGUE_LISTING_HINT = "List of all courses offered at each level."


def is_catalogue_listing(text: str) -> bool:
    """'Which courses do you offer?' / 'ايه الكورسات المتاحة؟'"""
    if not isinstance(text, str):
        return False
    arabic = normalize_for_matching(transliterate_arabizi(text))
    return bool(_LISTING_EN.search(text.casefold())) or any(cue in arabic for cue in _LISTING_AR)


def names_a_course(text: str) -> bool:
    """True when the message itself mentions a course ("the Python course")."""
    if not isinstance(text, str):
        return False
    arabic = normalize_for_matching(transliterate_arabizi(text))
    return bool(_COURSE_NOUN_EN.search(text.casefold())) or any(
        noun in arabic for noun in _COURSE_NOUN_AR
    )


def classify_course_intent(text: str, course_context: bool = False) -> str:
    """Return RECOMMENDATION, COURSE_QUESTION or GENERAL."""
    if not isinstance(text, str) or not text.strip():
        return GENERAL

    latin = text.casefold()
    arabic = normalize_for_matching(transliterate_arabizi(text))
    arabic_plain = normalize_for_matching(text)

    if _RECOMMENDATION_EN.search(latin) or any(
        cue in candidate
        for candidate in (arabic, arabic_plain)
        for cue in _RECOMMENDATION_AR
    ):
        return RECOMMENDATION

    if (
        _STRONG_INFO_EN.search(latin)
        or any(cue in arabic for cue in _STRONG_INFO_AR)
        or is_catalogue_listing(text)
    ):
        return COURSE_QUESTION

    has_course_word = bool(_COURSE_NOUN_EN.search(latin)) or any(
        noun in arabic for noun in _COURSE_NOUN_AR
    )

    if _looks_like_question(text, arabic) and (
        _ACADEMY_INFO_EN.search(latin)
        or any(cue in arabic for cue in _ACADEMY_INFO_AR)
        or (has_course_word and _AVAILABILITY_EN.search(latin))
    ):
        return COURSE_QUESTION
    has_weak_cue = bool(_WEAK_INFO_EN.search(latin)) or any(
        cue in arabic for cue in _WEAK_INFO_AR
    )

    if has_weak_cue and (has_course_word or course_context):
        return COURSE_QUESTION

    if _NEEDS_A_COURSE_EN.search(latin) or any(
        cue in candidate for candidate in (arabic, arabic_plain) for cue in _NEEDS_A_COURSE_AR
    ):
        return RECOMMENDATION

    return GENERAL
