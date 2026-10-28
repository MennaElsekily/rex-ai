"""
Rule-based parsing of curriculum requests (English and Arabic).

Pure functions only: no I/O, no LLM. Nothing here guesses beyond what
the user wrote -- when a value is not clearly present it is left as
None so the agent can fall back to a default or ask the user.

Matching is done on a normalized copy of the text (see
agents/text_utils.py) and values are sliced from the original text, so
Arabic topics keep their original spelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agents.text_utils import normalize_for_matching, prepare

MAX_TOPIC_CHARS = 120
MAX_AUDIENCE_CHARS = 60
MAX_MODULE_TITLE_CHARS = 80


@dataclass(frozen=True)
class ParsedRequest:
    """Generation settings found in a request."""

    topic: Optional[str]
    training_hours: Optional[int]
    target_audience: Optional[str]
    # The request with the hours/audience phrases removed (used both for
    # topic extraction and for searching examples).
    remaining_text: str
    # The request with only the hours phrase removed (audience words can
    # still help a search such as "curriculum for teachers").
    hours_free_text: str = ""
    # True when the request gave hours per week/day (not a total).
    ignored_hours_rate: bool = False


@dataclass(frozen=True)
class Refinement:
    """One change the user asked for on an existing draft."""

    kind: str  # set_hours | add_module | add_capstone | set_audience | unsupported
    value: object = None


# --------------------------------------------------------------------
# Training hours
# --------------------------------------------------------------------

# "40 hours", "40-hour", "40hrs", "7.5 hours", "٤٠ ساعة", "40 ساعات"
_HOURS_RE = re.compile(
    r"(?<![\w.,])(\d{1,4})(?:[.,](\d{1,2}))?\s*-?\s*"
    r"(?:hours?|hrs?|ساعه|ساعات)(?!\w)"
    # "5 hours per week", "10 hours a day": a rate, not a total.
    r"(?P<rate>\s*(?:per|a|an|each|every|/)\s*"
    r"(?:week|day|month|session|class|lesson|night|weekday)s?(?!\w)"
    r"|\s*(?:في|كل)\s+(?:ال)?(?:اسبوع|يوم|شهر)(?!\w)"
    r"|\s*(?:اسبوعيا|يوميا|شهريا)(?!\w))?"
)

# Words that introduce a duration and should leave with it
# ("for 40 hours", "lasting 40 hours", "لمدة 40 ساعة").
_LEFT_CONNECTORS = re.compile(
    r"(?:^|\s)(?:for|of|over|within|lasting|taking|running|spanning|with|"
    r"a\s+total\s+of|totaling|totalling|total\s+of|total|about|around|"
    r"approximately|roughly|at|in|to|"
    r"(?:(?:that|which)\s+)?(?:lasts?|takes?|runs?|spans?|is|are|"
    r"will\s+be|should\s+be)|"
    r"لمده|بمده|مدته|مدتها|في|خلال|الي|اجعلها|اجعله)\s*$"
)


def _extend_left(norm: str, start: int) -> int:
    while True:
        match = _LEFT_CONNECTORS.search(norm[:start])
        if not match:
            return start
        start = match.start()


def find_hours(
    clean: str, norm: str
) -> tuple[Optional[int], Optional[tuple[int, int]]]:
    """
    Return (hours, span-to-remove) for the first duration, if any.

    A rate such as "5 hours per week" is not a total duration: it comes
    back as (None, span) so the phrase is removed from the text but no
    hours are claimed.
    """
    match = _HOURS_RE.search(norm)
    if not match:
        return None, None

    if match.group("rate"):
        return None, (_extend_left(norm, match.start()), match.end())

    whole = int(match.group(1))
    fraction = match.group(2)
    value = whole + (int(fraction) / (10 ** len(fraction)) if fraction else 0)
    hours = int(value + 0.5)  # round half up

    return hours, (_extend_left(norm, match.start()), match.end())


# --------------------------------------------------------------------
# Target audience
# --------------------------------------------------------------------

_AUD_MODS = (
    r"(?:absolute|complete|total|young|adult|high\s+school|middle\s+school|"
    r"primary\s+school|secondary\s+school|elementary\s+school|school|"
    r"university|college|graduate|undergraduate|postgraduate|new|junior|"
    r"senior|entry-level|non-technical|technical|corporate|healthcare|"
    r"business|first-time|first\s+time|early-career|mid-career|busy)"
)
_AUD_HEADS = (
    r"(?:beginners?|students?|pupils?|teachers?|educators?|instructors?|"
    r"trainers?|managers?|supervisors?|leaders?|executives?|professionals?|"
    r"employees?|staff|workers?|engineers?|developers?|programmers?|"
    r"analysts?|graduates?|learners?|trainees?|children|kids|teenagers?|"
    r"adults?|newcomers?|freshers?|juniors?|seniors?|nurses|doctors|"
    r"entrepreneurs|freelancers|job\s+seekers|new\s+hires?|everyone|anyone)"
)
_AUD_PHRASE = rf"(?:{_AUD_MODS}\s+){{0,2}}{_AUD_HEADS}"

_AUD_LABEL_RE = re.compile(
    rf"\b(?:target\s+)?audience\s*(?:is|are|:|=|of)?\s*({_AUD_PHRASE})\b"
)
_AUD_EN_RE = re.compile(
    r"\b(?:(?:designed|suitable|intended|aimed|targeted|tailored|geared)"
    r"\s+(?:for|at|to|towards?)|aimed\s+at|targeting|for)\s+"
    rf"({_AUD_PHRASE})\b"
)

_AR_SIMPLE = (
    r"لل?(?:مبتدئين|مبتدئات|مبتدئ|معلمين|معلمات|مدربين|مدرسين|مديرين|"
    r"موظفين|مهنيين|خريجين|متعلمين|محترفين|مبرمجين|مهندسين|محللين|"
    r"اداريين|شباب|اطفال|طلاب|طالبات)"
)
_AR_IDAFA = (
    r"ل(?:طلاب|طالبات|معلمي|معلمات|مدربي|مديري|موظفي|خريجي|مهندسي|"
    r"محللي|مبرمجي|اطفال)(?:\s+ال\S+)?"
)
_AUD_AR_RE = re.compile(rf"(?<!\w)({_AR_IDAFA}|{_AR_SIMPLE})")


def fix_arabic_lam(phrase: str) -> str:
    """'للمبتدئين' -> 'المبتدئين'; 'لطلاب الجامعة' -> 'طلاب الجامعة'."""
    if phrase.startswith("لل"):
        return "ال" + phrase[2:]
    if phrase.startswith("ل"):
        return phrase[1:]
    return phrase


def _tidy_audience(value: str) -> Optional[str]:
    value = value.strip(" \t.,;:!?؟،-")
    if not value or len(value) > MAX_AUDIENCE_CHARS:
        return None
    if not any(ch.isalpha() for ch in value):
        return None
    if value.isascii():
        value = value[0].upper() + value[1:]
    return value


def find_audience(
    clean: str, norm: str
) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """Return (audience, span-to-remove) for an obvious audience phrase."""
    for pattern in (_AUD_LABEL_RE, _AUD_EN_RE):
        match = pattern.search(norm)
        if match:
            value = _tidy_audience(clean[match.start(1):match.end(1)])
            if value:
                return value, match.span()

    match = _AUD_AR_RE.search(norm)
    if match:
        value = _tidy_audience(
            fix_arabic_lam(clean[match.start(1):match.end(1)])
        )
        if value:
            return value, match.span(1)

    return None, None


# --------------------------------------------------------------------
# Creation intent and topic
# --------------------------------------------------------------------

_EN_VERBS = (
    r"(?:create|generate|design|build|develop|make|prepare|draft|write|"
    r"put\s+together|set\s+up)"
)
_EN_NOUN = (
    r"(?:curriculum|curricula|curriculums|syllabus|course|"
    r"training\s+program(?:me)?|learning\s+path)"
)
_EN_ADJ = (
    r"(?:new|training|learning|short|complete|full|detailed|brief|basic|"
    r"custom|sample|accredited|certified|advanced|intermediate|beginner|"
    r"online|interactive|practical|hands-on|comprehensive|professional|"
    r"official|standard|simple|quick|good|great|modern|updated)"
)
# Noun directly after the verb: "create a (new) curriculum ..."
_EN_A = re.compile(
    rf"\b{_EN_VERBS}\s+(?:me\s+|us\s+)?(?:(?:a|an|the|some)\s+)?"
    rf"(?:{_EN_ADJ}\s+)*{_EN_NOUN}\b"
)
# Topic before the noun: "design a Python course ..."
_EN_B = re.compile(
    rf"\b{_EN_VERBS}\s+(?:me\s+|us\s+)?(?:a|an|the)\s+"
    r"(?P<topic>(?:(?!(?:for|about|on|in|of|to|with|and)\b)[\w+#.&/-]+\s+)"
    rf"{{1,6}}?){_EN_NOUN}\b"
)

_AR_VERBS = (
    r"(?:اعداد|انشاء|انشئ|انشي|صمم|اعد|اعمل|ابني|اكتب|جهز|حضر|طور)"
)
_AR_NOUN = r"(?:منهج|مناهج|منهاج|دوره|دورات|مقرر)"
_AR_A = re.compile(
    rf"(?:^|\s){_AR_VERBS}\s+(?:لي\s+)?{_AR_NOUN}"
    r"(?:\s+(?:تدريبي|تدريبيه|دراسي|دراسيه|تعليمي|تعليميه))?(?=\s|$)"
)

_EN_CONNECT = re.compile(
    r"^[\s,:;-]*(?:(?:about|on|for|in|covering|regarding|concerning|around|"
    r"focused\s+on|focusing\s+on|titled|called|named|that\s+covers?|"
    r"that\s+teaches|to\s+teach|teaching|dealing\s+with)\b[\s,:;-]*)+"
)
_EN_REJECT = re.compile(
    r"^(?:and|with|that|which|then|plus|also|to|please|thanks?)\b"
)
_AR_CONNECT = re.compile(
    r"^[\s،,:;-]*(?:(?:عن|حول|في|بعنوان|بخصوص|لتعليم|يغطي|تغطي)(?=\s)"
    r"[\s،,:;-]*)+"
)
_AR_REJECT = re.compile(
    r"^(?:ثم|و|مع|كذلك|وايضا|كما)(?=\s|$)|^و(?:حمل|اعلان|تسويق)"
)
_TRAILING = re.compile(
    r"(?:\s+(?:please|thanks|thank\s+you|for\s+me|now))+$", re.IGNORECASE
)

# Phrases that always mean "make me a new curriculum/course". The
# regexes above cover these too; the list documents the contract with
# the orchestrator's routing and is checked by a test.
CREATION_PHRASES = (
    "create a curriculum",
    "create curriculum",
    "generate a curriculum",
    "generate curriculum",
    "design a curriculum",
    "build a curriculum",
    "develop a curriculum",
    "make a curriculum",
    "create a course",
    "generate a course",
    "design a course",
    "build a course",
    "develop a course",
    "make a course",
    "create a training course",
    "design a training course",
    "إنشاء منهج",
    "انشاء منهج",
    "أنشئ منهج",
    "انشئ منهج",
    "صمم منهج",
    "صمّم منهج",
    "إعداد منهج",
    "اعداد منهج",
    "إنشاء دورة",
    "انشاء دورة",
    "أنشئ دورة",
    "صمم دورة",
    "صمّم دورة",
    "إعداد دورة",
)
_NORMALIZED_CREATION_PHRASES = tuple(
    normalize_for_matching(phrase) for phrase in CREATION_PHRASES
)


def has_creation_intent(text: str) -> bool:
    """True when the user asks for a NEW curriculum/course."""
    norm = normalize_for_matching(text)

    if any(phrase in norm for phrase in _NORMALIZED_CREATION_PHRASES):
        return True

    return bool(
        _EN_A.search(norm) or _EN_B.search(norm) or _AR_A.search(norm)
    )


# "... about SQL and a marketing campaign": the marketing part belongs to
# the next agent in the plan, not to the curriculum topic.
_MARKETING_TAIL_EN = re.compile(
    r"(?:\s*,|\s+(?:and|then|plus|with|as\s+well\s+as))\s+(?:also\s+)?"
    r"(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){0,2}?"
    r"(?:marketing|campaigns?|advertis\w*|promotion\w*|social\s+media|"
    r"linkedin)\b.*$",
    re.IGNORECASE,
)
_MARKETING_TAIL_AR = re.compile(
    r"\s+(?:ثم|و)\s*(?:حمل|[اإ]علان|تسويق).*$"
)


def _clean_topic(raw: str) -> Optional[str]:
    raw = _MARKETING_TAIL_EN.sub("", raw)
    raw = _MARKETING_TAIL_AR.sub("", raw)
    # "... about Python and make it 50 hours": the instruction is not
    # part of the topic.
    raw = _CONJ_EN.sub("", raw)
    raw = _CONJ_AR.sub("", raw)
    topic = raw.strip(" \t.,;:!?؟،-\"'")
    previous = None
    while previous != topic:
        previous = topic
        topic = _TRAILING.sub("", topic).strip(" \t.,;:!?؟،-")

    if not topic or len(topic) > MAX_TOPIC_CHARS:
        return None
    if not any(ch.isalpha() for ch in topic):
        return None

    return topic


def extract_topic(remaining: str) -> Optional[str]:
    """
    Pull the subject out of a creation request, e.g.
    "Create a curriculum about Python" -> "Python". Returns None when
    no clear topic follows (for example "Create a curriculum and a
    marketing campaign"), so the caller can ask instead of guessing.
    """
    clean, norm = prepare(remaining)

    match = _EN_A.search(norm)
    if match:
        rest_clean = clean[match.end():]
        rest_clean = _EN_CONNECT.sub("", rest_clean, count=1)
        if _EN_REJECT.match(normalize_for_matching(rest_clean).strip()):
            return None
        return _clean_topic(rest_clean)

    match = _EN_B.search(norm)
    if match:
        return _clean_topic(clean[match.start("topic"):match.end("topic")])

    match = _AR_A.search(norm)
    if match:
        rest_clean = clean[match.end():]
        rest_clean = _AR_CONNECT.sub("", rest_clean, count=1).strip()
        if _AR_REJECT.match(normalize_for_matching(rest_clean)):
            return None
        if rest_clean.startswith("لل") or rest_clean.startswith("لت"):
            rest_clean = fix_arabic_lam(rest_clean) if rest_clean.startswith(
                "لل"
            ) else rest_clean[1:]
        return _clean_topic(rest_clean)

    return None


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return re.sub(r"\s+", " ", text).strip()


def parse_request(text: str) -> ParsedRequest:
    """Extract hours, audience and topic from a (creation) request."""
    clean, norm = prepare(text)

    hours, hours_span = find_hours(clean, norm)
    audience, audience_span = find_audience(clean, norm)

    spans = [span for span in (hours_span, audience_span) if span]
    remaining = _remove_spans(clean, spans)
    hours_free = _remove_spans(clean, [hours_span] if hours_span else [])

    return ParsedRequest(
        topic=extract_topic(remaining),
        training_hours=hours,
        target_audience=audience,
        remaining_text=remaining,
        hours_free_text=hours_free,
        ignored_hours_rate=hours is None and hours_span is not None,
    )


# --------------------------------------------------------------------
# Accreditation / gap analysis (not supported)
# --------------------------------------------------------------------

_ACCREDITATION_RE = re.compile(
    r"\baccredit"
    r"|\bgap[\s-]+(?:analysis|assessment)\b"
    r"|\bstandards?\s+alignment\b"
    r"|\balign(?:ed|ment|ing)?\s+(?:[\w-]+\s+){0,4}?(?:with|to|against)\s+"
    r"(?:the\s+|any\s+|our\s+|national\s+|international\s+|relevant\s+)*"
    r"standards?\b"
    r"|\bcompar(?:e|ing)\s+(?:[\w-]+\s+){0,4}?(?:with|to|against)\s+"
    r"(?:the\s+|any\s+|our\s+|national\s+|international\s+|relevant\s+)*"
    r"standards?\b"
    r"|اعتماد(?!\s+علي)"
    r"|تحليل\s+(?:ال)?فجو"
    r"|مطابقه\s+(?:ال)?معايير"
    r"|موا?ءمه|موائمه"
)


def is_accreditation_request(text: str) -> bool:
    """Accreditation alignment / gap analysis (not implemented)."""
    return bool(_ACCREDITATION_RE.search(normalize_for_matching(text)))


# --------------------------------------------------------------------
# Refinements of an existing draft
# --------------------------------------------------------------------

# A message is only treated as an edit of the draft when it STARTS like
# one ("add a capstone", "make it 40 hours", "من فضلك اجعلها 40 ساعة"), or
# is exactly a bare duration ("40 hours"). Searches, and other agents'
# answers that merely mention hours or a capstone, are left alone.
_EDIT_VERBS_EN = (
    r"(?:make|change|update|modify|adjust|edit|revise|rewrite|add|remove|"
    r"delete|replace|increase|decrease|reduce|extend|shorten|lengthen|set|"
    r"include|insert|append|put|switch|turn|raise|lower|cut|drop|exclude|"
    r"tailor|adapt|retarget)"
)
# Arabic verbs may carry an attached pronoun: اجعلها، اجعله، ضيفها.
_AR_PRONOUN = r"(?:ها|هم|ني|نا|ه)?"
_EDIT_VERBS_AR = (
    r"(?:غير|عدل|اضف|ضيف|احذف|ازل|امسح|اجعل|زد|قلل|خفض|ادرج|ضمن|"
    rf"استبعد|بدل){_AR_PRONOUN}"
)
_LEAD_INS = (
    r"(?:(?:please|now|also|then|and|ok|okay|next|can\s+you|could\s+you|"
    r"would\s+you|let[’']?s|i(?:[’']d|\s+would)?\s+(?:like|want|need)\s+"
    r"(?:you\s+)?to|من\s+فضلك|ممكن|لو\s+سمحت|الان|ثم)\s+)*"
)
_EDIT_START_RE = re.compile(
    rf"^\s*{_LEAD_INS}(?:{_EDIT_VERBS_EN}\b|{_EDIT_VERBS_AR}(?!\w))"
)
# Same, minus "make": "make a curriculum about X" is a creation request.
_REFINE_FIRST_RE = re.compile(
    rf"^\s*{_LEAD_INS}(?:(?!make\b){_EDIT_VERBS_EN}\b|{_EDIT_VERBS_AR}(?!\w))"
)
_BARE_HOURS_RE = re.compile(
    r"^(?:to\s+|الي\s+)?\d{1,4}(?:[.,]\d{1,2})?\s*-?\s*"
    r"(?:hours?|hrs?|ساعه|ساعات)$"
)
_NEGATION_RE = re.compile(
    r"\b(?:remove|delete|drop|without|exclude|get\s+rid\s+of|take\s+out)\b"
    r"|احذف|ازل|امسح|بدون|استبعد|شيل"
)
_DRAFT_PART_RE = re.compile(
    r"\b(?:module|unit|chapter|section|lesson|capstone|assessment|exercise|"
    r"outcome|objective|prerequisite)s?\b"
    r"|وحده|وحدات|فصل|موديول|مشروع|تقييم|تمرين|كابستون"
)
_CAPSTONE_RE = re.compile(
    r"capstone|مشروعا?\s+(?:ختامي|نهائي|التخرج|تخرج)ا?"
    r"|المشروع\s+(?:الختامي|النهائي)|كابستون"
)
_ADD_VERB_RE = re.compile(
    r"\b(?:add|include|insert|append|attach|incorporate|with|need|want|"
    r"plus|put)\b|اضف|اضاف|ضيف|ضمن|ادرج|اريد|ابغي|محتاج"
)
_ADD_MODULE_RE = re.compile(
    r"\b(?:add|include|insert|append)\s+(?:a\s+|an\s+|one\s+|another\s+)?"
    r"(?:new\s+|extra\s+|additional\s+)?(?:module|unit|chapter|section|"
    r"lesson)s?\b(?:\s+(?:about|on|for|covering|called|titled|named|"
    r"regarding|that\s+covers?)|\s*:)?\s*(?P<title>.*)$"
)
_AR_ADD_MODULE_RE = re.compile(
    r"(?:اضف|ضيف|اضافه|ادرج)\s+(?:وحده|موديول|فصل|درس|قسم)"
    r"(?:\s+(?:جديده|اضافيه))?\s*(?:(?:عن|حول|بعنوان|في)\s+)?(?P<title>.*)$"
)
_AUD_CHANGE_RE = re.compile(
    r"\b(?:target\s+)?audience\b\s*(?:to|=|:|is|should\s+be|will\s+be|as)?"
    r"\s*(?P<aud>[^.;!?]+)"
)
_AR_AUD_CHANGE_RE = re.compile(
    r"(?:الجمهور|الفئه)(?:\s+المستهدف\w*)?\s*"
    r"(?:(?:الي|هو|هم|ليكون|ليصبح)\s+)?(?P<aud>[^.؟!؛]+)"
)
_AUD_CUE_RE = re.compile(
    r"\b(?:change|switch|instead|update|set|retarget|adapt|adjust|tailor|"
    r"make\s+it)\b|غير|بدل|عدل|اجعل"
)
_CONJ_EN = re.compile(
    r"(?:\s+(?:and|then|also)|,)\s+(?:and\s+)?(?:also\s+)?"
    r"(?:make|change|set|add|include|update|increase|reduce|adjust|"
    r"extend|shorten)\b.*$"
)
_CONJ_AR = re.compile(r"\s+و(?:اجعل|غير|اضف|عدل|زد|قلل).*$")
_TARGET_SUFFIX_EN = re.compile(
    r"\s+(?:to|in|into|for)\s+(?:the|this|my|that|our)\s+"
    r"(?:\w+\s+)?(?:curriculum|course|draft|syllabus|program(?:me)?)$"
)
_TARGET_SUFFIX_AR = re.compile(r"\s+(?:الي|ل)\s*(?:ال)?(?:منهج|دوره)\w*$")


def _cut_title(title_clean: str, *, arabic: bool) -> Optional[str]:
    title = title_clean.strip()
    if arabic:
        title = _CONJ_AR.sub("", title)
        title = _TARGET_SUFFIX_AR.sub("", title)
    else:
        title = _CONJ_EN.sub("", title)
        title = _TARGET_SUFFIX_EN.sub("", title)
    title = re.sub(r"\s+", " ", title).strip(" \t.,;:!?؟،-\"'")

    if not title:
        return None

    return title[:MAX_MODULE_TITLE_CHARS]


def _audience_change(clean: str, norm: str) -> Optional[str]:
    """Audience named in a 'change the audience to ...' style message."""
    for pattern, arabic in ((_AUD_CHANGE_RE, False), (_AR_AUD_CHANGE_RE, True)):
        match = pattern.search(norm)
        if not match:
            continue

        raw = clean[match.start("aud"):match.end("aud")].strip()
        raw = _CONJ_AR.sub("", raw) if arabic else _CONJ_EN.sub("", raw)
        raw = re.sub(
            r"\s+(?:instead|please)$", "", raw, flags=re.IGNORECASE
        )
        if arabic:
            raw = fix_arabic_lam(raw)
        elif re.match(r"(?i)^(?:for|to|be)\s+", raw):
            raw = re.sub(r"(?i)^(?:for|to|be)\s+", "", raw)

        value = _tidy_audience(raw)
        if value:
            return value

    audience, _ = find_audience(clean, norm)
    if audience and _AUD_CUE_RE.search(norm):
        return audience

    return None


def detect_refinements(text: str) -> list[Refinement]:
    """
    Find refinement requests in a message about an existing draft:
    change hours, add a module, add a capstone, change the audience.
    Unsupported edits (such as removals) are reported as 'unsupported'.
    """
    clean, norm = prepare(text)
    refinements: list[Refinement] = []

    if not (
        _EDIT_START_RE.match(norm)
        or _BARE_HOURS_RE.match(norm.strip(" .!?؟"))
    ):
        return []

    hours, hours_span = find_hours(clean, norm)
    work_clean = _remove_spans(clean, [hours_span]) if hours_span else clean
    work_clean, work_norm = prepare(work_clean)

    # Audience.
    audience = _audience_change(work_clean, work_norm)
    if audience:
        refinements.append(Refinement("set_audience", audience))

    # Removals are not supported; say so instead of ignoring them.
    if _NEGATION_RE.search(work_norm) and _DRAFT_PART_RE.search(work_norm):
        return [Refinement("unsupported", "remove")]

    # Module / capstone.
    capstone_requested = False
    for pattern, arabic in ((_ADD_MODULE_RE, False), (_AR_ADD_MODULE_RE, True)):
        match = pattern.search(work_norm)
        if match:
            title = _cut_title(
                work_clean[match.start("title"):match.end("title")],
                arabic=arabic,
            )
            if title and _CAPSTONE_RE.search(normalize_for_matching(title)):
                capstone_requested = True
            else:
                refinements.append(
                    Refinement("add_module", title or "Additional Module")
                )
            break

    if (
        not capstone_requested
        and _CAPSTONE_RE.search(work_norm)
        and _ADD_VERB_RE.search(work_norm)
    ):
        capstone_requested = True

    if capstone_requested:
        refinements.append(Refinement("add_capstone"))

    # Hours last, so an explicit duration overrides any redistribution.
    if hours is not None:
        refinements.append(Refinement("set_hours", hours))

    return refinements


def looks_like_edit_attempt(text: str) -> bool:
    """A message that tries to edit the draft (even if we can't do it)."""
    return bool(_EDIT_START_RE.match(normalize_for_matching(text)))


def refinement_takes_priority(text: str) -> bool:
    """
    Starts with an edit verb other than "make" ("add a module on how to
    design a course"), so it is an edit even though it mentions creating.
    """
    return bool(_REFINE_FIRST_RE.match(normalize_for_matching(text)))
