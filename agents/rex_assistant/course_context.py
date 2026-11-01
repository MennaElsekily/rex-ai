"""
Which course is a direct question about? Deterministic course
identification for REX's question mode (no LLM involved).

The candidate courses come from the knowledge base itself: the course ids
and names in the indexed metadata (the list_courses MCP tool), or, when
that list is unavailable, the courses named in the retrieved evidence.
Nothing here is tied to one dataset, question or course name.

- resolve_reference(): does a message name one course, several equally
  well (ambiguous: "the Python course" when three course names contain
  "Python"), or none? A course id ("DA-301") or a complete course name
  is a clear reference; otherwise the course whose name shares the most
  distinctive words with the message wins, and a tie is ambiguous.
- refers_back(): "How long is it?", "What does this course cost?",
  "مدته كام؟" -- a question about a course already under discussion.
- parse_choice(): the learner's reply to "Which course do you mean?":
  a number ("2"), an ordinal ("the second", "التانية") or a course name.
- scope_evidence(): keep the passages of the selected course(s) and the
  general academy passages (FAQs, admission, instructors); drop passages
  about other courses so they can neither leak into the answer nor be
  listed as its sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agents.text_utils import normalize_for_matching
from knowledge_base.language import egyptian_to_msa, transliterate_arabizi

NAMED = "named"          # one course, or several each named clearly
AMBIGUOUS = "ambiguous"  # several courses match the same words equally
NONE = "none"            # no course of the catalogue is named

MAX_CHOICES = 8

# Words that occur in course names but say little on their own. They
# count only next to a distinctive name word, or two of them together
# ("social media", "full stack"): "Is the course full?" names no course.
_GENERIC_NAME_WORDS = frozenset({
    "introduction", "intro", "fundamental", "fundamentals", "basic", "basics",
    "advanced", "practitioner", "building", "application", "development",
    "developer", "data", "learning", "full", "stack", "paid", "social", "media",
    "power", "business", "digital", "mobile", "app", "campaign", "optimization",
    "analytic", "analytics", "beginner", "intermediate", "professional",
})

# Function words, never evidence of a course.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "at",
    "by", "from", "into", "about", "is", "are", "js", "s", "t", "vs",
})

_LATIN_WORD = re.compile(r"[a-z0-9]+")
_COURSE_ID = re.compile(r"\b([a-z]{2,4})[\s-]?(\d{2,4})\b")
_CODE_PREFIX = re.compile(r"\b([A-Z]{2,4})\b")
_PREFIX_BEFORE_COURSE = re.compile(r"\b([a-z]{2,4})\s+(?:course|program|programme|class|track)s?\b")


def _stem(word: str) -> str:
    """'apis' -> 'api', 'analysts' -> 'analyst'; 'analysis' stays."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "is", "us")):
        return word[:-1]
    return word


def _latin_words(text: str) -> set[str]:
    return {_stem(w) for w in _LATIN_WORD.findall(text.casefold()) if w not in _STOPWORDS}


def _matching_text(text: str) -> str:
    """
    Latin view of a message: Arabizi words to Arabic, Arabic-spelled tech
    names back to Latin ("بايثون" -> "Python"), so the Latin course words
    of an Arabic, Arabizi or mixed message can be compared with the names.
    """
    return egyptian_to_msa(transliterate_arabizi(text or ""))


def _id_key(course_id: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", (course_id or "").casefold())


# ---------------------------------------------------------------------
# Course records
# ---------------------------------------------------------------------

def clean_course(course: Any) -> Optional[dict[str, Optional[str]]]:
    """{course_id, course_name} with string values, or None when unusable."""
    if not isinstance(course, dict):
        return None
    course_id, name = course.get("course_id"), course.get("course_name")
    course_id = course_id.strip() if isinstance(course_id, str) and course_id.strip() else None
    name = name.strip() if isinstance(name, str) and name.strip() else None
    if not (course_id or name):
        return None
    return {"course_id": course_id, "course_name": name}


def clean_courses(courses: Any) -> list[dict[str, Optional[str]]]:
    """Valid, distinct courses from state or a tool result; anything else is dropped."""
    result, seen = [], set()
    for course in courses if isinstance(courses, list) else []:
        cleaned = clean_course(course)
        if cleaned and course_key(cleaned) not in seen:
            seen.add(course_key(cleaned))
            result.append(cleaned)
    return result


def course_key(course: dict[str, Any]) -> str:
    return _id_key(course.get("course_id")) or (course.get("course_name") or "").casefold()


def course_label(course: dict[str, Any]) -> str:
    """'DA-301 Python for Data Analysis'."""
    course_id, name = course.get("course_id") or "", course.get("course_name") or ""
    return f"{course_id} {name}".strip() if course_id and course_id not in name else (name or course_id)


def same_courses(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    return {course_key(c) for c in a} == {course_key(c) for c in b}


def known_courses(courses: list[dict[str, Any]], catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Courses from saved state that still exist. With no catalogue to check
    against, the saved courses are kept; an unknown course is dropped.
    """
    if not catalogue:
        return courses
    by_key = {course_key(c): c for c in catalogue}
    return [by_key[course_key(c)] for c in courses if course_key(c) in by_key]


# ---------------------------------------------------------------------
# Which course does a message name?
# ---------------------------------------------------------------------

@dataclass
class CourseReference:
    kind: str
    courses: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Score:
    course: dict[str, Any]
    exact_id: bool
    distinctive: set[str]
    generic: set[str]
    prefix: bool
    complete: bool

    @property
    def rank(self) -> tuple[int, int]:
        return len(self.distinctive), len(self.generic)


def _score(course: dict[str, Any], words: set[str], ids: set[str], prefixes: set[str]) -> _Score:
    name_words = _latin_words(course.get("course_name") or "")
    distinctive_name = name_words - _GENERIC_NAME_WORDS
    matched = words & name_words
    key = _id_key(course.get("course_id"))
    prefix = re.match(r"[a-z]+", key)
    return _Score(
        course=course,
        exact_id=bool(key) and key in ids,
        distinctive=matched - _GENERIC_NAME_WORDS,
        generic=matched & _GENERIC_NAME_WORDS,
        prefix=bool(prefix) and prefix.group(0) in prefixes and prefix.group(0) != key,
        complete=bool(distinctive_name) and distinctive_name <= matched,
    )


_COURSE_NOUN = re.compile(r"\b(?:course|program|programme|class|bootcamp|diploma)s?\b")
# Where a course phrase ends, reading back from its noun: determiners,
# question and attribute words ("the", "how", "online"), Arabizi articles.
# "for", "and", "with", "of", "to", "in" may be part of a name.
_NAME_INTERNAL = frozenset({"for", "and", "with", "of", "to", "in", "on"})
_MAX_PHRASE_WORDS = 8


def _course_phrases(text: str) -> list[set[str]]:
    """{'python', 'basic'} for "the Python Basics course" (Latin words only)."""
    phrases = []
    for noun in _COURSE_NOUN.finditer(text.casefold()):
        words = _LATIN_WORD.findall(text.casefold()[:noun.start()])
        phrase: list[str] = []
        for word in reversed(words):
            if len(phrase) == _MAX_PHRASE_WORDS or (
                (word in _FILLER_EN or word in _ARABIZI_FILLER) and word not in _NAME_INTERNAL
            ):
                break
            phrase.append(word)
        named = {_stem(w) for w in phrase if w not in _STOPWORDS and len(w) > 1 and not w.isdigit()}
        if named:
            phrases.append(named)
    return phrases


def _name_vocabulary(course: dict[str, Any]) -> set[str]:
    """The words a learner may use for a course: its name and code prefix."""
    prefix = re.match(r"[a-z]+", _id_key(course.get("course_id")))
    return _latin_words(course.get("course_name") or "") | ({prefix.group(0)} if prefix else set())


def resolve_reference(text: str, catalogue: list[dict[str, Any]]) -> CourseReference:
    """
    NAMED: the message names these courses (an id, a complete name, or
    the best-matching name). AMBIGUOUS: several courses match equally
    well -- ask which one. NONE: no course of `catalogue` is named.
    """
    if not catalogue or not isinstance(text, str) or not text.strip():
        return CourseReference(NONE)

    latin = _matching_text(text)
    words = _latin_words(latin)
    ids = {f"{letters}{digits}" for letters, digits in _COURSE_ID.findall(latin.casefold())}
    # A code prefix counts when written as one ("AI") or before a course
    # word ("ai courses") -- not Arabizi "da" ("this").
    prefixes = {p.casefold() for p in _CODE_PREFIX.findall(latin)} | {
        p for p in _PREFIX_BEFORE_COURSE.findall(latin.casefold())
    }
    scores = [_score(course, words, ids, prefixes) for course in catalogue]

    # 1. Course ids are unambiguous ("DA-101 or DA-201?" names both).
    exact = [s.course for s in scores if s.exact_id]
    if exact:
        return CourseReference(NAMED, exact)

    candidates = [s for s in scores if s.distinctive or len(s.generic) >= 2]

    # The words written before a course noun ("the Python Basics course")
    # must all belong to the course's name: a course REX does not know is
    # not taken for a similarly named one.
    phrases = _course_phrases(text)
    if candidates and phrases:
        candidates = [s for s in candidates if any(p <= _name_vocabulary(s.course) for p in phrases)]
        if not candidates:
            return CourseReference(NONE)
        # "the Excel course or the Django course": each phrase names one course.
        per_phrase = [[s for s in candidates if p <= _name_vocabulary(s.course)] for p in phrases]
        if len(phrases) >= 2 and all(len(match) == 1 for match in per_phrase):
            named = []
            for [score] in per_phrase:
                if score.course not in named:
                    named.append(score.course)
            if len(named) >= 2:
                return CourseReference(NAMED, named)

    # 2. Several complete names ("Excel for Business Analysis or Python for
    #    Data Analysis?"), unless one is contained in another's match.
    complete = [
        s for s in candidates
        if s.complete and not any(s.distinctive < o.distinctive for o in candidates)
    ]
    if len(complete) >= 2:
        return CourseReference(NAMED, [s.course for s in complete])

    # 3. The best-matching name; a tie is ambiguous.
    if candidates:
        best = max(s.rank for s in candidates)
        top = [s.course for s in candidates if s.rank == best]
        return CourseReference(NAMED if len(top) == 1 else AMBIGUOUS, top[:MAX_CHOICES])

    # 4. Only a course-code prefix ("the AI courses" -> AI-101, AI-201...).
    by_prefix = [s.course for s in scores if s.prefix]
    if by_prefix:
        return CourseReference(NAMED if len(by_prefix) == 1 else AMBIGUOUS, by_prefix[:MAX_CHOICES])

    return CourseReference(NONE)


def clearly_names(text: str, catalogue: list[dict[str, Any]]) -> bool:
    """
    The message names a course beyond doubt: its code, or every distinctive
    word of its name ("Python for Data Analysis"). A partial match ("data
    analyst" vs "SQL and Power BI for Data Analysts") is not clear -- as a
    reply to a discovery question it is the learner's own answer.
    """
    if not catalogue or not isinstance(text, str) or not text.strip():
        return False
    latin = _matching_text(text)
    words = _latin_words(latin)
    ids = {f"{letters}{digits}" for letters, digits in _COURSE_ID.findall(latin.casefold())}
    return any(s.exact_id or s.complete for s in (_score(c, words, ids, set()) for c in catalogue))


# ---------------------------------------------------------------------
# Does a question refer back to the course under discussion?
# ---------------------------------------------------------------------

def _normalized_set(words: str) -> frozenset[str]:
    return frozenset(normalize_for_matching(words).split())


# Question words, function words and course attributes: what a question
# about "the course" consists of when it names no course itself.
_FILLER_EN = frozenset("""
what whats how when where who whom which why whose is are was were be been being do does did
can could will would should shall may might must have has had i me my we our you your it its
this that these those there here they them their one ones the a an of for to in on at by with
about from into and or but so also too then than as if just please tell show give explain know
want need like more much many long cost costs price prices fee fees tuition duration last lasts
take takes took prerequisite prerequisites requirement requirements require required entry cover
covers covered content contents topic topics module modules syllabus curriculum teach teaches
taught teacher teachers instructor instructors trainer trainers learn study include includes
included online offline hybrid person format delivery mode schedule schedules start starts
starting begin begins date dates time times timing hour hours week weeks month months level
levels certificate certification accredited available availability seat seats open enrol enroll
register registration apply language suitable beginner beginners again same okay ok yes thanks
thank get got s t re ll d m ve course courses program programs programme programmes class classes
bootcamp bootcamps diploma diplomas any else exactly much
da dah deh di dih dy kors kurs dawra dawrah
""".split())
_ARABIZI_FILLER = frozenset({"el", "al", "fel", "fil", "bel", "lel", "3an", "3la", "ely", "elly"})

_FILLER_AR = _normalized_set("""
هل ما ماذا كيف متى امتى امتي متي اين فين مين من كم كام ايه إيه ازاي إزاي ليه لماذا بكام قد قديش
هو هي هم هذا هذه ده دي دا دى ذلك تلك دول في فى عن على علي الى إلى لي ليا انا أنا احنا يا و
مدة المدة مده المده مدته مدتها سعر السعر سعره سعرها تكلفة التكلفة تكلفته تكلفتها رسوم الرسوم
رسومه رسومها مصاريف المصاريف متطلبات المتطلبات متطلباته متطلباتها شروط الشروط شروطه شروطها
محتوى المحتوى محتواه محتواها يغطي بيغطي يشرح بيشرح يدرس بيدرس هتعلم هاتعلم اتعلم أتعلم هندرس
مواعيد المواعيد مواعيده مواعيدها يبدا يبدأ يبتدي بيبدا اونلاين أونلاين حضوري اوفلاين مستوى
المستوى مستواه مستواها شهادة الشهادة شهادته شهادتها مدرب المدرب مدربه مدربها المدربين المحاضر
ساعات ساعة ساعه اسابيع أسابيع اسبوع أسبوع شهور شهر لغة اللغة متاح متاحة متاحه طيب تمام ممكن
عايز عاوز عايزه اعرف أعرف لو سمحت كمان برضه برضو ايضا أيضا يعني بتاع بتاعه بتاعها الخاص
كورس الكورس كورسات الكورسات دورة الدورة دوره الدوره دورات الدورات برنامج البرنامج تفاصيل
ايه هو ايه هي اكتر أكثر قولي قول
""")

# References to one course. Strong: a pronoun or demonstrative ("it",
# "this course", "ده", "مدته" = its duration) -- always the course under
# discussion. Weak: "the course" / "الكورس", which may still be followed
# by a description ("the course on Kubernetes").
_STRONG_REFERENCE_EN = re.compile(
    r"\b(?:it|its)\b|\b(?:this|that|same)\s+(?:course|program|programme|class|bootcamp|diploma|one)\b"
    # Arabizi demonstratives ("el course da", "el dawra di").
    r"|\b(?:course|kors|kurs|dawra|dawrah|program)\s+(?:da|dah|deh|di|dih|dy)\b"
)
_WEAK_REFERENCE_EN = re.compile(r"\bthe\s+(?:course|program|programme|class|bootcamp|diploma)\b")
_STRONG_REFERENCE_AR = _normalized_set("""
ده دي دا دى هذا هذه ذلك تلك
مدته مدتها سعره سعرها تكلفته تكلفتها رسومه رسومها متطلباته متطلباتها شروطه شروطها محتواه
محتواها مواعيده مواعيدها مستواه مستواها شهادته شهادتها مدربه مدربها بتاعه بتاعها
""")
_WEAK_REFERENCE_AR = _normalized_set("الكورس الدورة الدوره البرنامج")
_AR_CLITICS = ("و", "ف", "ب", "ل", "ك")
_AR_WORD = re.compile(r"[ء-ي]+")


def _is_arabic_filler(word: str) -> bool:
    candidates = {word}
    if len(word) > 3 and word.startswith(_AR_CLITICS):
        candidates.add(word[1:])
    for candidate in list(candidates):
        if candidate.startswith("ال"):
            candidates.add(candidate[2:])
        else:
            candidates.add("ال" + candidate)
    return bool(candidates & _FILLER_AR)


def content_words(text: str) -> list[str]:
    """The words of a question beyond question words, attributes and 'the course'."""
    arabic = normalize_for_matching(transliterate_arabizi(text or ""))
    latin = [w for w in _LATIN_WORD.findall(arabic) if not w.isdigit() and len(w) > 1]
    words = [w for w in latin if w not in _FILLER_EN and _stem(w) not in _FILLER_EN]
    return words + [w for w in _AR_WORD.findall(arabic) if not _is_arabic_filler(w)]


def _references(text: str) -> tuple[bool, bool]:
    """(strong, weak) singular course references in the message."""
    # English patterns on the text as written (Arabizi transliteration
    # turns "course" into "كورس"); Arabic words also after it.
    latin = normalize_for_matching(text or "")
    words = set(_AR_WORD.findall(normalize_for_matching(transliterate_arabizi(text or ""))))
    strong = bool(_STRONG_REFERENCE_EN.search(latin) or words & _STRONG_REFERENCE_AR)
    weak = bool(_WEAK_REFERENCE_EN.search(latin) or words & _WEAK_REFERENCE_AR)
    return strong, weak


def refers_back(text: str) -> bool:
    """
    A question about the course under discussion that names no course:
    "How long is it?", "Is there a discount for it?", "What does the
    course cost?", "مدة الكورس كام؟", "Is it online?".
    """
    strong, _ = _references(text)
    return strong or not content_words(text)


def needs_a_course(text: str) -> bool:
    """
    A question about one unnamed course ("How much does it cost?", "How
    long is the course?") -- with no course under discussion, REX asks
    which one. "What are the prerequisites?" (no reference) is answered.
    """
    strong, weak = _references(text)
    return (strong or weak) and not content_words(text)


# ---------------------------------------------------------------------
# The learner's reply to "Which course do you mean?"
# ---------------------------------------------------------------------

_ORDINALS = {
    # Ordinals only: "one" would turn "the Django one" into option 1.
    **{word: index for index, words in enumerate((
        "first 1st awel awal",
        "second 2nd tany tani",
        "third 3rd talet talt",
        "fourth 4th rabe3",
        "fifth 5th khames",
        "sixth 6th",
        "seventh 7th",
        "eighth 8th",
    )) for word in words.split()},
}
_ORDINALS_AR = {
    normalize_for_matching(word): index
    for index, words in enumerate((
        "الأول الاول اول أول الأولى الاولى الاولي اولى واحد",
        "التاني الثاني تاني ثاني التانية الثانية التانيه الثانيه اتنين اثنين",
        "التالت الثالث تالت ثالث التالتة الثالثة التالته الثالثه تلاتة ثلاثة",
        "الرابع رابع الرابعة الرابعه اربعة",
        "الخامس خامس الخامسة الخامسه خمسة",
        "السادس سادس", "السابع سابع", "الثامن التامن ثامن",
    ))
    for word in words.split()
}
_LAST = {"last", "akher", normalize_for_matching("الأخير"), normalize_for_matching("الأخيرة"),
         normalize_for_matching("الاخيره"), normalize_for_matching("آخر")}
_MAX_CHOICE_WORDS = 6


def _choice_by_position(text: str, count: int) -> Optional[int]:
    """Index chosen by "2", "#2", "the second one", "التانية", or None."""
    normalized = normalize_for_matching(text or "")
    words = re.findall(r"[^\W_]+", normalized)
    if not words or len(words) > _MAX_CHOICE_WORDS:
        return None
    numbers = [int(w) for w in words if w.isdigit()]
    if len(numbers) == 1 and 1 <= numbers[0] <= count:
        return numbers[0] - 1
    for word in words:
        index = _ORDINALS.get(word, _ORDINALS_AR.get(word))
        if index is not None and index < count:
            return index
        if word in _LAST:
            return count - 1
    return None


def parse_choice(
    text: str, candidates: list[dict[str, Any]], catalogue: list[dict[str, Any]]
) -> CourseReference:
    """
    NAMED [course]: the reply picks one course. AMBIGUOUS: it narrows the
    options but still matches several. NONE: not an answer to the choice.
    With no candidates offered ("Which course do you mean?" after "How much
    is it?"), the reply is matched against the whole catalogue.
    """
    if candidates:
        # A name first ("the Django one"), then a position ("2", "the second").
        within = resolve_reference(text, candidates)
        if within.kind == NAMED and len(within.courses) == 1:
            return within
        if within.kind != NONE:
            return CourseReference(AMBIGUOUS, within.courses)
        index = _choice_by_position(text, len(candidates))
        if index is not None:
            return CourseReference(NAMED, [candidates[index]])
    return resolve_reference(text, catalogue) if catalogue else CourseReference(NONE)


# ---------------------------------------------------------------------
# Evidence for the selected course(s)
# ---------------------------------------------------------------------

def _chunk_course_key(chunk: dict[str, Any]) -> str:
    return course_key({"course_id": chunk.get("course_id"), "course_name": chunk.get("course_name")})


def scope_evidence(evidence: dict[str, Any], courses: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """
    The evidence restricted to `courses` plus general passages (no course),
    renumbered [1], [2]... Returns (evidence, number of passages of the
    selected courses). Statuses other than "ok" pass through unchanged.
    """
    if evidence.get("status") != "ok":
        return evidence, 0
    wanted = {course_key(c) for c in courses}
    kept, selected = [], 0
    for chunk in evidence.get("chunks") or []:
        key = _chunk_course_key(chunk)
        if key and key not in wanted:
            continue
        selected += bool(key)
        kept.append({**chunk, "reference": len(kept) + 1})
    scoped = {
        **evidence,
        "chunks": kept,
        "courses": [c for c in evidence.get("courses") or [] if course_key(c) in wanted],
    }
    if not kept:
        scoped["status"] = "no_results"
    return scoped, selected


def merge_evidence(own: dict[str, Any], general: dict[str, Any]) -> dict[str, Any]:
    """The selected courses' passages, then the general passages of `general`, renumbered."""
    seen = {chunk.get("chunk_id") for chunk in own.get("chunks") or [] if chunk.get("chunk_id") is not None}
    chunks = list(own.get("chunks") or []) + [
        chunk for chunk in general.get("chunks") or []
        if not _chunk_course_key(chunk) and (chunk.get("chunk_id") is None or chunk.get("chunk_id") not in seen)
    ]
    return {**own, "status": "ok" if chunks else "no_results",
            "chunks": [{**chunk, "reference": n} for n, chunk in enumerate(chunks, start=1)]}


def course_filter(courses: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Chroma filter for the passages of these courses (by id or intake code)."""
    ids = [c["course_id"] for c in courses if c.get("course_id")]
    if not ids:
        return None
    # Ingestion stores a numeric id ("1010") as a number, and Chroma
    # compares types strictly: match both forms, one type per list.
    forms: list[list[Any]] = [ids]
    numbers = [int(i) for i in ids if i.isdigit()]
    if numbers:
        forms.append(numbers)
    return {"$or": [{field: {"$in": values}} for values in forms
                    for field in ("field_course_id", "field_course_code")]}


def courses_of_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The distinct courses of the sources an answer relies on."""
    return clean_courses([
        {"course_id": s.get("course_id"), "course_name": s.get("course_name")} for s in sources or []
    ])


# ---------------------------------------------------------------------
# Fixed bilingual clarification messages
# ---------------------------------------------------------------------

_CLARIFY = {
    "options": {
        "en": "Which course do you mean? These courses match your question:\n{options}\n\n"
              "Reply with the number or the course name.",
        "ar": "أي دورة تقصد؟ هذه الدورات تطابق سؤالك:\n{options}\n\nاكتب رقم الدورة أو اسمها.",
    },
    "which": {
        "en": "Which course are you asking about? Please tell me the course name or code. "
              "To see all courses, ask \"What courses are available?\"",
        "ar": "عن أي دورة تسأل؟ اكتب اسم الدورة أو رمزها، أو اسأل «ما هي الدورات المتاحة؟» لعرض كل الدورات.",
    },
}


def clarification_message(candidates: list[dict[str, Any]], language: Optional[str]) -> str:
    lang = "ar" if language == "ar" else "en"
    if not candidates:
        return _CLARIFY["which"][lang]
    options = "\n".join(f"{n}. {course_label(c)}" for n, c in enumerate(candidates, start=1))
    return _CLARIFY["options"][lang].format(options=options)
