"""
Verified course records, read from the indexed chunks (read-only).

A recommendation may only name a course that the catalogue itself
identifies: a record needs BOTH an identifier and a name stated in the
indexed documents. Every other field is copied as written, or left None
when the catalogue does not state it -- nothing is inferred or filled in.

Two catalogue layouts are recognised:

1. Ingested record fields. The pipeline turns a "Course ID: DA-101"
   line into ``field_course_id`` metadata on the course's chunks, with
   the other "Key: value" lines (Level, Delivery mode, Fee (EGP), ...)
   as ``field_*`` metadata.
2. Programme records in running text, as in catalogues that head each
   programme with "PROGRAMME 01 · DA-101" and state its facts in a
   "Key: value | Key: value" record ("Programme: Excel for Business
   Analysis | Skill level: Beginner") and in titled sections ("Entry
   requirements", "Intended audience", "24 guided learning hours"). A
   programme's chunks are the ones from its heading up to the next
   programme heading of the same document, in chunk order.

course_map() also returns which course each chunk belongs to, so a
retrieved passage can be labelled with its course instead of being read
as free text (a lone "Entry requirements: Basic computer skills."
passage is not a course).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Optional

_FIELD_PREFIX = "field_"

# "PROGRAMME 01 · DA-101", "Course 3 - AI-201", "PROGRAM: SW-101"
_PROGRAMME_HEADING = re.compile(
    r"^\s*(?:PROGRAMME|PROGRAM|COURSE)\s*(?:\d{1,3})?\s*[·•:|\-–—]\s*([A-Z]{2,6}(?:-[A-Z0-9]{1,8}){1,3})\s*$",
    re.IGNORECASE,
)
# "Course ID: DA-101" written in the text of an index without record fields.
_ID_LINE = re.compile(r"^\s*(?:course|programme|program)\s*(?:id|code)\s*[:：]\s*([A-Z]{2,6}(?:-[A-Z0-9]{1,8}){1,3})\s*$",
                      re.IGNORECASE)
_COURSE_ID = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{1,8}){1,3}\b")
_HOURS_TITLE = re.compile(r"^\s*(\d{1,4})\s+(?:guided\s+|contact\s+|training\s+|learning\s+|total\s+)*hours?\b",
                          re.IGNORECASE)
_HOURS = re.compile(r"(\d{1,4})\s*(?:hours?|hrs?)\b", re.IGNORECASE)
_KEY_VALUE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ()/&_-]{0,40}?)\s*[:：]\s*(\S.*?)\s*$")

_NAME_KEYS = ("course_name", "programme", "program", "programme_name", "program_name", "programme_title",
              "course_title", "course", "name", "title")
_LEVEL_KEYS = ("level", "skill_level", "course_level")
_DELIVERY_KEYS = ("delivery_mode", "delivery", "delivery_format", "mode_of_delivery", "format")
_DEPARTMENT_KEYS = ("department",)
_AUDIENCE_KEYS = ("target_audience", "intended_audience", "audience")
_DURATION_KEYS = ("duration", "total_hours", "hours")
_ENTRY_KEYS = ("entry_requirements", "entry_requirement", "admission_requirements")
_STATUS_KEYS = ("data_status", "source_status")
# Section titles whose text is a field ("Entry requirements\n\nBasic computer literacy.").
_SECTION_FIELDS = {
    "entry requirements": "entry_requirements", "entry requirement": "entry_requirements",
    "intended audience": "target_audience", "target audience": "target_audience",
    "programme summary": "summary", "program summary": "summary", "course summary": "summary",
    "course description": "summary",
}


def _key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).casefold()).strip("_")


def _clean(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" ;|")   # as written (a final full stop is kept)
    return text or None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value or ""))
    return float(match.group(0).replace(",", "")) if match else None


def _facts_in_text(text: str) -> dict[str, str]:
    """{"skill_level": "Foundation", ...} from "Key: value" lines and "A: x | B: y" records."""
    facts: dict[str, str] = {}
    for line in str(text or "").splitlines():
        for segment in line.split(" | ") if " | " in line else [line]:
            match = _KEY_VALUE.match(segment)
            if match:
                value = _clean(match.group(2))
                if value:
                    facts.setdefault(_key(match.group(1)), value)
    return facts


def _title_and_body(text: str) -> tuple[str, str]:
    lines = [line for line in str(text or "").splitlines()]
    title = lines[0].strip() if lines else ""
    body = " ".join(line.strip() for line in lines[1:] if line.strip())
    return title, body


def _first(facts: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = _clean(facts.get(key))
        if value:
            return value
    return None


def _fee(facts: dict[str, Any]) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """(amount, currency, as written) of the first fee-like fact, e.g. "fee_egp": 3200."""
    for key, value in facts.items():
        if re.search(r"(?:^|_)(?:fee|fees|price|tuition|cost)(?:_|$)", key):
            amount = _number(value)
            if amount is None:
                continue
            currency = next((c.upper() for c in ("egp", "usd", "sar", "eur", "aed")
                             if c in key or c in str(value).casefold()), None)
            return amount, currency, _clean(value)
    return None, None, None


def _document_key(metadata: dict[str, Any]) -> str:
    return str(metadata.get("document_id") or metadata.get("source") or metadata.get("filename") or "")


def _file_name(metadata: dict[str, Any]) -> Optional[str]:
    source = metadata.get("filename") or metadata.get("source") or metadata.get("document_id")
    return re.split(r"[\\/]", str(source))[-1] if source else None


def _order(metadata: dict[str, Any]) -> tuple[float, float]:
    def as_number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return as_number(metadata.get("chunk_id")), as_number(metadata.get("page"))


def chunk_key(metadata: dict[str, Any]) -> tuple[str, str]:
    """The identity of one indexed chunk: (document, chunk id)."""
    return _document_key(metadata), str(metadata.get("chunk_id"))


class _Builder:
    def __init__(self, course_id: str) -> None:
        self.course_id = course_id
        self.facts: dict[str, Any] = {}
        self.sections: dict[str, str] = {}
        self.hours: Optional[int] = None
        self.files: list[str] = []
        self.pages: set[int] = set()
        self.section_name: Optional[str] = None

    def add(self, metadata: dict[str, Any], text: str) -> None:
        for key, value in metadata.items():
            if key.startswith(_FIELD_PREFIX) and value not in (None, ""):
                self.facts.setdefault(key[len(_FIELD_PREFIX):], value)
        for key, value in _facts_in_text(text).items():
            self.facts.setdefault(key, value)
        title, body = _title_and_body(text)
        section = _SECTION_FIELDS.get(title.casefold().rstrip(":"))
        if section and body:
            self.sections.setdefault(section, body)
        hours = _HOURS_TITLE.match(title)
        if hours and self.hours is None:
            self.hours = int(hours.group(1))
        if not self.section_name and metadata.get("section_path"):
            self.section_name = str(metadata["section_path"]).split(" > ")[0].strip() or None
        name = _file_name(metadata)
        if name and name not in self.files:
            self.files.append(name)
        page = metadata.get("page")
        if isinstance(page, (int, float)) and not isinstance(page, bool):
            self.pages.add(int(page))

    def record(self) -> Optional[dict[str, Any]]:
        facts = self.facts
        name = _first(facts, _NAME_KEYS) or self.section_name
        if name:
            name = re.sub(r"^\d{1,2}\.\s+", "", name)   # "1. Excel for Business Analysis"
        if not (self.course_id and name):
            return None   # unverifiable: no course without both an id and a name
        duration = _first(facts, _DURATION_KEYS)
        hours = self.hours
        if hours is None and duration:
            match = _HOURS.search(duration)
            hours = int(match.group(1)) if match else None
        fee, currency, fee_text = _fee(facts)
        prerequisite_text = _first(facts, ("prerequisite_courses", "prerequisites", "prerequisite"))
        entry = _first(facts, _ENTRY_KEYS) or self.sections.get("entry_requirements")
        return {
            "course_id": self.course_id,
            "course_name": name,
            "department": _first(facts, _DEPARTMENT_KEYS),
            "level": _first(facts, _LEVEL_KEYS),
            "total_hours": hours,
            "duration": duration,
            "delivery_mode": _first(facts, _DELIVERY_KEYS),
            "fee": fee,
            "fee_currency": currency,
            "fee_text": fee_text,
            "entry_requirements": entry,
            # Only course ids the catalogue itself lists as prerequisites.
            "prerequisite_courses": sorted(set(_COURSE_ID.findall(prerequisite_text or "")) - {self.course_id}),
            "prerequisites_text": prerequisite_text,
            "target_audience": _first(facts, _AUDIENCE_KEYS) or self.sections.get("target_audience"),
            "summary": self.sections.get("summary") or _first(facts, ("description", "course_description")),
            "data_status": _first(facts, _STATUS_KEYS),
            "source_files": list(self.files),
            "pages": sorted(self.pages),
        }


def course_map(metadatas: list[dict[str, Any]], documents: list[str]) -> tuple[list[dict[str, Any]],
                                                                               dict[tuple[str, str], dict[str, str]]]:
    """
    (verified course records sorted by course id, {chunk_key: {course_id, course_name}})
    for one index's chunks. A course stated in both layouts is merged by id.
    """
    builders: dict[str, _Builder] = {}
    members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_document: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)

    for metadata, text in zip(metadatas or [], documents or []):
        metadata = metadata or {}
        course_id = _clean(metadata.get("field_course_id") or metadata.get("field_course_code"))
        if course_id:   # layout 1: the ingested record fields
            builders.setdefault(course_id, _Builder(course_id)).add(metadata, text or "")
            members[course_id].append(chunk_key(metadata))
        else:
            by_document[_document_key(metadata)].append((metadata, text or ""))

    for chunks in by_document.values():   # layout 2: programme records in the text
        current: Optional[str] = None
        for metadata, text in sorted(chunks, key=lambda item: _order(item[0])):
            for line in text.splitlines():
                heading = _PROGRAMME_HEADING.match(line) or _ID_LINE.match(line)
                if heading:
                    current = heading.group(1).upper()
                    break
            if current:
                builders.setdefault(current, _Builder(current)).add(metadata, text)
                members[current].append(chunk_key(metadata))

    records, owner = [], {}
    for course_id in sorted(builders):
        record = builders[course_id].record()
        if record is None:
            continue
        records.append(record)
        for key in members[course_id]:
            owner[key] = {"course_id": record["course_id"], "course_name": record["course_name"]}
    return records, owner


# ---------------------------------------------------------------------
# Catalogue listing filters (used by EvidenceRetriever.course_records)
# ---------------------------------------------------------------------

_LEVEL_WORDS = (
    (1, ("foundation", "beginner", "beginners", "introductory", "entry")),
    (2, ("intermediate",)),
    (3, ("advanced", "expert")),
)
# Words of a listing request that name no topic ("What courses does the academy offer?").
_LISTING_WORDS = frozenset("""
what which who whom whose where when how do does did is are was were be been have has had can could would should
will may might i me my we us our you your they them their it its the a an and or of for in on at to from with by
about any all every each some there here this that these those show list give tell see find get want need please
course courses class classes program programs programme programmes training trainings track tracks catalogue catalog
offer offers offered offering provide provides provided available availability academy academies option options kind
kinds type types area areas field fields related currently also other more new study learn taking take enrol enroll
level levels beginner beginners foundation introductory entry intermediate advanced expert
""".split())
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]*")


def level_rank(text: Any) -> Optional[int]:
    """1 beginner/foundation, 2 intermediate, 3 advanced; "Foundation to Intermediate" -> 1 (its entry level)."""
    folded = str(text or "").casefold()
    found = sorted((folded.find(w), rank) for rank, words in _LEVEL_WORDS for w in words
                   if re.search(rf"\b{w}\b", folded))
    if not found:
        return None
    if len({rank for _, rank in found}) > 1 and re.search(r"\bto\b|-", folded):
        return found[0][1]          # a range starts at its first level
    return max(rank for _, rank in found)


def topic_terms(english_query: str) -> list[str]:
    """The topic words of a listing request: 'What Data Analytics courses ...' -> ['data', 'analytics']."""
    terms = []
    for word in _TERM.findall(english_query or ""):
        folded = word.casefold().strip(".-")
        acronym = word.isupper() and 2 <= len(word) <= 4          # "IT", "AI", "SQL"
        if (acronym or (len(folded) > 2 and folded not in _LISTING_WORDS)) and folded not in terms:
            terms.append(folded)
    return terms


# Where a topic is looked for, most specific first: a department that names
# the topic ("Data Analytics") wins over a course that only mentions it.
TOPIC_FIELDS = (("department",), ("department", "course_name"),
                ("department", "course_name", "summary", "target_audience"))


def matches_topic(record: dict[str, Any], terms: list[str],
                  fields: tuple[str, ...] = TOPIC_FIELDS[-1]) -> bool:
    """Every topic term appears (as a word or word start) in these fields of the record."""
    text = " ".join(str(record.get(k) or "") for k in fields)
    words = [w.casefold() for w in _TERM.findall(text)]
    return bool(terms) and all(any(w == t or (len(t) > 3 and w.startswith(t[:5])) for w in words) for t in terms)
