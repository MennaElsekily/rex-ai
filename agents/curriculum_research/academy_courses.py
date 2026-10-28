"""
The academy's course catalogue, read from the existing knowledge-base
index (knowledge_base/vector_db, collection "rex_knowledge") -- the same
index REX searches. Nothing is copied, re-indexed or written.

Data-agnostic: a course is every chunk that carries ``field_course_id``
(the ingestion pipeline turns "Course ID: ..." facts into that field).
Its other fields come from the chunks' ``field_*`` metadata; lists such
as outcomes, topics or skills are parsed generically from the chunk
text ("Heading" followed by "- item" lines). A field that no chunk
states is reported in ``missing``; nothing is filled in.

``data_status`` comes from the data itself (the demonstration catalogue
says "Synthetic demonstration data"), so callers can label results.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from agents.curriculum_research.schemas import LEVELS
from agents.text_utils import normalize_for_matching

MAX_CHUNKS = 5000
_FIELDS = {
    "department": "field_department", "level": "field_level", "target_audience": "field_target_audience",
    "duration": "field_duration", "industry": "field_industry", "accreditation": "field_accreditation",
    "language": "field_language_of_instruction", "updated": "field_updated", "data_status": "field_data_status",
    "country": "field_country", "delivery_mode": "field_delivery_mode",
}
_HOURS = re.compile(r"(\d{1,4})\s*(?:hours?|hrs?|ساعة|ساعات)", re.IGNORECASE)
_COURSE_ID_IN_TEXT = re.compile(r"\b[A-Z]{2,5}-\d{2,4}\b")
_WORD = re.compile(r"[a-z][a-z0-9+#.-]*")
_STOP = frozenset("""
a an and are as at be by for from in into is it of on or the to with using your you our we this that
course courses curriculum introduction intro fundamentals basics advanced beginner intermediate level
program programme training skills skill learn learning practical applied essentials guide
""".split())


def _terms(text: str) -> set[str]:
    return {w.strip(".-") for w in _WORD.findall(str(text or "").casefold())
            if len(w.strip(".-")) > 2 and w.strip(".-") not in _STOP}


def parse_sections(text: str) -> dict[str, Any]:
    """
    Generic structure of one chunk: ``facts`` ("Key: value" lines) and
    ``lists`` (a heading line followed by "- item" lines).
    """
    facts: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    heading = None
    for raw in str(text or "").splitlines()[1:]:  # line 1 is the section path
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            if heading:
                lists.setdefault(heading, []).append(line[2:].strip())
            continue
        key, sep, value = line.partition(":")
        if sep and 0 < len(key) <= 40 and value.strip():
            facts[key.strip()] = value.strip()
            continue
        heading = line
    return {"facts": facts, "lists": lists}


def _merge_lists(target: dict[str, list[str]], new: dict[str, list[str]]) -> None:
    for heading, items in new.items():
        bucket = target.setdefault(heading, [])
        bucket.extend(i for i in items if i not in bucket)


def find_list(lists: dict[str, list[str]], *keywords: str) -> list[str]:
    """Items of the first heading containing one of the keywords (e.g. "outcome")."""
    for heading, items in lists.items():
        if any(k in heading.casefold() for k in keywords):
            return items
    return []


class AcademyCatalogue:
    """Read-only access to the courses in the existing index."""

    def __init__(
        self,
        vector_store=None,
        evidence_search: Optional[Callable[[str], dict[str, Any]]] = None,
        translate: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        vector_store: knowledge_base.vector_store.VectorStore (default: the shared index).
        evidence_search: question -> knowledge_base.evidence result (default:
            the existing EvidenceRetriever over the same index; used to find
            a course from a name in any language).
        translate: text -> English (default: knowledge_base.language).
        """
        self._vector_store = vector_store
        self._evidence_search = evidence_search
        self._translate = translate

    # --------------------------------------------------- shared resources

    @property
    def vector_store(self):
        if self._vector_store is None:
            from knowledge_base.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def evidence_search(self, question: str) -> dict[str, Any]:
        if self._evidence_search is None:
            from knowledge_base.evidence import EvidenceRetriever, safe_search
            from knowledge_base.retriever import Retriever
            retriever = EvidenceRetriever(Retriever(vector_store=self.vector_store, top_k=8))
            self._evidence_search = lambda q: safe_search(retriever, q)
        return self._evidence_search(question)

    def to_english(self, text: str) -> str:
        if not re.search(r"[؀-ۿ]", str(text or "")):
            return str(text or "")
        if self._translate is None:
            from knowledge_base.language import prepare_retrieval_query
            self._translate = lambda t: prepare_retrieval_query(t).query
        try:
            return self._translate(text)
        except Exception:
            return str(text or "")

    # ------------------------------------------------------------ courses

    def _course_chunks(self) -> dict[str, list[tuple[dict[str, Any], str]]]:
        data = self.vector_store.collection.get(include=["metadatas", "documents"], limit=MAX_CHUNKS)
        grouped: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        for metadata, text in zip(data.get("metadatas") or [], data.get("documents") or []):
            course_id = str((metadata or {}).get("field_course_id") or "").strip()
            if course_id:
                grouped[course_id].append((metadata, text or ""))
        return grouped

    @staticmethod
    def _record(course_id: str, chunks: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
        record: dict[str, Any] = {"course_id": course_id}
        for name, key in _FIELDS.items():
            record[name] = next((m[key] for m, _ in chunks if m.get(key) not in (None, "")), None)
        names = [m.get("field_course_name") for m, _ in chunks if m.get("field_course_name")]
        paths = [str(m.get("section_path") or "").split(" > ")[0] for m, _ in chunks if m.get("section_path")]
        record["name"] = names[0] if names else (paths[0] if paths else None)
        hours = _HOURS.search(str(record.get("duration") or ""))
        record["total_hours"] = int(hours.group(1)) if hours else None
        lists: dict[str, list[str]] = {}
        facts: dict[str, str] = {}
        for _, text in chunks:
            parsed = parse_sections(text)
            _merge_lists(lists, parsed["lists"])
            for key, value in parsed["facts"].items():
                facts.setdefault(key, value)
        record["learning_outcomes"] = find_list(lists, "outcome")
        record["topics"] = find_list(lists, "topic")
        record["skills"] = find_list(lists, "skill")
        prerequisite_text = " ".join(v for k, v in facts.items() if "prerequisite" in k.casefold())
        record["prerequisite_courses"] = sorted(set(_COURSE_ID_IN_TEXT.findall(prerequisite_text)) - {course_id})
        record["entry_requirements"] = next((v for k, v in facts.items() if "requirement" in k.casefold()), None)
        record["description"] = next((" ".join(items) for h, items in lists.items() if "description" in h.casefold()),
                                     None)
        record["source_files"] = sorted({m.get("filename") or m.get("source") for m, _ in chunks} - {None})
        record["missing"] = [k for k in ("name", "department", "level", "target_audience", "total_hours",
                                         "learning_outcomes", "topics") if not record.get(k)]
        record["_lists"], record["_facts"] = lists, facts
        return record

    def courses(
        self,
        department: Optional[str] = None,
        level: Optional[str] = None,
        audience_contains: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Every course in the index, optionally filtered by its metadata."""
        out = []
        for course_id, chunks in sorted(self._course_chunks().items()):
            record = self._record(course_id, chunks)
            if department and normalize_for_matching(department) not in normalize_for_matching(
                    f"{record.get('department') or ''} {course_id}"):
                continue
            if level and str(record.get("level") or "").casefold() != level.casefold():
                continue
            if audience_contains and audience_contains.casefold() not in str(record.get("target_audience") or "").casefold():
                continue
            out.append(record)
        return out[:limit]

    def course(self, reference: str) -> tuple[Optional[dict[str, Any]], str]:
        """
        One course by id, then by name, then by the existing semantic search
        (which also understands Arabic names). Returns (course, how_found).
        """
        reference = str(reference or "").strip()
        if not reference:
            return None, "none"
        grouped = self._course_chunks()
        for course_id, chunks in grouped.items():
            if course_id.casefold() == reference.casefold():
                return self._record(course_id, chunks), "id"
        wanted = normalize_for_matching(self.to_english(reference))
        records = [self._record(cid, chunks) for cid, chunks in grouped.items()]
        for record in records:
            name = normalize_for_matching(record.get("name") or "")
            if name and (name == wanted or (len(wanted) > 4 and wanted in name)):
                return record, "name"
        result = self.evidence_search(reference)
        for chunk in result.get("chunks") or []:
            for record in records:
                if record["name"] and str(chunk.get("section_path") or chunk.get("text") or "").startswith(record["name"]):
                    return record, "search"
        return None, "none"

    # ------------------------------------------------- analysis (Workflow B)

    def overlap(self, title: str, topics: list[str], outcomes: list[str], limit: int = 5) -> dict[str, Any]:
        """
        How much a proposed course overlaps existing ones. Deterministic:
        name similarity (difflib ratio) and how many of the proposal's
        terms an existing course already covers. Arabic text is translated
        to English first, like REX's retrieval. duplicate = name similarity
        >= 0.8, or term coverage >= 0.6 with name similarity >= 0.4.
        """
        english_title = self.to_english(title)
        proposal = _terms(" ".join([english_title, *map(self.to_english, topics or []), *map(self.to_english, outcomes or [])]))
        results = []
        for record in self.courses():
            existing = _terms(" ".join([record.get("name") or "", *record["topics"], *record["skills"],
                                        *record["learning_outcomes"]]))
            shared = proposal & existing
            coverage = len(shared) / len(proposal) if proposal else 0.0
            name_similarity = SequenceMatcher(None, normalize_for_matching(english_title),
                                              normalize_for_matching(record.get("name") or "")).ratio()
            results.append({
                "course_id": record["course_id"], "name": record.get("name"), "level": record.get("level"),
                "department": record.get("department"), "name_similarity": round(name_similarity, 2),
                "term_coverage": round(coverage, 2), "shared_terms": sorted(shared)[:15],
            })
        results.sort(key=lambda r: (r["name_similarity"] + r["term_coverage"]), reverse=True)
        duplicate = any(r["name_similarity"] >= 0.8 or (r["term_coverage"] >= 0.6 and r["name_similarity"] >= 0.4)
                        for r in results)
        return {"overlaps": results[:limit], "duplicate": duplicate, "compared_in": "English"}

    def gap_report(self, department: Optional[str] = None) -> dict[str, Any]:
        """
        Confirmed observations from the catalogue only (no market claims):
        levels each department lacks, prerequisites that are not in the
        catalogue, single-course departments, and existing courses that
        look like near-duplicates. Each gap has an id ("G1"...).
        """
        records = self.courses(department=department)
        by_department: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_department[record.get("department") or "Unknown"].append(record)
        known_ids = {r["course_id"] for r in self.courses()}
        gaps: list[dict[str, Any]] = []

        def add(kind: str, dept: str, detail: str, evidence: list[str], **extra) -> None:
            gaps.append({"id": f"G{len(gaps) + 1}", "type": kind, "department": dept, "detail": detail,
                         "evidence_course_ids": evidence, "basis": "catalogue data", **extra})

        departments = []
        for dept, courses in sorted(by_department.items()):
            levels = [lvl for lvl in LEVELS if any(c.get("level") == lvl for c in courses)]
            departments.append({
                "department": dept, "course_ids": [c["course_id"] for c in courses], "levels": levels,
                "topics": sorted({t for c in courses for t in c["topics"]}),
                "skills": sorted({s for c in courses for s in c["skills"]}),
            })
            for lvl in LEVELS:
                if lvl not in levels:
                    add("missing_level", dept, f"No {lvl} course in {dept}.", [c["course_id"] for c in courses],
                        level=lvl)
            if len(courses) == 1:
                add("single_course_department", dept, f"{dept} has only one course.", [courses[0]["course_id"]])
            for course in courses:
                for prerequisite in course["prerequisite_courses"]:
                    if prerequisite not in known_ids:
                        add("missing_prerequisite", dept, f"{course['course_id']} needs {prerequisite}, which is "
                                                          "not in the catalogue.", [course["course_id"]])
        for index, first in enumerate(records):
            for second in records[index + 1:]:
                similarity = SequenceMatcher(None, (first.get("name") or "").casefold(),
                                             (second.get("name") or "").casefold()).ratio()
                if similarity >= 0.75:
                    add("possible_overlap", first.get("department") or "Unknown",
                        f"{first['course_id']} and {second['course_id']} have very similar names.",
                        [first["course_id"], second["course_id"]])
        statuses = sorted({r.get("data_status") for r in records if r.get("data_status")})
        return {"departments": departments, "gaps": gaps, "course_count": len(records), "data_status": statuses}
