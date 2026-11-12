"""
Shared evidence retrieval for the RAG knowledge base.

EvidenceRetriever.search() answers "which document passages are relevant
to this question?" and nothing more: it never writes a user-facing
answer. Any agent can use it (REX Assistant does, through the MCP tool
search_knowledge_base in agents/rag_mcp_server.py).

It only composes existing components:
    knowledge_base.language.prepare_retrieval_query  -> English query for
        the English-only embedding model (translation status reported)
    knowledge_base.retriever.Retriever               -> embedding, Chroma
        search, cross-encoder reranking, relevance threshold

The original question is returned unchanged next to the query actually
used for retrieval. Source references carry the file name only, never
absolute filesystem paths.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any

from knowledge_base.course_records import TOPIC_FIELDS, chunk_key, course_map, level_rank, matches_topic, topic_terms
from knowledge_base.language import prepare_retrieval_query
from knowledge_base.retriever import Retriever

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_EMPTY_INDEX = "empty_index"
STATUS_INVALID_REQUEST = "invalid_request"
STATUS_ERROR = "error"
# An Arabic (or mixed) query whose translation failed: the English-only
# index cannot be searched reliably with it, so nothing is retrieved.
STATUS_TRANSLATION_FAILED = "translation_failed"
# The index holds documents that are no longer in the documents folder
# (e.g. an obsolete index restored by a git branch switch): it is not used.
STATUS_STALE_INDEX = "stale_index"
MAX_TOP_K = 20

_PATH_SEPARATORS = re.compile(r"[\\/]")
_FIELD_PREFIX = "field_"

# Chroma "where" operators a caller may use in metadata_filter.
_FILTER_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"})
_FILTER_LOGIC = frozenset({"$and", "$or"})
_SCALAR = (str, int, float, bool)


def _check_filter(metadata_filter: Any, depth: int = 0) -> None:
    """
    A malformed filter is the caller's mistake (status invalid_request),
    not a retrieval failure. Accepts {"field": value},
    {"field": {"$op": value}} and {"$and"/"$or": [filter, ...]}.
    """
    if not isinstance(metadata_filter, dict) or not metadata_filter or depth > 4:
        raise ValueError("metadata_filter must be a non-empty dictionary")
    for key, value in metadata_filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata_filter keys must be non-empty strings")
        if key in _FILTER_LOGIC:
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError(f"{key} needs a list of at least two filters")
            for part in value:
                _check_filter(part, depth + 1)
        elif key.startswith("$"):
            raise ValueError(f"Unknown filter operator {key}")
        elif isinstance(value, dict):
            if len(value) != 1 or next(iter(value)) not in _FILTER_OPERATORS:
                raise ValueError(f"The filter on {key} must use one of {sorted(_FILTER_OPERATORS)}")
            (operator, operand), = value.items()
            if operator in ("$in", "$nin"):
                if not isinstance(operand, list) or not operand or not all(isinstance(v, _SCALAR) for v in operand):
                    raise ValueError(f"{operator} on {key} needs a non-empty list of values")
            elif not isinstance(operand, _SCALAR):
                raise ValueError(f"The filter value for {key} must be a string, number or boolean")
        elif not isinstance(value, _SCALAR):
            raise ValueError(f"The filter value for {key} must be a string, number or boolean")


def _course_of(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    The course a chunk belongs to, when its metadata says so: the course
    id (or code) and name as indexed. A course's catalogue section has its
    name as the first part of the section path. Nothing is guessed: a
    chunk without a course id or name has none.
    """
    course_id = metadata.get("field_course_id") or metadata.get("field_course_code")
    course_name = metadata.get("field_course_name")
    if course_id and not course_name:
        first = str(metadata.get("section_path") or "").split(" > ")[0].strip()
        course_name = first or None
    return (str(course_id).strip() or None) if course_id else None, (str(course_name).strip() or None) if course_name else None


def field_course_filter(course_ids: list[str]) -> dict[str, Any] | None:
    """Chroma filter for the passages whose record fields name these courses (by id or intake code)."""
    ids = [c for c in course_ids if c]
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


def _id_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value or "").casefold())


def _course_members(metadatas: list[dict[str, Any]],
                    owners: dict[tuple[str, str], dict[str, str]]) -> dict[str, dict[str, Any]]:
    """{course_id: {"field": has record fields, "chunks": {(metadata key, document): {chunk ids}}}}."""
    members: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        metadata = metadata or {}
        field_id, _ = _course_of(metadata)
        if field_id:
            members.setdefault(field_id, {"field": True, "chunks": {}})
            continue
        owner = owners.get(chunk_key(metadata))
        if not owner or metadata.get("chunk_id") is None:
            continue
        key = next((k for k in ("document_id", "source", "filename") if metadata.get(k)), None)
        if key:
            info = members.setdefault(owner["course_id"], {"field": False, "chunks": {}})
            info["chunks"].setdefault((key, metadata[key]), set()).add(int(metadata["chunk_id"]))
    return members


def _file_name(source: Any) -> str | None:
    """'E:\\...\\catalogue.pdf' -> 'catalogue.pdf' (no internal paths)."""
    if not source:
        return None
    return _PATH_SEPARATORS.split(str(source))[-1] or None


def _to_chunk(reference: int, result: dict[str, Any],
              owners: dict[tuple[str, str], dict[str, str]] | None = None) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    score = result.get("relevance_score")
    course_id, course_name = _course_of(metadata)
    if not (course_id or course_name) and owners:
        # A passage of a programme record whose chunks carry no course
        # fields ("Entry requirements: Basic computer literacy." on the
        # programme's page): labelled with the programme it belongs to.
        owner = owners.get(chunk_key(metadata)) or {}
        course_id, course_name = owner.get("course_id"), owner.get("course_name")

    return {
        "reference": reference,
        "text": result.get("text", "").strip(),
        # The course this passage is about, when the metadata says so.
        "course_id": course_id,
        "course_name": course_name,
        "source": _file_name(result.get("source") or metadata.get("source")),
        "page": result.get("page", metadata.get("page")),
        "page_end": metadata.get("page_end"),
        "section": result.get("section", metadata.get("section")),
        "section_path": metadata.get("section_path"),
        "record_id": metadata.get("record_id"),
        "chunk_id": result.get("chunk_id", metadata.get("chunk_id")),
        # Reranker relevance (higher is better) and the vector distance of
        # the similarity search (cosine; lower is closer).
        "relevance_score": round(float(score), 4) if score is not None else None,
        "distance": round(float(result["distance"]), 4) if result.get("distance") is not None else None,
        # Structured fields written in the source for this record or
        # section (e.g. {"level": "Beginner"}), exactly as indexed.
        "fields": {
            key[len(_FIELD_PREFIX):]: value
            for key, value in metadata.items()
            if key.startswith(_FIELD_PREFIX)
        },
    }


class EvidenceRetriever:
    """Retrieve relevant document passages and their source metadata."""

    def __init__(self, retriever: Retriever | None = None, documents_dir: str | Path | None = None) -> None:
        """
        documents_dir: the authoritative documents folder. When given, an
        index holding any document that is no longer in it is refused
        (status "stale_index") instead of serving an obsolete catalogue.
        """
        self.retriever = retriever or Retriever(top_k=5)
        self.documents_dir = Path(documents_dir) if documents_dir else None
        self._catalogue: tuple[Any, list[dict[str, Any]], dict] | None = None
        self._members: dict[str, dict[str, Any]] = {}
        self._stale: tuple[Any, list[str]] | None = None

    def stale_sources(self) -> list[str]:
        """
        Indexed source files missing from the documents folder (empty when
        the index matches it, or when no documents folder was given).
        Rechecked whenever the index changes.
        """
        if self.documents_dir is None:
            return []
        store = self.retriever.vector_store
        stamp = (getattr(store, "_index_stamp", None), store.collection.count())
        if self._stale is None or self._stale[0] != stamp:
            present = {p.name for p in self.documents_dir.rglob("*") if p.is_file()}
            metadatas = store.collection.get(include=["metadatas"]).get("metadatas") or []
            indexed = {_file_name((m or {}).get("filename") or (m or {}).get("source")) for m in metadatas} - {None}
            self._stale = (stamp, sorted(indexed - present))
        return self._stale[1]

    def _stale_error(self) -> str | None:
        stale = self.stale_sources()
        if not stale:
            return None
        logger.error("The knowledge-base index holds documents missing from %s: %s", self.documents_dir, stale)
        return (f"The knowledge-base index holds {len(stale)} document(s) that are not in the documents folder "
                f"(e.g. {stale[0]}). Rebuild it: python -m knowledge_base.pipeline")

    def _course_catalogue(self) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, str]]]:
        """
        (verified course records, chunk -> course) for the current index;
        rebuilt when the index changes. See knowledge_base/course_records.py.
        """
        store = self.retriever.vector_store
        stamp = (getattr(store, "_index_stamp", None), store.collection.count())
        if self._catalogue is None or self._catalogue[0] != stamp:
            data = store.collection.get(include=["metadatas", "documents"])
            metadatas = data.get("metadatas") if isinstance(data.get("metadatas"), list) else []
            documents = data.get("documents") if isinstance(data.get("documents"), list) else []
            records, owners = course_map(metadatas, documents)
            self._catalogue = (stamp, records, owners)
            self._members = _course_members(metadatas, owners)
        return self._catalogue[1], self._catalogue[2]

    def course_filter(self, course_ids: list[str]) -> dict[str, Any] | None:
        """
        A Chroma "where" filter selecting the chunks of these courses, or
        None when none of them is in the index. Courses with record fields
        are selected by field_course_id; a programme record in running text
        by its document and chunk ids.
        """
        self._course_catalogue()
        wanted = {_id_key(c) for c in course_ids if c}
        clauses: list[dict[str, Any]] = []
        field_ids = sorted({cid for cid, info in self._members.items() if _id_key(cid) in wanted and info["field"]})
        if field_ids:
            clauses.append(field_course_filter(field_ids))
        for cid, info in sorted(self._members.items()):
            if _id_key(cid) in wanted and not info["field"]:
                for (key, value), chunk_ids in sorted(info["chunks"].items()):
                    clauses.append({"$and": [{key: value}, {"chunk_id": {"$in": sorted(chunk_ids)}}]})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$or": clauses}

    def _chunk_owners(self, retrieved: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
        """The chunk -> course map, only needed when a passage has no course fields."""
        if all(_course_of(item.get("metadata") or {}) != (None, None) for item in retrieved):
            return {}
        try:
            return self._course_catalogue()[1]
        except Exception:
            logger.warning("Course records are unavailable; passages stay unlabelled", exc_info=True)
            return {}

    def search(
        self,
        question: str,
        retrieval_query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        course_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            question: The user's original question (any language). It is
                returned unchanged and is never replaced by a translation.
            retrieval_query: Optional search text prepared by the caller
                (e.g. built from a learner profile). Translated to English
                like the question would be, when needed.
            metadata_filter: Optional Chroma "where" filter over indexed
                metadata fields.
            course_ids: Optional course ids: only these courses' passages
                are searched (the filter is built here, for either catalogue
                layout; see course_filter()).
            top_k: Optional number of passages (1-20; default 5).

        Order of operations: detect the query language; translate Arabic
        or mixed text to English (Latin names, codes and numbers kept);
        embed the English text with the index's embedding model; similarity
        search in ChromaDB; rerank; label each passage with its course.

        Returns a dict with: status (ok / no_results / empty_index /
        translation_failed), question (the original, unchanged),
        retrieval_query and embedded_text (the text embedded),
        query_language, translation (not_needed / translated / failed),
        translation_error, embedding_model, embedding_dimensions,
        collection, top_k, course_ids, chunks (reference, text, course_id,
        course_name, source file name, page, section, chunk_id,
        relevance_score, distance, fields), courses (the distinct courses
        the chunks belong to, in relevance order), limitations and error.

        Raises ValueError for invalid arguments; retrieval failures raise
        too -- use safe_search() where a failure must become a result.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        if retrieval_query is not None and (
            not isinstance(retrieval_query, str) or not retrieval_query.strip()
        ):
            raise ValueError(
                "retrieval_query must be a non-empty string when provided"
            )

        if metadata_filter is not None:
            _check_filter(metadata_filter)
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K):
            raise ValueError(f"top_k must be a whole number from 1 to {MAX_TOP_K}")
        if course_ids is not None and (not isinstance(course_ids, list)
                                       or not all(isinstance(c, str) and c.strip() for c in course_ids)):
            raise ValueError("course_ids must be a list of course ids")

        original_question = question.strip()
        prepared = prepare_retrieval_query(
            original_question if retrieval_query is None else retrieval_query.strip()
        )

        result: dict[str, Any] = {
            "status": STATUS_OK,
            "question": original_question,
            "retrieval_query": prepared.query,
            "query_language": prepared.language,
            "translation": prepared.translation,
            "translation_error": prepared.error,
            "embedded_text": None,
            "embedding_model": None,
            "embedding_dimensions": None,
            "collection": getattr(self.retriever.vector_store, "collection_name", None),
            "top_k": top_k or getattr(self.retriever, "top_k", None),
            "course_ids": list(course_ids) if course_ids else None,
            "chunks": [],
            "courses": [],
            "limitations": [],
            "error": None,
        }

        if prepared.translation == "failed":
            # Searching the English-only index with untranslated Arabic
            # would return unrelated passages: nothing is retrieved.
            result["status"] = STATUS_TRANSLATION_FAILED
            result["error"] = "The question could not be translated for the English catalogue search."
            return result

        if self.retriever.vector_store.collection.count() == 0:
            result["status"] = STATUS_EMPTY_INDEX
            result["error"] = "The knowledge-base index is empty or missing."
            return result
        stale = self._stale_error()
        if stale:
            result["status"], result["error"] = STATUS_STALE_INDEX, stale
            return result

        if course_ids:
            scope = self.course_filter(course_ids)
            if scope is None:
                result["status"] = STATUS_NO_RESULTS
                result["error"] = "None of these courses is in the catalogue."
                return result
            metadata_filter = scope if metadata_filter is None else {"$and": [metadata_filter, scope]}

        retriever = self.retriever
        if top_k and top_k != getattr(retriever, "top_k", top_k):
            retriever = copy.copy(retriever)      # same store and models, another passage count
            retriever.top_k = top_k
        retrieved = retriever.retrieve(
            query=original_question,
            retrieval_query=prepared.query,
            metadata_filter=metadata_filter,
        )
        embedded = getattr(retriever, "last_query", None)
        if isinstance(embedded, dict):
            result.update({key: embedded.get(key) for key in
                           ("embedded_text", "embedding_model", "embedding_dimensions")})
        result["limitations"].append(
            f"The {result['top_k']} most relevant passages only: not a complete list of the catalogue's courses.")

        owners = self._chunk_owners(retrieved)
        result["chunks"] = [
            chunk
            for chunk in (
                _to_chunk(reference, item, owners)
                for reference, item in enumerate(retrieved, start=1)
            )
            if chunk["text"]
        ]

        # The distinct courses the evidence is about, in relevance order --
        # lets the agent notice a question that matches several courses.
        seen: set[tuple[str | None, str | None]] = set()
        for chunk in result["chunks"]:
            key = (chunk["course_id"], chunk["course_name"])
            if (chunk["course_id"] or chunk["course_name"]) and key not in seen:
                seen.add(key)
                result["courses"].append({"course_id": chunk["course_id"], "course_name": chunk["course_name"]})

        if not result["chunks"]:
            result["status"] = STATUS_NO_RESULTS

        return result

    def list_courses(self) -> dict[str, Any]:
        """
        The distinct courses named in the indexed metadata, as
        {"status", "courses": [{course_id, course_name}], "error"}, sorted
        by course id. Read-only: the index is only read, never changed.
        An agent uses it to tell which course a question refers to.
        """
        collection = self.retriever.vector_store.collection
        if collection.count() == 0:
            return {"status": STATUS_EMPTY_INDEX, "courses": [],
                    "error": "The knowledge-base index is empty or missing."}
        stale = self._stale_error()
        if stale:
            return {"status": STATUS_STALE_INDEX, "courses": [], "error": stale}

        by_key: dict[str, dict[str, Any]] = {}
        for metadata in collection.get(include=["metadatas"]).get("metadatas") or []:
            course_id, course_name = _course_of(metadata or {})
            key = course_id or course_name
            if not key:
                continue
            known = by_key.setdefault(key, {"course_id": course_id, "course_name": course_name})
            # The indexed course name wins over a name taken from a section path.
            if (metadata or {}).get("field_course_name"):
                known["course_name"] = course_name

        # Programme records stated in running text carry no course fields.
        try:
            records, _ = self._course_catalogue()
        except Exception:
            logger.warning("Course records are unavailable", exc_info=True)
            records = []
        for record in records:
            by_key.setdefault(record["course_id"], {"course_id": record["course_id"],
                                                     "course_name": record["course_name"]})

        courses = sorted(by_key.values(), key=lambda c: (c["course_id"] or "", c["course_name"] or ""))
        return {"status": STATUS_OK, "courses": courses, "error": None}


    def course_records(self, query: str | None = None, level: str | None = None) -> dict[str, Any]:
        """
        The catalogue's verified course records -- the COMPLETE list, not a
        top-k search -- optionally narrowed to a topic and/or level.

        A record is a course the indexed documents identify by BOTH an id
        and a name, with only the facts they state (level, hours, delivery
        mode, fee, entry requirements, prerequisite course ids, audience,
        source files and pages); a fact not stated is None.

        query: the learner's listing request in any language. It is
            translated like a search query; its topic words are matched
            against each record's department, name and description. A topic
            no record names is looked up by similarity search, and the
            courses whose passages rank are returned. A request with no
            topic word lists every course.
        level: "beginner" / "intermediate" / "advanced" (or found in the
            query): only courses whose entry level is that level.

        Returns {status, records, total_in_catalogue, complete (True when
        no filter narrowed the list), filter {query, query_language,
        translation, english_query, topic_terms, level}, match_method
        (all / topic / similarity / none), collection, limitations, error}.
        Read-only.
        """
        base: dict[str, Any] = {
            "records": [], "total_in_catalogue": 0, "complete": False,
            "filter": {"query": query, "query_language": None, "translation": None, "english_query": None,
                       "topic_terms": [], "level": level},
            "match_method": None, "collection": getattr(self.retriever.vector_store, "collection_name", None),
            "limitations": [], "error": None,
        }
        if self.retriever.vector_store.collection.count() == 0:
            return {**base, "status": STATUS_EMPTY_INDEX, "error": "The knowledge-base index is empty or missing."}
        stale = self._stale_error()
        if stale:
            return {**base, "status": STATUS_STALE_INDEX, "error": stale}
        records, _ = self._course_catalogue()
        base["total_in_catalogue"] = len(records)

        terms: list[str] = []
        if query and query.strip():
            prepared = prepare_retrieval_query(query.strip())
            base["filter"].update({"query_language": prepared.language, "translation": prepared.translation,
                                   "english_query": prepared.query})
            if prepared.translation == "failed":
                return {**base, "status": STATUS_TRANSLATION_FAILED,
                        "error": "The request could not be translated for the English catalogue."}
            terms = topic_terms(prepared.query)
            base["filter"]["topic_terms"] = terms
            if not level and level_rank(prepared.query):
                level = {1: "beginner", 2: "intermediate", 3: "advanced"}[level_rank(prepared.query)]
        wanted_level = level_rank(level) if level else None
        base["filter"]["level"] = level

        selected, method = list(records), "all"
        if terms:
            selected, method = [], "topic"
            for fields in TOPIC_FIELDS:         # the department first, then names, then descriptions
                selected = [r for r in records if matches_topic(r, terms, fields)]
                if selected:
                    break
            if not selected:
                selected = self._courses_found_by_search(base["filter"]["english_query"], records)
                method = "similarity"
                base["limitations"].append("No course names this topic; the courses whose catalogue passages "
                                           "the similarity search ranked are listed.")
        if wanted_level:
            selected = [r for r in selected if level_rank(r.get("level")) == wanted_level]
        base["complete"] = not terms and not wanted_level
        base["match_method"] = method if selected else "none"
        return {**base, "status": STATUS_OK, "records": [dict(r) for r in selected]}

    def _courses_found_by_search(self, english_query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The courses whose passages a similarity search for the topic ranks (10 passages)."""
        retriever = copy.copy(self.retriever)
        retriever.top_k = 10
        owners = self._course_catalogue()[1]
        found: list[str] = []
        for item in retriever.retrieve(query=english_query, retrieval_query=english_query):
            metadata = item.get("metadata") or {}
            course_id, _ = _course_of(metadata)
            course_id = course_id or (owners.get(chunk_key(metadata)) or {}).get("course_id")
            if course_id and course_id not in found:
                found.append(course_id)
        by_id = {r["course_id"]: r for r in records}
        return [by_id[c] for c in found if c in by_id]


def safe_course_records(evidence_retriever: EvidenceRetriever, query: str | None = None,
                        level: str | None = None) -> dict[str, Any]:
    """EvidenceRetriever.course_records() that never raises (status "error")."""
    try:
        return evidence_retriever.course_records(query=query, level=level)
    except Exception:
        logger.exception("Reading the course records failed")
        return {"status": STATUS_ERROR, "records": [], "error": "The course records are unavailable."}


def safe_list_courses(evidence_retriever: EvidenceRetriever) -> dict[str, Any]:
    """EvidenceRetriever.list_courses() that never raises (status "error")."""
    try:
        return evidence_retriever.list_courses()
    except Exception:
        logger.exception("Listing the indexed courses failed")
        return {"status": STATUS_ERROR, "courses": [], "error": "The course list is unavailable."}


def safe_search(
    evidence_retriever: EvidenceRetriever,
    question: str,
    retrieval_query: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    course_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    EvidenceRetriever.search() that never raises: invalid arguments and
    retrieval failures become a result with status "invalid_request" or
    "error" and no chunks. Internal details are logged, not returned.
    """
    failure: dict[str, Any] = {
        "question": question,
        "retrieval_query": None,
        "query_language": None,
        "translation": None,
        "translation_error": None,
        "embedded_text": None,
        "embedding_model": None,
        "embedding_dimensions": None,
        "collection": None,
        "top_k": top_k,
        "course_ids": course_ids,
        "chunks": [],
        "courses": [],
        "limitations": [],
    }

    try:
        return evidence_retriever.search(
            question,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            course_ids=course_ids,
            top_k=top_k,
        )
    except ValueError as exc:
        return {**failure, "status": STATUS_INVALID_REQUEST, "error": str(exc)}
    except Exception:
        logger.exception("Knowledge-base search failed")
        return {
            **failure,
            "status": STATUS_ERROR,
            "error": "The knowledge-base search failed.",
        }
