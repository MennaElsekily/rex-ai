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

import logging
import re
from typing import Any

from knowledge_base.language import prepare_retrieval_query
from knowledge_base.retriever import Retriever

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_EMPTY_INDEX = "empty_index"
STATUS_INVALID_REQUEST = "invalid_request"
STATUS_ERROR = "error"

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


def _file_name(source: Any) -> str | None:
    """'E:\\...\\catalogue.pdf' -> 'catalogue.pdf' (no internal paths)."""
    if not source:
        return None
    return _PATH_SEPARATORS.split(str(source))[-1] or None


def _to_chunk(reference: int, result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    score = result.get("relevance_score")
    course_id, course_name = _course_of(metadata)

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
        "relevance_score": round(float(score), 4) if score is not None else None,
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

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever(top_k=5)

    def search(
        self,
        question: str,
        retrieval_query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
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

        Returns a dict with: status (ok / no_results / empty_index),
        question, retrieval_query (the text embedded), query_language,
        translation (not_needed / translated / failed), translation_error,
        chunks (reference, text, course_id, course_name, source file name,
        page, section, chunk_id, relevance_score, fields), courses (the
        distinct courses the chunks belong to, in relevance order) and
        error.

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
            "chunks": [],
            "courses": [],
            "error": None,
        }

        if self.retriever.vector_store.collection.count() == 0:
            result["status"] = STATUS_EMPTY_INDEX
            result["error"] = "The knowledge-base index is empty or missing."
            return result

        retrieved = self.retriever.retrieve(
            query=original_question,
            retrieval_query=prepared.query,
            metadata_filter=metadata_filter,
        )

        result["chunks"] = [
            chunk
            for chunk in (
                _to_chunk(reference, item)
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

        courses = sorted(by_key.values(), key=lambda c: (c["course_id"] or "", c["course_name"] or ""))
        return {"status": STATUS_OK, "courses": courses, "error": None}


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
        "chunks": [],
        "courses": [],
    }

    try:
        return evidence_retriever.search(
            question,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
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
