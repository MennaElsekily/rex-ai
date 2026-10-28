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


def _file_name(source: Any) -> str | None:
    """'E:\\...\\catalogue.pdf' -> 'catalogue.pdf' (no internal paths)."""
    if not source:
        return None
    return _PATH_SEPARATORS.split(str(source))[-1] or None


def _to_chunk(reference: int, result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    score = result.get("relevance_score")

    return {
        "reference": reference,
        "text": result.get("text", "").strip(),
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
        chunks (reference, text, source file name, page, section,
        chunk_id, relevance_score) and error.

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

        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise ValueError("metadata_filter must be a dictionary")

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

        if not result["chunks"]:
            result["status"] = STATUS_NO_RESULTS

        return result


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
