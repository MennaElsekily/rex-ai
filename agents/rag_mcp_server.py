
import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from agents.config import load_env
from knowledge_base.evidence import EvidenceRetriever, safe_course_records, safe_list_courses, safe_search
from knowledge_base.pipeline import configured_documents_dir

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# The MCP client starts this server with a minimal environment: settings
# such as REX_VECTOR_DB_DIR (the index of this deployment) come from .env
# here, or are passed in by the client (agents/rex_assistant/mcp_client.py).
load_env()

mcp = MCPServer("REX AI Knowledge Base")

# One retriever (embedding model, reranker, Chroma connection) for the
# lifetime of this server process. It refuses an index holding documents
# that are not in the authoritative documents folder (status "stale_index"),
# e.g. an obsolete index restored by a git branch switch.
evidence = EvidenceRetriever(documents_dir=configured_documents_dir())


@mcp.tool()
def search_knowledge_base(
    question: str,
    retrieval_query: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
    course_ids: Optional[list[str]] = None,
    top_k: Optional[int] = None,
) -> dict:
    """
    Retrieve document passages relevant to a question (similarity search
    in ChromaDB). Returns evidence only -- it does NOT write an answer; the
    calling agent does that.

    Order of operations: detect the query language; translate Arabic or
    mixed text to English (names, codes and numbers in Latin script kept);
    embed the English text with the index's embedding model; similarity
    search; rerank; label each passage with the course it belongs to.

    Args:
        question: The user's original question, in Arabic or English.
            Returned unchanged.
        retrieval_query: Optional search text prepared by the agent (e.g.
            the request plus a learner profile); translated when needed.
        metadata_filter: Optional filter over indexed metadata fields.
        course_ids: Optional course ids: search only these courses' passages.
        top_k: Optional number of passages, 1-20 (default 5).

    Returns:
        status: "ok", "no_results", "empty_index", "translation_failed",
            "invalid_request" or "error" (never confused with each other;
            a failed translation retrieves nothing).
        question (original), retrieval_query and embedded_text (the text
        embedded), query_language, translation ("not_needed" /
        "translated" / "failed"), translation_error, embedding_model,
        embedding_dimensions, collection, top_k, course_ids,
        chunks: [{reference, text, course_id, course_name (the course the
            passage belongs to, when the catalogue identifies it), source
            (file name only), page, page_end, section, section_path,
            record_id, chunk_id, relevance_score (reranker), distance
            (vector distance), fields}],
        courses: the distinct [{course_id, course_name}] of the chunks,
        limitations, error.
    """
    return safe_search(
        evidence,
        question,
        retrieval_query=retrieval_query,
        metadata_filter=metadata_filter,
        course_ids=course_ids,
        top_k=top_k,
    )


@mcp.tool()
def list_courses() -> dict:
    """
    List the courses of the index (read-only): those named in the indexed
    metadata and the verified programme records.

    Returns:
        status: "ok", "empty_index" or "error".
        courses: [{course_id, course_name}], sorted by course id.
        error.
    """
    return safe_list_courses(evidence)


@mcp.tool()
def get_course_records(query: Optional[str] = None, level: Optional[str] = None) -> dict:
    """
    The catalogue's verified course records (read-only): the COMPLETE list
    of courses the indexed documents identify by both an id and a name --
    not a top-k search -- optionally narrowed to a topic or level.

    Args:
        query: Optional listing request in any language ("What Data
            Analytics courses do you offer?"); translated like a search
            query. Its topic words narrow the list; with none, every course
            is listed.
        level: Optional "beginner", "intermediate" or "advanced" (also read
            from the query).

    Returns:
        status: "ok", "empty_index", "translation_failed" or "error".
        records: [{course_id, course_name, department, level, total_hours,
            duration, delivery_mode, fee, fee_currency, fee_text,
            entry_requirements, prerequisite_courses, prerequisites_text,
            target_audience, summary, data_status, source_files, pages}];
            a fact the catalogue does not state is null.
        total_in_catalogue, complete (true when nothing narrowed the list),
        filter, match_method ("all", "topic", "similarity" or "none"),
        collection, limitations, error.
    """
    return safe_course_records(evidence, query=query, level=level)


if __name__ == "__main__":
    mcp.run()
