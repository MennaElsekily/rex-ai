
import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from knowledge_base.evidence import EvidenceRetriever, safe_search

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

mcp = MCPServer("REX AI Knowledge Base")

# One retriever (embedding model, reranker, Chroma connection) for the
# lifetime of this server process.
evidence = EvidenceRetriever()


@mcp.tool()
def search_knowledge_base(
    question: str,
    retrieval_query: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Retrieve document passages relevant to a question. Returns evidence
    only -- it does NOT write an answer; the calling agent does that.

    Args:
        question: The user's original question, in Arabic or English.
            Returned unchanged.
        retrieval_query: Optional search text prepared by the agent (e.g.
            from a learner profile); translated to English when needed.
        metadata_filter: Optional filter over indexed metadata fields.

    Returns:
        status: "ok", "no_results", "empty_index", "invalid_request" or
            "error" (no results and failures are never confused).
        question, retrieval_query (text actually searched),
        query_language, translation ("not_needed" / "translated" /
        "failed"), translation_error,
        chunks: [{reference, text, source (file name only), page,
            page_end, section, section_path, record_id, chunk_id,
            relevance_score, fields (structured fields of the source
            record or section, e.g. {"level": "Beginner"})}],
        error.
    """
    return safe_search(
        evidence,
        question,
        retrieval_query=retrieval_query,
        metadata_filter=metadata_filter,
    )


if __name__ == "__main__":
    mcp.run()
