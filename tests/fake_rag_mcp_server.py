"""
Minimal stand-in for agents/rag_mcp_server.py used by tests/test_mcp_client.py:
same MCP server class and tool name, but no models and no knowledge base.
Asking the question "crash" kills the process to simulate a server failure.
"""

import os
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Fake REX AI Knowledge Base")


@mcp.tool()
def search_knowledge_base(
    question: str,
    retrieval_query: Optional[str] = None,
    metadata_filter: Optional[dict[str, Any]] = None,
) -> dict:
    """Retrieve document passages relevant to a question (fake evidence)."""
    if question == "crash":
        os._exit(1)

    return {
        "status": "ok",
        "question": question,
        "retrieval_query": retrieval_query or question,
        "query_language": "en",
        "translation": "not_needed",
        "translation_error": None,
        "chunks": [{
            "reference": 1,
            "text": f"evidence for: {question}",
            "source": "catalogue.pdf",
            "page": 1,
            "section": None,
            "chunk_id": 7,
            "relevance_score": 1.5,
        }],
        "error": None,
    }


if __name__ == "__main__":
    mcp.run()
