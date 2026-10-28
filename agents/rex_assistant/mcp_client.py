"""
Persistent MCP client for Agent 1 to reach the existing RAG knowledge
base through agents/rag_mcp_server.py, instead of importing the RAG
internals (knowledge_base/) directly.

The connection handling (one server subprocess kept open across turns,
a background thread for the asynchronous MCP session, restart after the
server dies) is shared with the other agents' clients: see
agents/mcp_tool_client.py. This class only names the server and gives
the search_knowledge_base tool a typed, synchronous method.

The server opens the ChromaDB index when it starts; the embedding
model, cross-encoder and translator load on the first search.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.mcp_tool_client import PROJECT_ROOT, McpToolClient, McpToolError, _drop_none

__all__ = ["PROJECT_ROOT", "McpToolError", "RagMcpClient", "_drop_none"]


class RagMcpClient(McpToolClient):
    """
    Keeps one MCP connection to agents/rag_mcp_server.py alive and
    exposes the search_knowledge_base tool as a synchronous method.
    """

    def __init__(
        self,
        server_module: str = "agents.rag_mcp_server",
        startup_timeout: float = 120.0,
    ) -> None:
        super().__init__(server_module, startup_timeout=startup_timeout, server_name="RAG")

    def search_knowledge_base(
        self,
        question: str,
        retrieval_query: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Call the search_knowledge_base MCP tool: retrieved passages and
        source metadata with a status, no written answer. See
        agents/rag_mcp_server.py for the result fields.
        """
        return self.call_tool("search_knowledge_base", {
            "question": question,
            "retrieval_query": retrieval_query,
            "metadata_filter": metadata_filter,
        })
