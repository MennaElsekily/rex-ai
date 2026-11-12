"""
Persistent MCP client for Agent 1 to reach the existing RAG knowledge
base through agents/rag_mcp_server.py, instead of importing the RAG
internals (knowledge_base/) directly.

The connection handling (one server subprocess kept open across turns,
a background thread for the asynchronous MCP session, restart after the
server dies) is shared with the other agents' clients: see
agents/mcp_tool_client.py. This class only names the server and gives
its tools typed, synchronous methods.

The server opens the ChromaDB index when it starts; the embedding
model, cross-encoder and translator load on the first search.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from agents.mcp_tool_client import PROJECT_ROOT, McpToolClient, McpToolError, _drop_none

__all__ = ["PROJECT_ROOT", "McpToolError", "RagMcpClient", "_drop_none"]

# Settings the RAG server needs from this process. The MCP SDK starts the
# server with a minimal environment, so they are passed on explicitly
# (the server also reads them from .env).
_FORWARDED_SETTINGS = ("REX_VECTOR_DB_DIR", "REX_DOCUMENTS_DIR")


class RagMcpClient(McpToolClient):
    """
    Keeps one MCP connection to agents/rag_mcp_server.py alive and
    exposes its tools as synchronous methods.
    """

    def __init__(
        self,
        server_module: str = "agents.rag_mcp_server",
        startup_timeout: float = 120.0,
    ) -> None:
        env = {name: os.environ[name] for name in _FORWARDED_SETTINGS if os.environ.get(name, "").strip()}
        super().__init__(server_module, startup_timeout=startup_timeout, server_name="RAG", env=env or None)

    def search_knowledge_base(
        self,
        question: str,
        retrieval_query: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        course_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
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
            "course_ids": course_ids,
            "top_k": top_k,
        })

    def list_courses(self) -> dict[str, Any]:
        """Call the list_courses MCP tool: the indexed courses (id and name)."""
        return self.call_tool("list_courses", {})

    def get_course_records(self, query: Optional[str] = None, level: Optional[str] = None) -> dict[str, Any]:
        """
        Call the get_course_records MCP tool: the complete list of verified
        course records, optionally narrowed to the topic of a listing
        request or a level (see the server).
        """
        return self.call_tool("get_course_records", {"query": query, "level": level})
