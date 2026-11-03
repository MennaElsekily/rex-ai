"""
Persistent MCP client for Agent 4 to reach its tools through
agents/document_mcp_server.py. The connection handling is shared with the
other agents (agents/mcp_tool_client.py); this class only names the server
and gives each tool a typed method with the same signature as
DocumentToolbox, so tests can use the toolbox in-process instead.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.mcp_tool_client import McpToolClient


class DocumentMcpClient(McpToolClient):
    def __init__(self, server_module: str = "agents.document_mcp_server", startup_timeout: float = 120.0,
                 env: Optional[dict[str, str]] = None) -> None:
        super().__init__(server_module, startup_timeout=startup_timeout, server_name="document", env=env)

    def list_document_types(self) -> dict[str, Any]:
        return self.call_tool("list_document_types", {})

    def list_approved_curricula(self, limit=50, offset=0) -> dict[str, Any]:
        return self.call_tool("list_approved_curricula", {"limit": limit, "offset": offset})

    def get_approved_curriculum(self, curriculum_id, version=None) -> dict[str, Any]:
        return self.call_tool("get_approved_curriculum", {"curriculum_id": curriculum_id, "version": version})

    def save_document_draft(self, content, document_id=None, change_summary=None) -> dict[str, Any]:
        return self.call_tool("save_document_draft", {"content": content, "document_id": document_id,
                                                      "change_summary": change_summary})

    def render_document_pdf(self, document_id, version=None) -> dict[str, Any]:
        return self.call_tool("render_document_pdf", {"document_id": document_id, "version": version})

    def get_document(self, document_id, version=None) -> dict[str, Any]:
        return self.call_tool("get_document", {"document_id": document_id, "version": version})

    def list_documents(self, status=None, curriculum_id=None, limit=50) -> dict[str, Any]:
        return self.call_tool("list_documents", {"status": status, "curriculum_id": curriculum_id, "limit": limit})
