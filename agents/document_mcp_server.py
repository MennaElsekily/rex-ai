"""
MCP server for Agent 4 (Educational Content & Document Generation).

Each tool calls the matching method of
agents/document_generation/document_tools.DocumentToolbox, where the logic
and input checks live. There is deliberately no approve tool, no tool that
changes a curriculum, and no tool that takes a file path or returns file
contents: an MCP server cannot tell who is calling, so approval and
downloads stay behind the admin session in agents/admin_service.py.

Why a server at all (rather than the agent opening the databases itself):
it is the same boundary the other agents use, so the agent can only do what
these tools allow, and the curriculum store is reached read-only through
get_approved_curriculum.

Settings (environment or .env, all optional): DOCUMENT_DB_PATH,
DOCUMENT_OUTPUT_DIR, CURRICULUM_DB_PATH.

Run (normally started by DocumentMcpClient):
    python -m agents.document_mcp_server
"""

import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from agents.document_generation.document_tools import DocumentToolbox, toolbox_from_settings

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)

mcp = MCPServer("REX AI Document Generation")

_toolbox: Optional[DocumentToolbox] = None


def _tools() -> DocumentToolbox:
    global _toolbox
    if _toolbox is None:
        _toolbox = toolbox_from_settings()
    return _toolbox


@mcp.tool()
def list_document_types() -> dict:
    """The document types that can be generated, the ones that cannot (with the reason), and the extra information an administrator may supply. Read-only."""
    return _tools().list_document_types()


@mcp.tool()
def list_approved_curricula(limit: int = 50, offset: int = 0) -> dict:
    """One page (limit up to 200, from offset) of the curricula that have an approved version: id, title, language, level, hours, approved and current version, plus total and has_more. Read-only."""
    return _tools().list_approved_curricula(limit, offset)


@mcp.tool()
def get_approved_curriculum(curriculum_id: str, version: Optional[int] = None) -> dict:
    """
    The approved version of a curriculum (or the given version if it is
    approved) with its content. "not_approved" when the curriculum is still
    draft, pending review or has changes requested. Read-only.
    """
    return _tools().get_approved_curriculum(curriculum_id, version)


@mcp.tool()
def save_document_draft(content: dict[str, Any], document_id: Optional[str] = None,
                        change_summary: Optional[str] = None) -> dict:
    """
    Save a generated document, or a new version of one. The server validates
    it against the approved curriculum itself; the status is pending_review
    when valid, otherwise draft. Cannot approve.
    """
    return _tools().save_document_draft(content, document_id, change_summary)


@mcp.tool()
def render_document_pdf(document_id: str, version: Optional[int] = None) -> dict:
    """Render a document version as a PDF (English or Arabic) and record it. A version's PDF is created once and never replaced."""
    return _tools().render_document_pdf(document_id, version)


@mcp.tool()
def get_document(document_id: str, version: Optional[int] = None) -> dict:
    """A document (current or given version) with its content, validation and version list."""
    return _tools().get_document(document_id, version)


@mcp.tool()
def list_documents(status: Optional[str] = None, curriculum_id: Optional[str] = None, limit: int = 50) -> dict:
    """Saved documents, optionally by review status or curriculum."""
    return _tools().list_documents(status, curriculum_id, limit)


if __name__ == "__main__":
    mcp.run()
