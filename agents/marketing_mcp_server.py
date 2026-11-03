"""
MCP server for Agent 5 (AI Marketing Agent).

Each tool calls the matching method of agents/marketing/marketing_tools.MarketingToolbox, where the logic and input
checks live. There is deliberately no approve tool, no tool that changes a curriculum, no tool that publishes or sends
anything, and no tool that returns files: an MCP server cannot tell who is calling, so approval and downloads stay
behind the admin session in agents/admin_service.py.

Why a server at all (rather than the agent opening the databases itself): it is the same boundary the other agents
use, so the agent can only do what these tools allow, and the curriculum store is reached read-only through
get_approved_course (only approved versions).

Settings (environment or .env, all optional): MARKETING_DB_PATH, CURRICULUM_DB_PATH.

Run (normally started by MarketingMcpClient):
    python -m agents.marketing_mcp_server
"""

import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from agents.marketing.marketing_tools import MarketingToolbox, toolbox_from_settings

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)

mcp = MCPServer("REX AI Marketing")

_toolbox: Optional[MarketingToolbox] = None


def _tools() -> MarketingToolbox:
    global _toolbox
    if _toolbox is None:
        _toolbox = toolbox_from_settings()
    return _toolbox


@mcp.tool()
def list_approved_courses(limit: int = 50, offset: int = 0) -> dict:
    """One page (limit up to 200, from offset) of the courses that have an approved version: id, title, language, level, hours, target audience, approved and current version, plus total and has_more. Read-only."""
    return _tools().list_approved_courses(limit, offset)


@mcp.tool()
def get_approved_course(curriculum_id: str, version: Optional[int] = None) -> dict:
    """
    The approved version of a course (or the given version if it is approved) with its content. "not_approved" when the
    curriculum is still draft, pending review or has changes requested. Read-only.
    """
    return _tools().get_approved_course(curriculum_id, version)


@mcp.tool()
def save_campaign_draft(content: dict[str, Any], campaign_id: Optional[str] = None, change_summary: Optional[str] = None) -> dict:
    """
    Save a generated marketing campaign, or a new version of one. The server validates it against the approved course
    itself; the status is pending_review when valid, otherwise draft. Cannot approve, publish or send.
    """
    return _tools().save_campaign_draft(content, campaign_id, change_summary)


@mcp.tool()
def get_campaign(campaign_id: str, version: Optional[int] = None) -> dict:
    """A campaign (current or given version) with its content, validation and version list."""
    return _tools().get_campaign(campaign_id, version)


@mcp.tool()
def list_campaigns(status: Optional[str] = None, curriculum_id: Optional[str] = None, limit: int = 50) -> dict:
    """Saved campaigns, optionally by review status or course."""
    return _tools().list_campaigns(status, curriculum_id, limit)


if __name__ == "__main__":
    mcp.run()
