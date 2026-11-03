"""
Persistent MCP client for Agent 5 to reach its tools through agents/marketing_mcp_server.py. The connection handling is
shared with the other agents (agents/mcp_tool_client.py); this class only names the server and gives each tool a typed
method with the same signature as MarketingToolbox, so tests can use the toolbox in-process instead.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.mcp_tool_client import McpToolClient


class MarketingMcpClient(McpToolClient):
    def __init__(self, server_module: str = "agents.marketing_mcp_server", startup_timeout: float = 120.0,
                 env: Optional[dict[str, str]] = None) -> None:
        super().__init__(server_module, startup_timeout=startup_timeout, server_name="marketing", env=env)

    def list_approved_courses(self, limit=50, offset=0) -> dict[str, Any]:
        return self.call_tool("list_approved_courses", {"limit": limit, "offset": offset})

    def get_approved_course(self, curriculum_id, version=None) -> dict[str, Any]:
        return self.call_tool("get_approved_course", {"curriculum_id": curriculum_id, "version": version})

    def save_campaign_draft(self, content, campaign_id=None, change_summary=None) -> dict[str, Any]:
        return self.call_tool("save_campaign_draft", {"content": content, "campaign_id": campaign_id,
                                                      "change_summary": change_summary})

    def get_campaign(self, campaign_id, version=None) -> dict[str, Any]:
        return self.call_tool("get_campaign", {"campaign_id": campaign_id, "version": version})

    def list_campaigns(self, status=None, curriculum_id=None, limit=50) -> dict[str, Any]:
        return self.call_tool("list_campaigns", {"status": status, "curriculum_id": curriculum_id, "limit": limit})
