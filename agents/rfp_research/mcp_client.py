"""
Persistent MCP client for Agent 2 (RFP Research) to reach its tender
tools through agents/tender_mcp_server.py.

The connection handling is shared with REX's client
(agents/mcp_tool_client.py); this class only names the server and gives
each tool a typed, synchronous method. See tender_tools.py for the
result fields.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.mcp_tool_client import McpToolClient


class TenderMcpClient(McpToolClient):
    """Keeps one MCP connection to agents/tender_mcp_server.py alive."""

    def __init__(
        self,
        server_module: str = "agents.tender_mcp_server",
        startup_timeout: float = 120.0,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(server_module, startup_timeout=startup_timeout,
                         server_name="tender", env=env)

    def search_procurement_notices(self, country: str, query: str) -> dict[str, Any]:
        return self.call_tool("search_procurement_notices", {"country": country, "query": query})

    def search_tender_web_pages(
        self,
        countries: list[str],
        categories: Optional[list[str]] = None,
        priority: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.call_tool("search_tender_web_pages", {
            "countries": countries, "categories": categories, "priority": priority,
        })

    def find_known_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call_tool("find_known_tenders", {"tenders": tenders})

    def save_discovered_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call_tool("save_discovered_tenders", {"tenders": tenders})
