"""
Public interface for the RFP/Tender Research Agent (Agent 2).

RFPResearchAgent.process_request(request: dict) -> dict is written to
directly match the orchestrator's AgentHandler contract from the
start, the same way CurriculumAgent does -- so no separate
top-level adapter is needed in register_agents.py. Internally it runs
a LangGraph workflow (graph.py) that queries the tender APIs and
searches the web for tenders in Egypt and Saudi Arabia. Every tender
found is stored as pending in the OpportunityStore for admin approval.
The sources and the store are reached through the tender MCP tools
(agents/tender_mcp_server.py via TenderMcpClient). The orchestrator
never sees any of that, or the stores used internally.

request["context"]["search"] carries the admin's search form choices
(countries, categories, priority, eligibility_notes, deadline_from,
deadline_to); a chat request without it searches both countries.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.conversation_store import ConversationStore
from agents.rfp_research.graph import RFP_AGENT_NAME, TenderTools, build_graph
from agents.rfp_research.mcp_client import TenderMcpClient
from agents.rfp_research.opportunity_store import OpportunityStore
from agents.rfp_research.tender_tools import TENDER_DB_SETTING


def default_tender_tools(store: OpportunityStore) -> TenderTools:
    """
    The tender MCP server, using the same database file as `store`, so
    the admin dashboard (which reads `store`) sees what the agent saves.
    tests/conftest.py replaces this with the in-process tools.
    """
    return TenderMcpClient(env={TENDER_DB_SETTING: str(store.db_path)})


class RFPResearchAgent:
    """Discovers, filters, and assesses educational tender/RFP opportunities."""

    AGENT_NAME = RFP_AGENT_NAME

    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        opportunity_store: Optional[OpportunityStore] = None,
        model: str = "qwen2.5:3b",
        web_search: bool = True,
        tools: Optional[TenderTools] = None,
    ) -> None:
        """
        web_search: False queries the tender APIs only.
        tools: the tender tools; defaults to the tender MCP server.
        """
        self.conversation_store = conversation_store
        self.opportunity_store = opportunity_store or OpportunityStore()
        self.tools = tools or default_tender_tools(self.opportunity_store)
        self.graph = build_graph(
            self.tools,
            conversation_store=conversation_store,
            model=model,
            web_search=web_search,
        )

    def close(self) -> None:
        """Stop the tender MCP server subprocess, if one was started."""
        close = getattr(self.tools, "close", None)
        if close:
            close()

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process one RFP/tender research request."""
        request_id = request.get("request_id", "unknown")
        user_input = str(request.get("user_input", "")).strip()
        context = request.get("context") or {}
        conversation_id = context.get("conversation_id")

        if not user_input:
            return self._error_response(
                request_id, "The tender research request cannot be empty."
            )

        try:
            final_state = self.graph.invoke({
                "request_text": user_input,
                "conversation_id": conversation_id,
                "language": "ar" if context.get("language") == "ar" else "en",
                "search": dict(context.get("search") or {}),
            })
        except Exception as exc:
            return self._error_response(request_id, str(exc))

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "success",
            "result": final_state.get("response"),
            "error_message": None,
        }

    def _error_response(
        self, request_id: str, message: str
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "error",
            "result": None,
            "error_message": message,
        }
