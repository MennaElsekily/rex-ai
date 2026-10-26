"""
Public interface for the RFP/Tender Research Agent (Agent 2).

RFPResearchAgent.process_request(request: dict) -> dict is written to
directly match the orchestrator's AgentHandler contract from the
start, the same way CurriculumResearchAgent does -- so no separate
top-level adapter is needed in register_agents.py. Internally it runs
a LangGraph workflow (graph.py) that queries real tender APIs
(agents/rfp_research/api_clients/) and uses an LLM only to judge
relevance and qualification (tools.py) -- no web search or HTML
scraping is used. The orchestrator never sees any of that, or the
OpportunityStore/ConversationStore used internally.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.conversation_store import ConversationStore
from agents.rfp_research.graph import RFP_AGENT_NAME, build_graph
from agents.rfp_research.opportunity_store import OpportunityStore


class RFPResearchAgent:
    """Discovers, filters, and assesses real educational tender/RFP opportunities."""

    AGENT_NAME = RFP_AGENT_NAME

    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        opportunity_store: Optional[OpportunityStore] = None,
        model: str = "qwen2.5:3b",
    ) -> None:
        self.conversation_store = conversation_store
        self.opportunity_store = opportunity_store or OpportunityStore()
        self.graph = build_graph(
            conversation_store=conversation_store,
            opportunity_store=self.opportunity_store,
            model=model,
        )

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
