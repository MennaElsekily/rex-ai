"""
Public interface for Agent 1 (REX Assistant).

This is the contract a future central orchestrator (implemented
separately by a teammate, not by this module) can call. It follows the
common request/response structure already documented in
docs/agent_interfaces.md: a request carries user_input and context: a
response carries status, result, and error_message.

This module does not implement, replace, or take ownership of the
orchestrator, and it does not route between REX AI's other agents.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.rex_assistant.graph import build_graph
from agents.rex_assistant.lead_store import JSONLLeadStore, LeadStore
from agents.rex_assistant.mcp_client import RagMcpClient


def _new_state() -> dict[str, Any]:
    return {
        "profile": {},
        "consent": None,
        "consent_pending": False,
        "lead_captured": False,
        "preferred_language": None,
        "history": [],
    }


class RexAssistantAgent:
    """
    Agent 1: Arabic/English course-recommendation chatbot (REX Assistant).

    Usage:
        agent = RexAssistantAgent()
        response = agent.handle_message("I want to learn AI")
        # keep passing response["state"] back in on each subsequent turn
        response = agent.handle_message("I'm a beginner", state=response["state"])

    Session state (profile, consent status, language preference) is the
    caller's responsibility to persist between turns -- this agent does
    not introduce its own session storage, per the project's scope
    guidance to avoid unnecessary persistence systems.

    The knowledge base is reached exclusively through the MCP tool in
    agents/rag_mcp_server.py (see mcp_client.py) -- this class never
    imports or calls the RAG internals directly. Call close() when the
    agent is no longer needed to shut down the MCP server subprocess.
    """

    agent_name = "REX Assistant"

    def __init__(
        self,
        lead_store: Optional[LeadStore] = None,
        model: str = "qwen2.5:3b",
        mcp_client: Optional[RagMcpClient] = None,
    ) -> None:
        self.lead_store = lead_store or JSONLLeadStore()
        self._graph, self._mcp_client = build_graph(
            lead_store=self.lead_store,
            model=model,
            mcp_client=mcp_client,
        )

    def close(self) -> None:
        """Shut down the MCP server subprocess used for RAG access."""
        self._mcp_client.close()

    def handle_message(
        self,
        user_input: str,
        state: Optional[dict[str, Any]] = None,
        preferred_language: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Process one user turn and return the common response structure
        plus the updated `state` to pass into the next call.

        Args:
            user_input: The learner's message for this turn.
            state: The `state` dict returned by the previous call for
                this session, or None to start a new conversation.
            preferred_language: Optional "ar"/"en" override for this
                turn. When omitted, the preference already stored in
                `state` is kept; if none was ever set, the response
                mirrors the detected language of the question.
            request_id: Optional identifier echoed back in the response.

        Returns:
            {
                "request_id": ..., "agent_name": "REX Assistant",
                "status": "success" | "error",
                "result": {
                    "answer": str,
                    "sources": list[dict],
                    "profile": dict,
                    "consent_pending": bool,
                    "lead_captured": bool,
                    "detected_language": str | None,
                    "response_language": str | None,
                } | None,
                "error_message": str | None,
                "state": dict,  # pass this into the next handle_message call
            }
        """
        if not isinstance(user_input, str) or not user_input.strip():
            return {
                "request_id": request_id,
                "agent_name": self.agent_name,
                "status": "error",
                "result": None,
                "error_message": "user_input must be a non-empty string",
                "state": state or _new_state(),
            }

        working_state = dict(state) if state else _new_state()

        if preferred_language:
            working_state["preferred_language"] = preferred_language

        working_state["user_input"] = user_input.strip()

        try:
            result_state = self._graph.invoke(working_state)
        except Exception as exc:
            return {
                "request_id": request_id,
                "agent_name": self.agent_name,
                "status": "error",
                "result": None,
                "error_message": f"{type(exc).__name__}: {exc}",
                "state": working_state,
            }

        history = list(result_state.get("history", []))
        history.append({"role": "user", "content": user_input.strip()})
        history.append(
            {"role": "assistant", "content": result_state.get("response", "")}
        )
        result_state["history"] = history

        rag_result = result_state.get("rag_result") or {}

        return {
            "request_id": request_id,
            "agent_name": self.agent_name,
            "status": "success",
            "result": {
                "answer": result_state.get("response", ""),
                "sources": rag_result.get("sources", []),
                "profile": result_state.get("profile", {}),
                "consent_pending": result_state.get("consent_pending", False),
                "lead_captured": result_state.get("lead_captured", False),
                "detected_language": result_state.get("detected_language"),
                "response_language": result_state.get("response_language"),
            },
            "error_message": None,
            "state": result_state,
        }
