"""
Backend entry point for the User interface: REX Assistant only.

User messages go straight to REX Assistant's LangGraph workflow (which
reaches the knowledge base through the RAG MCP tool). They never pass
through the orchestrator, so the User interface cannot reach any admin
agent.

The conversation state lives in the caller's session memory and is not
written to disk; REX persists learner details only after explicit
consent (agents/rex_assistant/lead_store.py).
"""

from __future__ import annotations

from typing import Any, Optional

from agents.rex_assistant import RexAssistantAgent
from agents.rex_assistant.interface import _new_state


class RexUserService:
    def __init__(self, agent: Optional[RexAssistantAgent] = None) -> None:
        self.agent = agent or RexAssistantAgent()

    @staticmethod
    def new_conversation(interface_language: Optional[str] = None) -> dict[str, Any]:
        """
        A fresh conversation state. The interface language only seeds the
        conversation language: REX still answers in the language the user
        writes in, and falls back to this one for replies too short to
        detect ("ok", "5"). It is not a fixed preference.
        """
        state = _new_state()
        if interface_language in ("ar", "en"):
            state["conversation_language"] = interface_language
        return state

    def chat(
        self,
        message: str,
        state: Optional[dict[str, Any]] = None,
        preferred_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """One user turn. Pass the returned "state" into the next call."""
        return self.agent.handle_message(
            message, state=state, preferred_language=preferred_language
        )

    def reload_knowledge(self) -> None:
        """Restart the RAG MCP server so newly ingested documents are used."""
        self.agent.close()

    def close(self) -> None:
        self.agent.close()
