"""
Registers the Document Generation Agent (Agent 4) with an orchestrator.

Kept as its own small function (called by agents/register_agents.py) so
the exact registration code can be tested without importing the other
agents' heavier dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agents.conversation_store import ConversationStore
from agents.document_generation.document_agent import DocumentAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


def register_document_generation(
    orchestrator: "Orchestrator",
    conversation_store: Optional[ConversationStore] = None,
) -> DocumentAgent:
    """
    Create the agent and register it.

    - Requests are handled by DocumentAgent.process_request (the LangGraph
      workflow; tools through the Document MCP server, started on the first
      tool call, so registering starts nothing).
    - The continuation check lets "2" or a curriculum id reach the agent
      after it asked "which curriculum?" -- only while that question is open.
    """
    agent = DocumentAgent(conversation_store=conversation_store)
    orchestrator.register_agent(agent.AGENT_NAME, agent.process_request, continuation_check=agent.wants_follow_up)
    return agent
