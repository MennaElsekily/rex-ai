"""
Registers the Curriculum Agent (Agent 3) with an orchestrator.

Kept as its own small function (called by agents/register_agents.py) so
the exact registration code can be tested without importing the other
agents' heavier dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agents.conversation_store import ConversationStore
from agents.curriculum_research.curriculum_agent import CurriculumAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


def register_curriculum_research(
    orchestrator: "Orchestrator",
    conversation_store: Optional[ConversationStore] = None,
) -> CurriculumAgent:
    """
    Create the agent and register it.

    - Requests are handled by CurriculumAgent.process_request (the
      LangGraph workflow; tools through the Curriculum MCP server, started
      on the first tool call, so registering starts nothing).
    - The shared ConversationStore (its own "curriculum_research" key)
      remembers the active curriculum, an open question and the last
      suggestions per conversation_id.
    - The continuation check lets a follow-up reach the agent without a
      curriculum keyword ("make it 40 hours", "generate suggestion 2",
      "40 hours, intermediate" after a question) -- only when this
      conversation has such state; explicit requests for other agents
      still win.
    """

    agent = CurriculumAgent(conversation_store=conversation_store)

    orchestrator.register_agent(
        agent.AGENT_NAME,
        agent.process_request,
        continuation_check=agent.wants_follow_up,
    )

    return agent
