"""
Registers the Curriculum Research Agent with an orchestrator.

Kept as its own small function (called by agents/register_agents.py) so
the exact registration code can be tested without importing the other
agents' heavier dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agents.conversation_store import ConversationStore
from agents.curriculum_research.agent import CurriculumResearchAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


def register_curriculum_research(
    orchestrator: "Orchestrator",
    conversation_store: Optional[ConversationStore] = None,
) -> CurriculumResearchAgent:
    """
    Create the agent and register it.

    - Requests are handled by CurriculumResearchAgent.process_request.
    - The shared ConversationStore (its own "curriculum_research" key)
      remembers the latest draft per conversation_id.
    - The agent also tells the orchestrator when a message is a
      refinement of that draft ("make it 40 hours"), so such follow-ups
      reach it even without a curriculum keyword. Only when the
      conversation has a stored draft; explicit requests for other
      agents (tenders, course recommendations, ...) still win.
    """

    agent = CurriculumResearchAgent(conversation_store=conversation_store)

    orchestrator.register_agent(
        agent.AGENT_NAME,
        agent.process_request,
        continuation_check=agent.has_refinable_draft,
    )

    return agent
