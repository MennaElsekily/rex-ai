"""
Registers the Marketing Agent (Agent 5) with an orchestrator.

Kept as its own small function (called by agents/register_agents.py) so the exact registration code can be tested
without importing the other agents' heavier dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agents.conversation_store import ConversationStore
from agents.marketing.marketing_agent import MarketingAgent

if TYPE_CHECKING:
    from agents.orchestrator import Orchestrator


def register_marketing(orchestrator: "Orchestrator", conversation_store: Optional[ConversationStore] = None) -> MarketingAgent:
    """
    Create the agent and register it.

    - Requests are handled by MarketingAgent.process_request (the LangGraph workflow; tools through the Marketing MCP
      server, started on the first tool call, so registering starts nothing).
    - The continuation check lets "2", a course id, or the answer to "who is the audience?" reach the agent after it
      asked -- only while that question is open.
    """
    agent = MarketingAgent(conversation_store=conversation_store)
    orchestrator.register_agent(agent.AGENT_NAME, agent.process_request, continuation_check=agent.wants_follow_up)
    return agent
