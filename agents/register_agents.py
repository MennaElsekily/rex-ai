
from typing import Any, Optional

from agents.orchestrator import AgentHandler, Orchestrator
from agents.curriculum_research.registration import (
    register_curriculum_research,
)
from agents.conversation_store import ConversationStore
from agents.rex_assistant import RexAssistantAgent
from agents.rfp_research import RFPResearchAgent
from agents.rfp_research.opportunity_store import OpportunityStore


REX_ASSISTANT_AGENT_NAME = "rex_assistant"


def make_rex_assistant_handler(
    agent: RexAssistantAgent,
    store: ConversationStore,
) -> AgentHandler:
    """
    Adapt RexAssistantAgent.handle_message to the orchestrator's
    single-dict AgentHandler contract, and persist/restore its
    conversation state via ConversationStore so a follow-up message in
    the same conversation doesn't lose context.

    RexAssistantAgent itself is unmodified and stays stateless between
    calls -- this adapter is what gives it memory across separate
    orchestrator requests.
    """

    def handler(request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id", "unknown")
        user_input = str(request.get("user_input", ""))
        context = request.get("context") or {}

        # The caller (e.g. the Streamlit app) should supply a stable
        # conversation_id to get multi-turn memory; falling back to
        # request_id keeps a single-turn call working without one, at
        # the cost of that call having no prior context of its own.
        conversation_id = context.get("conversation_id") or request_id
        preferred_language = context.get("preferred_language")

        previous_state = store.get(conversation_id, REX_ASSISTANT_AGENT_NAME)

        response = agent.handle_message(
            user_input,
            state=previous_state,
            preferred_language=preferred_language,
            request_id=request_id,
        )

        updated_state = response.get("state")

        if isinstance(updated_state, dict):
            store.set(conversation_id, REX_ASSISTANT_AGENT_NAME, updated_state)

        # Normalize to the common response contract and the
        # orchestrator's snake_case agent name -- the updated_state blob
        # (profile, consent status, etc.) is intentionally not exposed
        # here; it only lives in the ConversationStore.
        return {
            "request_id": request_id,
            "agent_name": REX_ASSISTANT_AGENT_NAME,
            "status": response.get("status", "error"),
            "result": response.get("result"),
            "error_message": response.get("error_message"),
        }

    return handler


def create_orchestrator(
    rex_assistant_agent: Optional[RexAssistantAgent] = None,
    rfp_research_agent: Optional[RFPResearchAgent] = None,
    conversation_store: Optional[ConversationStore] = None,
    opportunity_store: Optional[OpportunityStore] = None,
) -> Orchestrator:
    """
    Create an orchestrator and register implemented agents.

    rex_assistant_agent / rfp_research_agent / conversation_store are
    overridable so tests can inject fakes without needing a live Ollama
    server, ChromaDB, or the RAG MCP server subprocess.

    REX Assistant and RFP Research share one ConversationStore instance
    (store), demonstrating its reuse across agents: REX Assistant uses
    it for learner-profile/consent continuity, RFP Research uses it for
    cross-request duplicate-opportunity detection -- two different
    state shapes, isolated from each other by agent_name, through the
    same store.
    """

    orchestrator = Orchestrator()

    store = conversation_store or ConversationStore()

    # Curriculum Research keeps the latest draft of each conversation in
    # the shared store (its own "curriculum_research" key) so users can
    # refine it with follow-ups. See agents/curriculum_research/registration.py.
    curriculum_agent = register_curriculum_research(orchestrator, store)

    rex_agent = rex_assistant_agent or RexAssistantAgent()
    orchestrator.register_agent(
        REX_ASSISTANT_AGENT_NAME,
        make_rex_assistant_handler(rex_agent, store),
    )

    rfp_agent = rfp_research_agent or RFPResearchAgent(
        conversation_store=store,
        opportunity_store=opportunity_store,
    )
    orchestrator.register_agent(
        rfp_agent.AGENT_NAME,
        rfp_agent.process_request,
    )

    # Exposed for inspection/testing; the orchestrator's own code never
    # reads these attributes.
    orchestrator.conversation_store = store
    orchestrator.curriculum_research_agent = curriculum_agent
    orchestrator.rex_assistant_agent = rex_agent
    orchestrator.rfp_research_agent = rfp_agent

    return orchestrator
