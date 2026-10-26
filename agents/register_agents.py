
from typing import Any, Optional

from agents.orchestrator import AgentHandler, Orchestrator
from agents.curriculum_research import CurriculumResearchAgent
from agents.conversation_store import ConversationStore
from agents.rex_assistant import RexAssistantAgent


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
    conversation_store: Optional[ConversationStore] = None,
) -> Orchestrator:
    """
    Create an orchestrator and register implemented agents.

    rex_assistant_agent / conversation_store are overridable so tests
    can inject fakes without needing a live Ollama server, ChromaDB, or
    the RAG MCP server subprocess.
    """

    orchestrator = Orchestrator()

    curriculum_agent = CurriculumResearchAgent()
    orchestrator.register_agent(
        "curriculum_research",
        curriculum_agent.process_request,
    )

    store = conversation_store or ConversationStore()
    rex_agent = rex_assistant_agent or RexAssistantAgent()

    orchestrator.register_agent(
        REX_ASSISTANT_AGENT_NAME,
        make_rex_assistant_handler(rex_agent, store),
    )

    # Exposed for inspection/testing; the orchestrator's own code never
    # reads these attributes.
    orchestrator.conversation_store = store
    orchestrator.rex_assistant_agent = rex_agent

    return orchestrator
