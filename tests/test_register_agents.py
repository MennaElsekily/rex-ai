"""
Tests for agents.register_agents: REX Assistant registration, the
orchestrator-to-RexAssistantAgent adapter, and conversation-state
continuity/isolation through that adapter.

RexAssistantAgent is mocked everywhere here, so these tests require no
live Ollama server, ChromaDB, or RAG MCP server subprocess.
"""

from unittest.mock import MagicMock

from agents.conversation_store import ConversationStore
from agents.register_agents import (
    REX_ASSISTANT_AGENT_NAME,
    create_orchestrator,
    make_rex_assistant_handler,
)


def _rex_response(answer="ok", state=None, status="success"):
    return {
        "request_id": "ignored-by-adapter",
        "agent_name": "REX Assistant",
        "status": status,
        "result": {"answer": answer},
        "error_message": None,
        "state": state if state is not None else {},
    }


def test_rex_assistant_is_registered(tmp_path):
    fake_agent = MagicMock()
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")

    orchestrator = create_orchestrator(
        rex_assistant_agent=fake_agent,
        conversation_store=store,
    )

    assert REX_ASSISTANT_AGENT_NAME in orchestrator.agents
    assert "curriculum_research" in orchestrator.agents


def test_adapter_extracts_and_passes_correct_arguments(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.return_value = _rex_response(
        answer="hi", state={"profile": {"background": "cs"}}
    )

    handler = make_rex_assistant_handler(fake_agent, store)

    response = handler({
        "request_id": "r1",
        "agent_name": "rex_assistant",
        "user_input": "I want to learn AI",
        "context": {"conversation_id": "conv-1", "language": "en"},
    })

    fake_agent.handle_message.assert_called_once_with(
        "I want to learn AI",
        state=None,
        preferred_language=None,
        request_id="r1",
    )

    assert response == {
        "request_id": "r1",
        "agent_name": "rex_assistant",
        "status": "success",
        "result": {"answer": "hi"},
        "error_message": None,
    }


def test_adapter_does_not_expose_internal_state_in_response(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.return_value = _rex_response(
        state={"profile": {"background": "cs"}, "consent": True}
    )

    handler = make_rex_assistant_handler(fake_agent, store)
    response = handler({
        "request_id": "r1",
        "user_input": "hi",
        "context": {"conversation_id": "conv-1"},
    })

    assert "state" not in response


def test_adapter_persists_state_for_followup_turn(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.side_effect = [
        _rex_response(
            answer="What's your background?",
            state={"profile": {}, "turn": 1},
        ),
        _rex_response(
            answer="Great, noted.",
            state={"profile": {"background": "cs"}, "turn": 2},
        ),
    ]

    handler = make_rex_assistant_handler(fake_agent, store)

    handler({
        "request_id": "r1",
        "user_input": "Hi",
        "context": {"conversation_id": "conv-1"},
    })
    handler({
        "request_id": "r2",
        "user_input": "I have a CS background",
        "context": {"conversation_id": "conv-1"},
    })

    first_call_kwargs = fake_agent.handle_message.call_args_list[0].kwargs
    second_call_kwargs = fake_agent.handle_message.call_args_list[1].kwargs

    assert first_call_kwargs["state"] is None
    assert second_call_kwargs["state"] == {"profile": {}, "turn": 1}

    # And the state after turn 2 is now the one on record for this conversation.
    assert store.get("conv-1", REX_ASSISTANT_AGENT_NAME) == {
        "profile": {"background": "cs"},
        "turn": 2,
    }


def test_adapter_isolates_different_conversations(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.return_value = _rex_response(
        state={"profile": {"background": "cs"}}
    )

    handler = make_rex_assistant_handler(fake_agent, store)

    handler({
        "request_id": "r1",
        "user_input": "hi",
        "context": {"conversation_id": "conv-A"},
    })
    fake_agent.handle_message.reset_mock()

    handler({
        "request_id": "r2",
        "user_input": "hi",
        "context": {"conversation_id": "conv-B"},
    })

    call_kwargs = fake_agent.handle_message.call_args.kwargs
    assert call_kwargs["state"] is None  # conv-B must not see conv-A's state


def test_adapter_falls_back_to_request_id_without_conversation_id(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.return_value = _rex_response()

    handler = make_rex_assistant_handler(fake_agent, store)
    handler({"request_id": "req-only", "user_input": "hi", "context": {}})

    # No conversation_id was supplied, so request_id was used as the key.
    assert store.get("req-only", REX_ASSISTANT_AGENT_NAME) is not None


def test_orchestrator_can_invoke_rex_assistant_end_to_end(tmp_path):
    store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    fake_agent = MagicMock()
    fake_agent.handle_message.return_value = _rex_response(
        answer="Try the AI Foundations course."
    )

    orchestrator = create_orchestrator(
        rex_assistant_agent=fake_agent,
        conversation_store=store,
    )

    response = orchestrator.handle_request(
        "Can you recommend a course for me?",
        request_id="req-1",
    )

    assert response["status"] == "success"
    assert response["result"][0]["agent_name"] == "rex_assistant"
    assert (
        response["result"][0]["result"]["answer"]
        == "Try the AI Foundations course."
    )
