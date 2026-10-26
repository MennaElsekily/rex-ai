"""
Tests for agents.rex_assistant (Agent 1 / REX Assistant).

The chat LLM (_get_llm) and the RAG MCP client are mocked so these
tests validate Agent 1's own graph logic -- profile tracking,
follow-up routing, consent handling, and error handling -- without
requiring a running Ollama server, the embedding/reranking models, or
spawning the actual agents/rag_mcp_server.py subprocess.

A small number of real integration checks (live Ollama, live RAG,
live MCP subprocess) were run manually during development; see the
final report for what was verified live versus mocked here.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.rex_assistant.graph import build_graph
from agents.rex_assistant.interface import RexAssistantAgent
from agents.rex_assistant.lead_store import LeadStore


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self.reply)


class FakeLeadStore(LeadStore):
    def __init__(self):
        self.saved = []

    def save_lead(self, lead):
        self.saved.append(lead)
        return True


def _base_state(**overrides):
    state = {
        "profile": {},
        "consent": None,
        "consent_pending": False,
        "lead_captured": False,
        "preferred_language": None,
        "history": [],
    }
    state.update(overrides)
    return state


@pytest.fixture
def mock_mcp_client():
    client = MagicMock()
    client.ask_knowledge_base.return_value = {
        "answer": "Entry requirements: basic computer literacy. [1]",
        "sources": [{"reference": 1, "source": "doc.pdf"}],
        "model": "qwen2.5:3b",
    }
    return client


def _build_graph(lead_store, mcp_client, profile_json="{}"):
    with patch(
        "agents.rex_assistant.graph._get_llm",
        return_value=FakeLLM(profile_json),
    ):
        graph, _ = build_graph(lead_store=lead_store, mcp_client=mcp_client)
    return graph


def test_incomplete_profile_routes_to_followup(mock_mcp_client):
    lead_store = FakeLeadStore()

    with patch(
        "agents.rex_assistant.graph._get_llm",
        return_value=FakeLLM('{"background": "computer science"}'),
    ):
        graph, _ = build_graph(lead_store=lead_store, mcp_client=mock_mcp_client)
        state = _base_state(user_input="Hi, I want to learn something new")
        result = graph.invoke(state)

    assert result["profile"]["background"] == "computer science"
    mock_mcp_client.ask_knowledge_base.assert_not_called()


def test_known_fields_are_not_requested_again(mock_mcp_client):
    lead_store = FakeLeadStore()

    with patch(
        "agents.rex_assistant.graph._get_llm",
        return_value=FakeLLM('{"career_goal": "data analyst"}'),
    ):
        graph, _ = build_graph(lead_store=lead_store, mcp_client=mock_mcp_client)
        state = _base_state(
            user_input="I want to be a data analyst",
            profile={"background": "already known"},
        )
        result = graph.invoke(state)

    # background must survive; only the newly mentioned field is added.
    assert result["profile"]["background"] == "already known"
    assert result["profile"]["career_goal"] == "data analyst"


def test_direct_knowledge_question_invokes_mcp_tool(mock_mcp_client):
    lead_store = FakeLeadStore()
    graph = _build_graph(lead_store, mock_mcp_client)

    state = _base_state(
        user_input="What are the course fees and prerequisites?",
    )
    result = graph.invoke(state)

    mock_mcp_client.ask_knowledge_base.assert_called_once()
    call_kwargs = mock_mcp_client.ask_knowledge_base.call_args.kwargs
    assert call_kwargs["question"] == state["user_input"]
    assert "Entry requirements" in result["response"]


def test_sufficient_profile_triggers_recommendation_and_consent_prompt(mock_mcp_client):
    lead_store = FakeLeadStore()
    graph = _build_graph(lead_store, mock_mcp_client)

    state = _base_state(
        user_input="I think I'm ready",
        profile={
            "background": "cs",
            "career_goal": "ml engineer",
            "skill_level": "beginner",
        },
    )
    result = graph.invoke(state)

    mock_mcp_client.ask_knowledge_base.assert_called_once()
    call_kwargs = mock_mcp_client.ask_knowledge_base.call_args.kwargs
    # A recommendation-oriented retrieval query should be built from the
    # profile rather than sending the raw small-talk message.
    assert call_kwargs["retrieval_query"] is not None
    assert "ml engineer" in call_kwargs["retrieval_query"]
    assert result["consent_pending"] is True
    assert "yes" in result["response"].lower() or "no" in result["response"].lower()


def test_consent_yes_saves_lead_and_does_not_rerun_profile_extraction(mock_mcp_client):
    lead_store = FakeLeadStore()
    fake_llm = FakeLLM("{}")

    with patch("agents.rex_assistant.graph._get_llm", return_value=fake_llm):
        graph, _ = build_graph(lead_store=lead_store, mcp_client=mock_mcp_client)
        state = _base_state(
            user_input="yes",
            profile={
                "background": "cs",
                "career_goal": "ml engineer",
                "skill_level": "beginner",
            },
            consent_pending=True,
        )
        result = graph.invoke(state)

    assert result["lead_captured"] is True
    assert len(lead_store.saved) == 1
    # Profile extraction must be skipped for a consent reply.
    assert fake_llm.calls == []
    # "yes" must not be absorbed into the profile as a fact.
    assert result["profile"] == {
        "background": "cs",
        "career_goal": "ml engineer",
        "skill_level": "beginner",
    }


def test_consent_no_does_not_save_lead(mock_mcp_client):
    lead_store = FakeLeadStore()
    graph = _build_graph(lead_store, mock_mcp_client)

    state = _base_state(user_input="no", consent_pending=True)
    result = graph.invoke(state)

    assert result["lead_captured"] is False
    assert lead_store.saved == []
    assert result["consent"] is False


def test_ambiguous_consent_reply_asks_again(mock_mcp_client):
    lead_store = FakeLeadStore()
    graph = _build_graph(lead_store, mock_mcp_client)

    state = _base_state(user_input="maybe later", consent_pending=True)
    result = graph.invoke(state)

    assert result["consent_pending"] is True
    assert lead_store.saved == []


def test_mcp_tool_failure_degrades_gracefully():
    lead_store = FakeLeadStore()
    broken_client = MagicMock()
    broken_client.ask_knowledge_base.side_effect = RuntimeError(
        "MCP server unavailable"
    )
    graph = _build_graph(lead_store, broken_client)

    state = _base_state(user_input="What are the course fees?")
    result = graph.invoke(state)

    assert "temporarily unavailable" in result["response"]
    assert "MCP server unavailable" in result["rag_result"]["error"]


def test_arabic_question_resolves_arabic_response_language(mock_mcp_client):
    lead_store = FakeLeadStore()
    graph = _build_graph(lead_store, mock_mcp_client)

    state = _base_state(user_input="ما هي رسوم الدورة؟")
    result = graph.invoke(state)

    assert result["detected_language"] == "ar"
    assert result["response_language"] == "ar"


def test_interface_rejects_empty_message(mock_mcp_client):
    agent = RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=mock_mcp_client)
    response = agent.handle_message("   ")

    assert response["status"] == "error"
    assert response["result"] is None


def test_interface_returns_common_response_shape(mock_mcp_client):
    with patch("agents.rex_assistant.graph._get_llm", return_value=FakeLLM("{}")):
        agent = RexAssistantAgent(
            lead_store=FakeLeadStore(), mcp_client=mock_mcp_client
        )
        response = agent.handle_message(
            "What are the course fees and prerequisites?",
            request_id="req-1",
        )

    assert response["request_id"] == "req-1"
    assert response["agent_name"] == "REX Assistant"
    assert response["status"] == "success"
    assert "answer" in response["result"]
    assert "state" in response


def test_interface_close_delegates_to_mcp_client(mock_mcp_client):
    agent = RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=mock_mcp_client)
    agent.close()

    mock_mcp_client.close.assert_called_once()
