"""
RagMcpClient against a real MCP subprocess (tests/fake_rag_mcp_server.py):
real stdio transport and session handling, but no models, no network and
no knowledge-base data. Covers restarting after close(), retrying after a
failed start, and recovering after the server process dies.
"""

import pytest

from agents.rex_assistant.mcp_client import McpToolError, RagMcpClient

FAKE_SERVER = "tests.fake_rag_mcp_server"


@pytest.fixture
def client():
    client = RagMcpClient(server_module=FAKE_SERVER, startup_timeout=60)
    yield client
    client.close()


def _evidence_text(result):
    return result["chunks"][0]["text"]


def test_search_tool_round_trip_returns_structured_evidence(client):
    result = client.search_knowledge_base(
        question="ما هي الدورات المتاحة؟", retrieval_query="available courses"
    )

    assert result["status"] == "ok"
    assert result["question"] == "ما هي الدورات المتاحة؟"  # unchanged
    assert result["retrieval_query"] == "available courses"
    assert result["chunks"][0]["source"] == "catalogue.pdf"
    assert "answer" not in result


def test_the_server_lists_its_tool_with_a_schema(client):
    [tool] = client.list_tools()

    assert tool["name"] == "search_knowledge_base"
    assert tool["description"]
    assert tool["input_schema"]["required"] == ["question"]
    assert set(tool["input_schema"]["properties"]) == {"question", "retrieval_query", "metadata_filter"}


def test_unset_optional_arguments_use_the_server_defaults(client):
    result = client.search_knowledge_base(question="hello")

    assert result["retrieval_query"] == "hello"


def test_repeated_calls_share_one_session(client):
    client.search_knowledge_base(question="a")
    session = client._session

    client.search_knowledge_base(question="b")

    assert client._session is session


def test_client_restarts_after_close(client):
    assert _evidence_text(client.search_knowledge_base(question="one")) == "evidence for: one"
    client.close()

    assert _evidence_text(client.search_knowledge_base(question="two")) == "evidence for: two"


def test_failed_start_is_retried_on_the_next_call():
    client = RagMcpClient(server_module="tests.no_such_module", startup_timeout=60)

    try:
        with pytest.raises(McpToolError):
            client.search_knowledge_base(question="hi")

        client._server_params.args = ["-m", FAKE_SERVER]
        assert _evidence_text(client.search_knowledge_base(question="hi")) == "evidence for: hi"
    finally:
        client.close()


def test_client_recovers_after_the_server_process_dies(client):
    assert _evidence_text(client.search_knowledge_base(question="before")) == "evidence for: before"

    with pytest.raises(McpToolError):
        client.search_knowledge_base(question="crash")

    assert _evidence_text(client.search_knowledge_base(question="after")) == "evidence for: after"
