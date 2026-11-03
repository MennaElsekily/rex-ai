"""
Regressions found while debugging the running Streamlit app:

- a model that is not installed (or an Ollama that is not running) is named in the error, not hidden behind
  "ResponseError", and the original exception is kept as the cause;
- an MCP call that times out says which tool, which server and how long, instead of an empty message;
- the Curricula tab shows a newly saved version instead of staying on the one chosen before.
"""

import asyncio

import pytest
from ollama import ResponseError

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.curriculum_research.llm import CurriculumModelError, describe_model_error, structured
from agents.curriculum_research.schemas import RequestSlots
from agents.mcp_tool_client import McpToolClient, McpToolError
from tests.curriculum_fakes import FakeLLM
from tests.curriculum_fixtures import valid_curriculum
from tests.test_rex_ai_app import env, start, unlock  # noqa: F401  (env is a fixture)


class NamedModel:
    """What ChatOllama exposes: the model name, and a structured-output wrapper that always fails."""

    def __init__(self, model, error):
        self.model, self.error = model, error

    def with_structured_output(self, schema):
        error = self.error

        class Runnable:
            def invoke(self, messages):
                raise error

        return Runnable()


def test_a_missing_model_is_named_with_the_command_that_fixes_it():
    llm = NamedModel("qwen2.5:3b", ResponseError("model 'qwen2.5:3b' not found", 404))

    with pytest.raises(CurriculumModelError) as raised:
        structured(llm, RequestSlots, [], "outline")

    message = str(raised.value)
    assert "outline step" in message and "'qwen2.5:3b' is not installed in Ollama" in message
    assert "ollama pull qwen2.5:3b" in message
    assert isinstance(raised.value.__cause__, ResponseError)          # the original error is kept
    assert raised.value.__cause__.status_code == 404


def test_an_ollama_that_is_not_running_is_named():
    class ConnectError(Exception):
        pass

    llm = NamedModel("qwen2.5:3b", ConnectError("All connection attempts failed"))
    with pytest.raises(CurriculumModelError, match="Ollama is not reachable"):
        structured(llm, RequestSlots, [], "outline")


@pytest.mark.parametrize("exc, llm, expected", [
    (ResponseError("model 'x' not found", 404), NamedModel("x", None), "ollama pull x"),
    (ResponseError("model 'x' not found", 404), object(), "The configured model is not installed"),
    (ValueError("bad json"), NamedModel("x", None), "ValueError: bad json"),
    (ValueError(""), None, "ValueError"),
])
def test_describe_model_error(exc, llm, expected):
    assert expected in describe_model_error(exc, llm)


def test_other_failures_keep_working_as_before():
    failing = FakeLLM({"RequestSlots": [ValueError("bad"), ValueError("bad again")]})
    with pytest.raises(CurriculumModelError, match="test step") as raised:
        structured(failing, RequestSlots, [], "test")
    assert "ValueError" in str(raised.value) and len(failing.calls) == 2


# --------------------------------------------------------------------
# MCP: a timeout is explained
# --------------------------------------------------------------------

class _TimedOut:
    def result(self, timeout=None):
        raise TimeoutError()


def _client_with_a_hung_server(monkeypatch):
    client = McpToolClient("agents.some_server", startup_timeout=7, server_name="curriculum")
    client._session, client._loop = object(), asyncio.new_event_loop()
    closed = []
    monkeypatch.setattr(client, "start", lambda: None)
    monkeypatch.setattr(client, "close", lambda: closed.append(True))

    def run_threadsafe(coroutine, loop):
        coroutine.close()   # never awaited: the "server" never answers
        return _TimedOut()

    monkeypatch.setattr("agents.mcp_tool_client.asyncio.run_coroutine_threadsafe", run_threadsafe)
    return client, closed


def test_a_tool_call_that_times_out_says_which_tool_and_how_long(monkeypatch):
    client, closed = _client_with_a_hung_server(monkeypatch)

    with pytest.raises(McpToolError) as raised:
        client.call_tool("get_course_details", {"course": "data analysis"})

    message = str(raised.value)
    assert "curriculum MCP server" in message and "get_course_details" in message and "7 seconds" in message
    assert message.strip() != "MCP tool call failed:"
    assert isinstance(raised.value.__cause__, TimeoutError) and closed == [True]   # server dropped, next call restarts


def test_a_timeout_without_a_tool_name_is_still_explained(monkeypatch):
    client, _ = _client_with_a_hung_server(monkeypatch)
    with pytest.raises(McpToolError, match="did not answer the request within 7 seconds"):
        client.list_tools()


# --------------------------------------------------------------------
# Streamlit: the Curricula tab follows a new version
# --------------------------------------------------------------------

def test_the_curricula_tab_shows_a_new_version_after_a_revision(env):
    store = CurriculumStore()   # the temporary default path set in tests/conftest.py
    saved = store.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    curriculum_id = saved["curriculum_id"]
    at = unlock(start())
    assert next(s for s in at.selectbox if s.key == f"cur_version_{curriculum_id}_1").value == 1

    revised = valid_curriculum()
    revised["title"] = "Machine Learning Foundations, revised"
    store.save_version(revised, {"valid": True, "errors": []}, [], curriculum_id=curriculum_id)
    at.run()

    assert not at.exception, at.exception
    selector = next(s for s in at.selectbox if s.key == f"cur_version_{curriculum_id}_2")
    assert selector.value == 2            # not stuck on version 1
    assert any("revised" in e.label and "v2" in e.label for e in at.expander)   # the list shows the new title and version
