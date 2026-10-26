"""
Integration tests: the REAL Curriculum Research Agent, registered through
the real registration code, run through the real orchestrator and the
shared ConversationStore. Other agents are recording stand-ins, used to
check routing priority and that nothing else is disturbed.

The final section goes through agents.register_agents.create_orchestrator.
That module imports the REX Assistant (MCP, embeddings, ...), so those
tests are skipped -- not failed -- where those dependencies are missing.
"""

from unittest.mock import MagicMock

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research.registration import (
    register_curriculum_research,
)
from agents.orchestrator import Orchestrator

AGENT_KEY = "curriculum_research"


class Recorder:
    """Stand-in agent that records the requests it receives."""

    def __init__(self, name):
        self.name = name
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return {
            "request_id": request["request_id"],
            "agent_name": self.name,
            "status": "success",
            "result": {"handled_by": self.name},
            "error_message": None,
        }


@pytest.fixture
def store():
    return ConversationStore(db_path=":memory:")


@pytest.fixture
def system(store):
    orchestrator = Orchestrator()
    agent = register_curriculum_research(orchestrator, store)

    others = {name: Recorder(name) for name in (
        "rex_assistant", "rfp_research", "marketing", "document_generation",
    )}
    for name, recorder in others.items():
        orchestrator.register_agent(name, recorder)

    # Spy on what the curriculum handler really receives.
    received = []
    real_handler = orchestrator.agents[AGENT_KEY]

    def spy(request):
        received.append(request)
        return real_handler(request)

    orchestrator.agents[AGENT_KEY] = spy

    class System:
        pass

    result = System()
    result.orchestrator = orchestrator
    result.agent = agent
    result.others = others
    result.received = received
    result.store = store

    def ask(text, conv="conv-1"):
        return orchestrator.handle_request(
            text, context={"conversation_id": conv}, request_id="req-1"
        )

    result.ask = ask
    return result


def curriculum_result(response):
    assert response["result"][0]["agent_name"] == AGENT_KEY
    return response["result"][0]["result"]


def curricula(response):
    return curriculum_result(response)["curricula"]


# --------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------

def test_registration_connects_the_real_agent_and_the_shared_store(store):
    orchestrator = Orchestrator()

    agent = register_curriculum_research(orchestrator, store)

    handler = orchestrator.agents[AGENT_KEY]
    assert handler.__self__ is agent
    assert handler.__func__ is agent.process_request.__func__
    assert agent.conversation_store is store
    assert orchestrator._continuation_checks[AGENT_KEY].__self__ is agent


def test_registration_without_a_store_still_works():
    orchestrator = Orchestrator()
    register_curriculum_research(orchestrator)

    response = orchestrator.handle_request("Find a curriculum about AI")

    assert response["status"] == "success"


# --------------------------------------------------------------------
# Search through the orchestrator (English and Arabic)
# --------------------------------------------------------------------

def test_english_search_end_to_end(system):
    response = system.ask("Find a curriculum about data analysis")

    assert response["status"] == "success"
    assert response["request_id"] == "req-1"
    assert [c["title"] for c in curricula(response)] == [
        "Introductory Data Analysis"
    ]


def test_ai_search_end_to_end(system):
    response = system.ask("I need a training curriculum for artificial intelligence")

    assert [c["title"] for c in curricula(response)] == [
        "AI Literacy for Educators"
    ]


def test_arabic_search_end_to_end(system):
    response = system.ask("أريد منهج تدريبي لتحليل البيانات")

    assert response["status"] == "success"
    assert [c["title"] for c in curricula(response)] == [
        "Introductory Data Analysis"
    ]
    assert "العثور" in curriculum_result(response)["message"]


# --------------------------------------------------------------------
# Context propagation
# --------------------------------------------------------------------

def test_conversation_id_language_and_request_id_reach_the_agent(system):
    system.ask("Find a curriculum about data analysis", conv="conv-42")
    system.ask("أريد منهج تدريبي لتحليل البيانات", conv="conv-43")

    english, arabic = system.received

    assert english["request_id"] == "req-1"
    assert english["agent_name"] == AGENT_KEY
    assert english["context"]["conversation_id"] == "conv-42"
    assert english["context"]["language"] == "en"
    assert english["context"]["previous_results"] == []
    assert arabic["context"]["conversation_id"] == "conv-43"
    assert arabic["context"]["language"] == "ar"


def test_arabic_requests_get_arabic_messages_end_to_end(system):
    response = system.ask("أنشئ منهج عن تحليل البيانات")

    assert "تم إنشاء مسودة" in curriculum_result(response)["message"]


def test_curriculum_output_reaches_the_next_agent_in_a_workflow(system):
    response = system.ask(
        "Create a curriculum about SQL and a marketing campaign"
    )

    assert [r["agent_name"] for r in response["result"]] == [
        AGENT_KEY, "marketing"
    ]
    assert response["status"] == "success"

    previous = system.others["marketing"].requests[0]["context"][
        "previous_results"
    ]
    assert previous[0]["agent_name"] == AGENT_KEY
    assert previous[0]["result"]["curricula"][0]["title"] == (
        "SQL Training Curriculum"
    )


# --------------------------------------------------------------------
# Creation parameters
# --------------------------------------------------------------------

def test_creation_with_hours_and_audience_end_to_end(system):
    response = system.ask(
        "Create a curriculum about Python for 40 hours for beginners"
    )
    draft = curricula(response)[0]

    assert draft["title"] == "Python Training Curriculum"
    assert draft["training_hours"] == 40
    assert draft["target_audience"] == "Beginners"


def test_course_creation_phrase_without_curriculum_keyword(system):
    response = system.ask("Design a Python course for high school students")
    draft = curricula(response)[0]

    assert draft["title"] == "Python Training Curriculum"
    assert draft["target_audience"] == "High school students"


# --------------------------------------------------------------------
# Accreditation: an honest, non-success response end to end
# --------------------------------------------------------------------

def test_accreditation_request_is_reported_as_unsupported(system):
    response = system.ask("Do an accreditation gap analysis for my AI curriculum")

    assert response["status"] == "error"
    assert "not supported" in response["error_message"]
    assert response["result"][0]["result"]["code"] == "unsupported_request"
    assert "No matching examples" not in response["error_message"]


# --------------------------------------------------------------------
# Refinement follow-ups end to end
# --------------------------------------------------------------------

def test_refinement_follow_ups_reach_the_agent_and_update_the_draft(system):
    system.ask("Create a curriculum about data analysis")

    hours = system.ask("make it 40 hours")
    capstone = system.ask("Add a capstone to that curriculum")
    module = system.ask("add a module on SQL")
    audience = system.ask("change the audience to teachers")

    for response in (hours, capstone, module, audience):
        assert response["status"] == "success"
        assert response["result"][0]["agent_name"] == AGENT_KEY

    draft = curricula(audience)[0]
    assert draft["training_hours"] == 40
    assert [m["title"] for m in draft["modules"]][-2:] == [
        "Capstone Project", "SQL"
    ]
    assert draft["target_audience"] == "Teachers"
    assert draft["revision"] == 5
    assert sum(m["hours"] for m in draft["modules"]) == 40

    stored = system.store.get("conv-1", AGENT_KEY)["draft"]
    assert stored == draft


def test_arabic_follow_ups_end_to_end(system):
    system.ask("أنشئ منهج عن تحليل البيانات")

    response = system.ask("اجعلها 60 ساعة")

    assert response["status"] == "success"
    assert curricula(response)[0]["training_hours"] == 60
    assert "تم تحديث مسودة المنهج" in curriculum_result(response)["message"]


def test_follow_up_without_a_draft_gets_the_normal_routing_error(system):
    response = system.ask("make it 40 hours")

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == "orchestrator"
    assert system.received == []


def test_drafts_are_isolated_between_conversations(system):
    system.ask("Create a curriculum about data analysis", conv="A")
    system.ask("Create a curriculum about SQL", conv="B")

    system.ask("make it 40 hours", conv="A")

    assert system.store.get("A", AGENT_KEY)["draft"]["training_hours"] == 40
    assert system.store.get("B", AGENT_KEY)["draft"]["training_hours"] == 30

    # A conversation with no draft cannot refine anything.
    assert system.ask("make it 40 hours", conv="C")["status"] == "error"


@pytest.mark.parametrize("text, agent_name", [
    ("Find education tenders in Egypt", "rfp_research"),
    ("Can you recommend a course for me?", "rex_assistant"),
    ("Which course should I take?", "rex_assistant"),
    ("Generate a course participant guide", "document_generation"),
    ("Write a marketing campaign", "marketing"),
])
def test_other_agents_keep_their_requests_even_with_a_draft(
    system, text, agent_name
):
    system.ask("Create a curriculum about data analysis")
    before = system.store.get("conv-1", AGENT_KEY)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == agent_name
    assert len(system.others[agent_name].requests) == 1
    assert system.store.get("conv-1", AGENT_KEY) == before


@pytest.mark.parametrize("text", [
    "yes",
    "I'm a beginner",
    "I want to learn AI",
    "Hello there",
    "What are the modules?",
])
def test_curriculum_does_not_hijack_unrelated_follow_ups(system, text):
    system.ask("Create a curriculum about data analysis")
    calls_before = len(system.received)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == "orchestrator"
    assert len(system.received) == calls_before
    assert all(not r.requests for r in system.others.values())


def test_the_rex_conversation_state_is_never_touched(system):
    system.store.set("conv-1", "rex_assistant", {"profile": {"background": "x"}})

    system.ask("Create a curriculum about data analysis")
    system.ask("make it 40 hours")

    assert system.store.get("conv-1", "rex_assistant") == {
        "profile": {"background": "x"}
    }


# --------------------------------------------------------------------
# Through agents.register_agents.create_orchestrator
# (skipped where REX Assistant's dependencies are not installed)
# --------------------------------------------------------------------

try:
    from agents.register_agents import create_orchestrator
    _FULL_ENV_ERROR = None
except Exception as exc:  # missing optional heavy dependencies
    create_orchestrator = None
    _FULL_ENV_ERROR = f"{type(exc).__name__}: {exc}"

needs_full_environment = pytest.mark.skipif(
    create_orchestrator is None,
    reason=f"agents.register_agents cannot be imported here ({_FULL_ENV_ERROR})",
)


def _fake_rfp():
    agent = MagicMock()
    agent.AGENT_NAME = "rfp_research"
    agent.process_request.return_value = {
        "request_id": "r", "agent_name": "rfp_research",
        "status": "success", "result": {"ok": True}, "error_message": None,
    }
    return agent


@needs_full_environment
def test_create_orchestrator_registers_curriculum_with_the_shared_store(tmp_path):
    shared = ConversationStore(db_path=tmp_path / "conv.sqlite3")

    orchestrator = create_orchestrator(
        rex_assistant_agent=MagicMock(),
        rfp_research_agent=_fake_rfp(),
        conversation_store=shared,
    )

    agent = orchestrator.curriculum_research_agent
    assert orchestrator.agents[AGENT_KEY].__self__ is agent
    assert agent.conversation_store is shared
    assert orchestrator.conversation_store is shared
    assert AGENT_KEY in orchestrator._continuation_checks
    assert {"rex_assistant", "rfp_research"} <= set(orchestrator.agents)


@needs_full_environment
def test_create_orchestrator_end_to_end_refinement(tmp_path):
    shared = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    orchestrator = create_orchestrator(
        rex_assistant_agent=MagicMock(),
        rfp_research_agent=_fake_rfp(),
        conversation_store=shared,
    )

    def ask(text):
        return orchestrator.handle_request(
            text, context={"conversation_id": "conv-9"}
        )

    created = ask("Create a curriculum about SQL for 20 hours")
    refined = ask("make it 40 hours")

    assert created["result"][0]["result"]["curricula"][0]["training_hours"] == 20
    assert refined["result"][0]["result"]["curricula"][0]["training_hours"] == 40
    assert shared.get("conv-9", AGENT_KEY)["draft"]["training_hours"] == 40


@needs_full_environment
def test_create_orchestrator_keeps_rex_and_rfp_routing(tmp_path):
    rex = MagicMock()
    rex.handle_message.return_value = {
        "request_id": "r", "agent_name": "REX Assistant", "status": "success",
        "result": {"answer": "Try AI Foundations."}, "error_message": None,
        "state": {},
    }
    orchestrator = create_orchestrator(
        rex_assistant_agent=rex,
        rfp_research_agent=_fake_rfp(),
        conversation_store=ConversationStore(db_path=tmp_path / "c.sqlite3"),
    )

    recommend = orchestrator.handle_request("Can you recommend a course for me?")
    tenders = orchestrator.handle_request("Find education tenders in Egypt")

    assert recommend["result"][0]["agent_name"] == "rex_assistant"
    assert tenders["result"][0]["agent_name"] == "rfp_research"
