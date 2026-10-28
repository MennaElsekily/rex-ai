"""
Integration tests: the REAL Curriculum Agent (Agent 3), registered through
the real registration code, run through the real orchestrator and the
shared ConversationStore. The agent uses its real tools in-process on
temporary storage (tests/conftest.py) and a stand-in model (FakeLLM).
Other agents are recording stand-ins, used to check routing priority and
that nothing else is disturbed.

Rewritten for the rebuilt agent (Phase 5); the behaviours of the old
integration tests that still apply are kept (registration contract,
context propagation, Arabic replies, workflow hand-off, follow-ups,
isolation between conversations, no hijacking of other agents' turns).
"""

from unittest.mock import MagicMock

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research.curriculum_agent import CurriculumAgent
from agents.curriculum_research.registration import register_curriculum_research
from agents.orchestrator import Orchestrator
from tests.curriculum_fakes import FakeLLM, canned_answers

AGENT_KEY = "curriculum_research"
FULL_REQUEST = "Create a beginner curriculum about data analysis, 40 hours"
FULL_REQUEST_AR = "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين"


class Recorder:
    """Stand-in agent that records the requests it receives."""

    def __init__(self, name):
        self.name = name
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return {"request_id": request["request_id"], "agent_name": self.name, "status": "success",
                "result": {"handled_by": self.name}, "error_message": None}


@pytest.fixture
def store():
    return ConversationStore(db_path=":memory:")


@pytest.fixture
def system(store):
    orchestrator = Orchestrator()
    agent = register_curriculum_research(orchestrator, store)
    agent._llm = FakeLLM({k: [v] * 10 for k, v in canned_answers().items()})

    others = {name: Recorder(name) for name in ("rex_assistant", "rfp_research", "marketing", "document_generation")}
    for name, recorder in others.items():
        orchestrator.register_agent(name, recorder)

    received = []
    real_handler = orchestrator.agents[AGENT_KEY]

    def spy(request):
        received.append(request)
        return real_handler(request)

    orchestrator.agents[AGENT_KEY] = spy

    class System:
        pass

    result = System()
    result.orchestrator, result.agent, result.others, result.received, result.store = (
        orchestrator, agent, others, received, store)

    def ask(text, conv="conv-1"):
        return orchestrator.handle_request(text, context={"conversation_id": conv}, request_id="req-1")

    result.ask = ask
    return result


def curriculum_result(response):
    assert response["result"][0]["agent_name"] == AGENT_KEY
    return response["result"][0]["result"]


def stored(system, response):
    result = curriculum_result(response)
    return system.agent.tools.store.get(result["curriculum_id"])


# --------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------

def test_registration_connects_the_new_agent_and_the_shared_store(store):
    orchestrator = Orchestrator()

    agent = register_curriculum_research(orchestrator, store)

    assert isinstance(agent, CurriculumAgent)
    handler = orchestrator.agents[AGENT_KEY]
    assert handler.__self__ is agent and handler.__func__ is CurriculumAgent.process_request
    assert agent.conversation_store is store
    assert orchestrator._continuation_checks[AGENT_KEY].__self__ is agent


def test_registration_starts_no_server_and_loads_no_model(store):
    agent = register_curriculum_research(Orchestrator(), store)
    assert agent._tools is None and agent._llm is None and agent._graph is None


def test_registration_without_a_store_still_works():
    orchestrator = Orchestrator()
    register_curriculum_research(orchestrator)

    response = orchestrator.handle_request("Create a curriculum about AI")

    assert response["status"] == "success"
    assert response["result"][0]["result"]["kind"] == "question"


# --------------------------------------------------------------------
# Generation through the orchestrator (English and Arabic)
# --------------------------------------------------------------------

def test_english_generation_end_to_end(system):
    response = system.ask(FULL_REQUEST)

    assert response["status"] == "success" and response["request_id"] == "req-1"
    result = curriculum_result(response)
    assert result["kind"] == "curriculum" and result["review_status"] == "pending_review"
    assert stored(system, response)["content"]["total_hours"] == 40


def test_arabic_generation_end_to_end(system):
    system.agent._llm = FakeLLM(canned_answers("ar"))

    response = system.ask(FULL_REQUEST_AR)

    assert "تم إعداد مسودة المنهج" in curriculum_result(response)["message"]
    assert stored(system, response)["content"]["language"] == "ar"


def test_course_creation_phrase_without_curriculum_keyword(system):
    response = system.ask("Design an intermediate Python course for high school students, 30 hours")

    assert curriculum_result(response)["kind"] == "curriculum"
    assert stored(system, response)["content"]["level"] == "Intermediate"


def test_conversation_id_language_and_request_id_reach_the_agent(system):
    system.ask(FULL_REQUEST, conv="conv-42")
    system.ask("أنشئ منهج عن تحليل البيانات", conv="conv-43")

    english, arabic = system.received
    assert english["request_id"] == "req-1" and english["agent_name"] == AGENT_KEY
    assert english["context"]["conversation_id"] == "conv-42" and english["context"]["language"] == "en"
    assert english["context"]["previous_results"] == []
    assert arabic["context"]["conversation_id"] == "conv-43" and arabic["context"]["language"] == "ar"


def test_curriculum_output_reaches_the_next_agent_in_a_workflow(system):
    response = system.ask("Create a beginner curriculum about SQL, 30 hours, and a marketing campaign")

    assert [r["agent_name"] for r in response["result"]] == [AGENT_KEY, "marketing"]
    previous = system.others["marketing"].requests[0]["context"]["previous_results"]
    assert previous[0]["agent_name"] == AGENT_KEY
    assert previous[0]["result"]["curriculum"]["title"] == "Cloud Computing Essentials"  # the (fake) model's title
    assert previous[0]["result"]["review_status"] == "pending_review"


# --------------------------------------------------------------------
# Catalogue planning (Workflow B) is routed to this agent
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Suggest new courses for the Data Analytics department",
                                  "Review the catalogue and identify missing learning paths",
                                  "اقترح دورات جديدة للأكاديمية"])
def test_catalogue_planning_reaches_the_agent(system, text):
    system.agent._llm = FakeLLM({"SuggestionList": {"suggestions": []}})

    response = system.ask(text)

    assert curriculum_result(response)["kind"] == "suggestions"
    assert not system.others["rex_assistant"].requests


# --------------------------------------------------------------------
# Follow-ups end to end
# --------------------------------------------------------------------

def test_a_question_and_its_answer_reach_the_agent(system):
    question = system.ask("Create a curriculum about data analysis")
    assert curriculum_result(question)["missing"] == ["total_hours", "level"]

    answer = system.ask("40 hours, intermediate")

    result = curriculum_result(answer)
    assert result["kind"] == "curriculum"
    assert stored(system, answer)["content"]["level"] == "Intermediate"


def test_revisions_and_alignment_requests_reach_the_agent(system):
    created = system.ask(FULL_REQUEST)
    curriculum_id = curriculum_result(created)["curriculum_id"]

    hours = system.ask("make it 30 hours")
    capstone = system.ask("add a capstone project")
    align = system.ask("Align it with the NICE framework")
    approve = system.ask("approve it")

    assert curriculum_result(hours)["version"] == 2
    assert curriculum_result(capstone)["kind"] == "no_change"  # it already has one
    assert curriculum_result(align)["alignment_status"] == "unable_to_determine"  # no approved framework
    assert approve["status"] == "error" and "administrator" in approve["error_message"]
    assert system.agent.tools.store.get(curriculum_id)["status"] == "pending_review"
    memory = system.store.get("conv-1", AGENT_KEY)
    assert memory["active_curriculum_id"] == curriculum_id and memory["active_version"] == 2


def test_arabic_follow_up_end_to_end(system):
    system.agent._llm = FakeLLM(canned_answers("ar"))
    system.ask(FULL_REQUEST_AR)

    response = system.ask("اجعلها 60 ساعة")

    assert "تم تعديل المنهج" in curriculum_result(response)["message"]
    assert stored(system, response)["content"]["total_hours"] == 60


def test_follow_up_without_state_gets_the_normal_routing_error(system):
    response = system.ask("make it 40 hours")

    assert response["status"] == "error" and response["result"][0]["agent_name"] == "orchestrator"
    assert system.received == []


def test_conversations_are_isolated(system):
    a = curriculum_result(system.ask(FULL_REQUEST, conv="A"))["curriculum_id"]
    b = curriculum_result(system.ask(FULL_REQUEST, conv="B"))["curriculum_id"]

    system.ask("make it 30 hours", conv="A")

    store = system.agent.tools.store
    assert store.get(a)["content"]["total_hours"] == 30 and store.get(b)["content"]["total_hours"] == 40
    assert system.ask("make it 30 hours", conv="C")["status"] == "error"


@pytest.mark.parametrize("text, agent_name", [
    ("Find education tenders in Egypt", "rfp_research"),
    ("Can you recommend a course for me?", "rex_assistant"),
    ("Recommend new courses for me", "rex_assistant"),
    ("Which course should I take?", "rex_assistant"),
    ("Generate a course participant guide", "document_generation"),
    ("Write a marketing campaign", "marketing"),
])
def test_other_agents_keep_their_requests_even_with_a_draft(system, text, agent_name):
    system.ask(FULL_REQUEST)
    before = system.store.get("conv-1", AGENT_KEY)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == agent_name
    assert len(system.others[agent_name].requests) == 1
    assert system.store.get("conv-1", AGENT_KEY) == before


@pytest.mark.parametrize("text", ["yes", "I'm a beginner", "Hello there", "What are the modules?"])
def test_curriculum_does_not_hijack_unrelated_follow_ups(system, text):
    system.ask(FULL_REQUEST)  # complete: no question is open
    calls_before = len(system.received)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == "orchestrator"
    assert len(system.received) == calls_before
    assert all(not r.requests for r in system.others.values())


@pytest.mark.parametrize("text", ["I want to learn AI", "عايز أتعلم الذكاء الاصطناعي"])
def test_a_learning_goal_after_a_draft_goes_to_rex_not_curriculum(system, text):
    system.ask(FULL_REQUEST)
    calls_before = len(system.received)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == "rex_assistant"
    assert len(system.received) == calls_before


def test_the_rex_conversation_state_is_never_touched(system):
    system.store.set("conv-1", "rex_assistant", {"profile": {"background": "x"}})

    system.ask(FULL_REQUEST)
    system.ask("make it 30 hours")

    assert system.store.get("conv-1", "rex_assistant") == {"profile": {"background": "x"}}


def test_blank_or_missing_conversation_ids_get_no_memory(store):
    agent = CurriculumAgent(conversation_store=store, llm=FakeLLM(canned_answers()))
    for context in ({}, {"conversation_id": "   "}, {"conversation_id": None}):
        agent.process_request({"request_id": "r", "user_input": "Create a curriculum about SQL", "context": context})
        assert not agent.wants_follow_up({"user_input": "40 hours", "context": context})


def test_conversation_store_failures_do_not_break_requests():
    broken = MagicMock()
    broken.get.side_effect = RuntimeError("database locked")
    broken.set.side_effect = RuntimeError("database locked")
    agent = CurriculumAgent(conversation_store=broken, llm=FakeLLM(canned_answers()))

    response = agent.process_request({"request_id": "r", "user_input": FULL_REQUEST,
                                      "context": {"conversation_id": "c"}})

    assert response["status"] == "success" and response["result"]["kind"] == "curriculum"


def test_a_stale_active_curriculum_is_ignored(store):
    store.set("c", AGENT_KEY, {"active_curriculum_id": "CUR-000000000000", "active_version": 3})
    agent = CurriculumAgent(conversation_store=store, llm=FakeLLM({}))

    response = agent.process_request({"request_id": "r", "user_input": "hello", "context": {"conversation_id": "c"}})

    assert response["result"]["kind"] == "help"


# --------------------------------------------------------------------
# Through agents.register_agents
# --------------------------------------------------------------------

from agents.register_agents import create_admin_orchestrator, create_orchestrator  # noqa: E402


def _fake_rfp():
    agent = MagicMock()
    agent.AGENT_NAME = "rfp_research"
    agent.process_request.return_value = {"request_id": "r", "agent_name": "rfp_research", "status": "success",
                                          "result": {"ok": True}, "error_message": None}
    return agent


def test_create_orchestrator_registers_curriculum_with_the_shared_store(tmp_path):
    shared = ConversationStore(db_path=tmp_path / "conv.sqlite3")

    orchestrator = create_orchestrator(rex_assistant_agent=MagicMock(), rfp_research_agent=_fake_rfp(),
                                       conversation_store=shared)

    agent = orchestrator.curriculum_research_agent
    assert isinstance(agent, CurriculumAgent) and orchestrator.agents[AGENT_KEY].__self__ is agent
    assert agent.conversation_store is shared and AGENT_KEY in orchestrator._continuation_checks


def test_create_orchestrator_end_to_end_revision(tmp_path):
    shared = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    orchestrator = create_orchestrator(rex_assistant_agent=MagicMock(), rfp_research_agent=_fake_rfp(),
                                       conversation_store=shared)
    orchestrator.curriculum_research_agent._llm = FakeLLM(canned_answers())

    created = orchestrator.handle_request(FULL_REQUEST, context={"conversation_id": "conv-9"})
    revised = orchestrator.handle_request("make it 30 hours", context={"conversation_id": "conv-9"})

    assert created["result"][0]["result"]["version"] == 1
    assert revised["result"][0]["result"]["version"] == 2


def test_the_admin_orchestrator_routes_planning_to_agent_3_and_keeps_rex_out(tmp_path):
    orchestrator = create_admin_orchestrator(rfp_research_agent=_fake_rfp(),
                                             conversation_store=ConversationStore(tmp_path / "c.sqlite3"))
    orchestrator.curriculum_research_agent._llm = FakeLLM({"SuggestionList": {"suggestions": []}})

    planning = orchestrator.handle_request("Suggest new courses for the AI track")
    learner = orchestrator.handle_request("Can you recommend a course for me?")

    assert planning["result"][0]["agent_name"] == AGENT_KEY
    assert learner["status"] == "error" and "REX Assistant" in learner["error_message"]


def test_a_suggestion_picked_through_the_admin_orchestrator_is_generated(tmp_path):
    """Regression (Phase 6 live check): "generate suggestion 1" reached REX's rule and was refused."""
    orchestrator = create_admin_orchestrator(rfp_research_agent=_fake_rfp(),
                                             conversation_store=ConversationStore(tmp_path / "c.sqlite3"))
    suggestion = {"suggestions": [{
        "title": "Advanced Data Analytics", "department": "Data Analytics", "level": "Advanced",
        "target_audience": "Analysts", "prerequisite_course_ids": [], "outcomes": ["Build models"],
        "proposed_hours": 40, "modules_outline": ["Modelling"], "rationale": "No Advanced course.",
        "gap_ids": ["G1"], "source_ids": [], "confidence": "medium"}]}
    orchestrator.curriculum_research_agent._llm = FakeLLM({**canned_answers(), "SuggestionList": suggestion})
    context = {"conversation_id": "plan-1"}

    orchestrator.handle_request("Suggest new courses for the Data Analytics department", context=context)
    response = orchestrator.handle_request("generate suggestion 1", context=context)

    assert response["result"][0]["agent_name"] == AGENT_KEY
    assert response["result"][0]["result"]["kind"] == "curriculum"
