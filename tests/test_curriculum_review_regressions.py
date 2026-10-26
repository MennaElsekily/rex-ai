"""
Regression tests added during the pre-commit review of the Curriculum
Research changes. Each test pins a bug that was found by probing the
first implementation:

- searches / other agents' answers were mistaken for edits of the draft
- an edit that mentions "creating a course" was treated as creation
- translate / export / compare returned an empty "success"
- "5 hours per week" was read as a total and leaked into the title
- instructions after the topic leaked into the title
"""

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research import CurriculumResearchAgent
from agents.curriculum_research.registration import (
    register_curriculum_research,
)
from agents.orchestrator import Orchestrator

AGENT_KEY = "curriculum_research"


def req(text, conv="conv-1", language=None):
    context = {}
    if conv is not None:
        context["conversation_id"] = conv
    if language is not None:
        context["language"] = language
    return {"request_id": "req-1", "user_input": text, "context": context}


@pytest.fixture
def store():
    return ConversationStore(db_path=":memory:")


@pytest.fixture
def agent():
    return CurriculumResearchAgent()


@pytest.fixture
def stateful(store):
    return CurriculumResearchAgent(conversation_store=store)


def create(agent, text="Create a curriculum about data analysis", conv="conv-1"):
    return agent.process_request(req(text, conv=conv))


def draft_of(store, conv="conv-1"):
    return store.get(conv, AGENT_KEY)["draft"]


def titles(response):
    return [c["title"] for c in response["result"]["curricula"]]


# --------------------------------------------------------------------
# Edits vs creation, searches and other agents' answers
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "add a module on how to design a course",
    "add a module on how to create a course about SQL",
])
def test_an_edit_that_mentions_creating_is_still_an_edit(stateful, store, text):
    create(stateful)

    response = stateful.process_request(req(text))
    draft = draft_of(store)

    assert response["status"] == "success"
    assert draft["title"] == "data analysis Training Curriculum"  # not replaced
    assert draft["modules"][-1]["title"] == text.removeprefix("add a module on ")
    assert draft["revision"] == 2


def test_make_a_curriculum_still_creates_a_new_one_even_with_a_draft(
    stateful, store
):
    create(stateful)

    stateful.process_request(req("Make a curriculum about SQL for 50 hours"))

    assert draft_of(store)["title"] == "SQL Training Curriculum"
    assert draft_of(store)["training_hours"] == 50


@pytest.mark.parametrize("text", [
    "Show me a curriculum with a capstone",
    "I want a curriculum with a capstone project",
    "Find a curriculum about cybersecurity and add a capstone",
    "About 10 hours",
    "5 hours per week",
])
def test_searches_and_other_answers_never_change_the_draft(
    stateful, store, text
):
    create(stateful)
    before = draft_of(store)

    response = stateful.process_request(req(text))

    assert draft_of(store) == before
    assert stateful.has_refinable_draft(req(text)) is False
    assert "Updated your curriculum draft" not in str(response["result"])


def test_a_module_title_may_mention_accreditation_but_alignment_requests_do_not_pass(
    stateful, store
):
    create(stateful)

    allowed = stateful.process_request(req("add a module on accreditation"))
    refused = stateful.process_request(
        req("Align this curriculum with the standards")
    )

    assert allowed["status"] == "success"
    assert draft_of(store)["modules"][-1]["title"] == "accreditation"
    assert refused["status"] == "error"
    assert refused["result"]["capability"] == "accreditation_gap_analysis"


def test_arabic_edit_with_pronoun_and_polite_prefix(stateful, store):
    stateful.process_request(req("أنشئ منهج عن تحليل البيانات", language="ar"))

    response = stateful.process_request(
        req("من فضلك اجعلها 40 ساعة", language="ar")
    )

    assert response["status"] == "success"
    assert draft_of(store)["training_hours"] == 40


# --------------------------------------------------------------------
# Unsupported requests are never a misleading success
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, arabic", [
    ("translate this curriculum to Arabic", False),
    ("Export this curriculum as PDF", False),
    ("Compare these two curricula", False),
    ("Summarize this curriculum", False),
    ("ترجم هذا المنهج إلى الإنجليزية", True),
    ("صدرها بصيغة PDF", True),
    ("قارن بين المنهجين", True),
])
def test_unsupported_actions_are_never_a_success(agent, stateful, text, arabic):
    for candidate in (agent, stateful):  # with and without a store
        response = candidate.process_request(
            req(text, language="ar" if arabic else "en")
        )

        assert response["status"] == "error"
        assert response["result"] == {
            "code": "unsupported_request",
            "capability": "unsupported_action",
        }
        assert "No matching examples" not in response["error_message"]
        assert "لم يتم العثور" not in response["error_message"]

        if arabic:
            assert "غير مدعومة" in response["error_message"]
        else:
            assert "not supported" in response["error_message"]


@pytest.mark.parametrize("text", [
    "Find a curriculum about translation",
    "Create a curriculum about export management",
])
def test_topics_that_sound_like_actions_still_work(agent, text):
    assert agent.process_request(req(text))["status"] == "success"


# --------------------------------------------------------------------
# Creation parameters
# --------------------------------------------------------------------

def test_hours_per_week_are_not_taken_as_total_hours(agent):
    response = agent.process_request(
        req("Create a curriculum about Python, 5 hours per week")
    )
    draft = response["result"]["curricula"][0]

    assert draft["title"] == "Python Training Curriculum"
    assert draft["training_hours"] == 30
    assert "not supported, so 30 total training hours" in (
        response["result"]["message"]
    )


def test_conflicting_instruction_does_not_pollute_the_title(agent):
    draft = agent.process_request(req(
        "Create a curriculum about Python for 40 hours and make it 50 hours"
    ))["result"]["curricula"][0]

    assert draft["title"] == "Python Training Curriculum"
    assert draft["training_hours"] == 40


# --------------------------------------------------------------------
# A change is applied completely or not at all
# --------------------------------------------------------------------

def test_a_partly_failing_message_changes_nothing(stateful, store):
    create(stateful)
    before = draft_of(store)

    response = stateful.process_request(req(
        "change the audience to teachers and add a capstone and make it 2 hours"
    ))

    assert response["status"] == "error"
    assert draft_of(store) == before
    assert draft_of(store)["target_audience"] == "Beginners"
    assert len(draft_of(store)["modules"]) == 4


def test_a_rejected_change_does_not_affect_the_next_one(stateful, store):
    create(stateful)
    stateful.process_request(req("make it 2 hours"))  # rejected

    response = stateful.process_request(req("make it 40 hours"))

    assert response["result"]["curricula"][0]["training_hours"] == 40
    assert draft_of(store)["revision"] == 2  # only one real change


def test_the_stored_draft_is_not_shared_with_the_returned_one(stateful, store):
    create(stateful)
    response = stateful.process_request(req("make it 40 hours"))

    response["result"]["curricula"][0]["modules"].clear()  # caller mutates

    assert len(draft_of(store)["modules"]) == 4


def test_a_corrupted_stored_draft_gives_an_error_not_a_crash(stateful, store):
    broken = {"draft": {"modules": "not-a-list"}}
    store.set("conv-1", AGENT_KEY, broken)

    response = stateful.process_request(req("make it 40 hours"))

    assert response["status"] == "error"
    assert response["error_message"]
    assert store.get("conv-1", AGENT_KEY) == broken


@pytest.mark.parametrize("state", [
    {"draft": "not-a-dict"},
    {"draft": None},
    {"something_else": 1},
    {},
])
def test_stored_state_without_a_draft_is_ignored(stateful, store, state):
    store.set("conv-1", AGENT_KEY, state)

    assert stateful.has_refinable_draft(req("make it 40 hours")) is False
    response = stateful.process_request(req("make it 40 hours"))
    assert response["status"] == "success"
    assert titles(response) == []


@pytest.mark.parametrize("conversation_id", ["", "   ", None])
def test_blank_conversation_ids_get_no_memory(stateful, store, conversation_id):
    request = req("Create a curriculum about SQL", conv=None)
    request["context"]["conversation_id"] = conversation_id

    response = stateful.process_request(request)

    assert response["status"] == "success"
    assert "refine this draft" not in response["result"]["message"]
    count = store._connection.execute(
        "SELECT COUNT(*) FROM conversation_state"
    ).fetchone()
    assert count == (0,)


# --------------------------------------------------------------------
# End to end through the real orchestrator and registration
# --------------------------------------------------------------------

class Recorder:
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
def system(store):
    orchestrator = Orchestrator()
    register_curriculum_research(orchestrator, store)
    others = {n: Recorder(n) for n in (
        "rex_assistant", "rfp_research", "marketing", "document_generation",
    )}
    for name, recorder in others.items():
        orchestrator.register_agent(name, recorder)

    received = []
    real = orchestrator.agents[AGENT_KEY]

    def spy(request):
        received.append(request)
        return real(request)

    orchestrator.agents[AGENT_KEY] = spy

    class System:
        pass

    result = System()
    result.store = store
    result.received = received
    result.others = others
    result.ask = lambda text, conv="conv-1": orchestrator.handle_request(
        text, context={"conversation_id": conv}, request_id="req-1"
    )
    return result


def curricula(response):
    return response["result"][0]["result"]["curricula"]


@pytest.mark.parametrize("text", [
    "5 hours per week",
    "About 10 hours",
    "I have 10 hours",
    "I can study 20 hours a week",
    "٥ ساعات في الأسبوع",
    "نعم",
    "no",
])
def test_answers_meant_for_other_agents_are_not_claimed_by_a_draft(system, text):
    system.ask("Create a curriculum about data analysis")
    before = system.store.get("conv-1", AGENT_KEY)
    calls = len(system.received)

    response = system.ask(text)

    assert response["result"][0]["agent_name"] == "orchestrator"
    assert len(system.received) == calls
    assert system.store.get("conv-1", AGENT_KEY) == before


@pytest.mark.parametrize("text", [
    "translate this curriculum to Arabic",
    "Export this curriculum as PDF",
    "Compare these two curricula",
    "ترجم هذا المنهج إلى الإنجليزية",
    "قارن بين المنهجين",
])
def test_unsupported_actions_end_to_end_are_errors_not_empty_successes(
    system, text
):
    system.ask("Create a curriculum about data analysis")
    before = system.store.get("conv-1", AGENT_KEY)

    response = system.ask(text)

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == AGENT_KEY
    assert response["result"][0]["status"] == "error"
    assert "No matching examples" not in response["error_message"]
    assert "لم يتم العثور" not in response["error_message"]
    assert system.store.get("conv-1", AGENT_KEY) == before


def test_unsupported_actions_error_even_without_a_draft(system):
    response = system.ask("Export this curriculum as PDF", conv="fresh")

    assert response["status"] == "error"
    assert "can't do that yet" in response["error_message"]


def test_accreditation_in_arabic_is_an_error_not_a_success(system):
    response = system.ask("أريد تحليل الفجوات لمنهج الأمن السيبراني")

    assert response["status"] == "error"
    assert "غير مدعوم" in response["error_message"]
    assert "لم يتم العثور" not in response["error_message"]


def test_an_edit_that_mentions_creating_edits_the_draft_end_to_end(system):
    system.ask("Create a curriculum about data analysis")

    response = system.ask("add a module on how to design a course")

    draft = curricula(response)[0]
    assert draft["title"] == "data analysis Training Curriculum"
    assert draft["modules"][-1]["title"] == "how to design a course"
    assert system.store.get("conv-1", AGENT_KEY)["draft"] == draft


def test_polite_arabic_follow_up_reaches_the_agent(system):
    system.ask("أنشئ منهج عن تحليل البيانات")

    response = system.ask("من فضلك اجعلها 40 ساعة")

    assert response["result"][0]["agent_name"] == AGENT_KEY
    assert curricula(response)[0]["training_hours"] == 40


def test_search_wording_with_a_draft_still_searches(system):
    system.ask("Create a curriculum about data analysis")
    before = system.store.get("conv-1", AGENT_KEY)

    response = system.ask("Show me a curriculum with a capstone")

    assert response["status"] == "success"
    assert all("example_id" in c for c in curricula(response))
    assert system.store.get("conv-1", AGENT_KEY) == before
