"""
Routing by what a request asks FOR: a subject ("a curriculum about digital marketing") is not a marketing request, a sentence
with two deliverables runs both agents, and a follow-up continues its own workflow. No model, no database.
"""

from unittest.mock import MagicMock

import pytest

from agents import routing
from agents.orchestrator import Orchestrator

CID = "CUR-0123456789ab"
CURRICULUM, DOCUMENT, MARKETING, RFP, REX = ("curriculum_research", "document_generation", "marketing", "rfp_research", "rex_assistant")


@pytest.fixture
def router():
    return Orchestrator()


# ---- the cases of the audit -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, plan", [
    # curriculum only: "marketing" is the subject
    ("Create a curriculum about digital marketing.", [CURRICULUM]),
    ("Create a curriculum about marketing analytics.", [CURRICULUM]),
    ("Design a course on social media advertising", [CURRICULUM]),
    ("Create a marketing curriculum", [CURRICULUM]),
    ("Create a course about marketing analytics", [CURRICULUM]),
    ("Design a course on how to write a marketing campaign", [CURRICULUM]),
    ("أنشئ منهج عن التسويق الرقمي", [CURRICULUM]),
    ("Create a curriculum about SQL with an assessment plan", [CURRICULUM]),
    ("Create a curriculum about SQL for business analysts, 24 hours", [CURRICULUM]),
    # marketing only
    (f"Create a social media campaign for approved course {CID}.", [MARKETING]),
    (f"Write LinkedIn posts promoting approved course {CID}.", [MARKETING]),
    (f"Write a marketing campaign for the curriculum {CID}", [MARKETING]),
    ("Write a post on LinkedIn about the course", [MARKETING]),
    ("Write a marketing campaign", [MARKETING]),
    ("Draft ad copy and WhatsApp messages for the data course", [MARKETING]),
    ("Write promotional messages for the new course", [MARKETING]),
    ("اكتب حملة تسويقية", [MARKETING]),
    (f"اكتب حملة تسويقية للمنهج {CID}", [MARKETING]),
    # document only
    (f"Create a course description for curriculum {CID}.", [DOCUMENT]),
    (f"Prepare a course description for the marketing curriculum {CID}", [DOCUMENT]),
    (f"Generate a participant guide for {CID}", [DOCUMENT]),
    ("Create a course question bank", [DOCUMENT]),
    ("Generate a course participant guide", [DOCUMENT]),
    ("Create a course proposal", [DOCUMENT]),
    # more than one deliverable: every agent that was asked for runs, in the order curriculum, documents, marketing
    (f"Generate a question bank for {CID} and create a marketing campaign for the same course.", [DOCUMENT, MARKETING]),
    (f"Create a marketing campaign for {CID} and generate a question bank", [DOCUMENT, MARKETING]),
    ("Create a curriculum about data analysis and prepare a course proposal.", [CURRICULUM, DOCUMENT]),
    ("Create a curriculum and a course proposal", [CURRICULUM, DOCUMENT]),
    ("Create a curriculum and a marketing campaign", [CURRICULUM, MARKETING]),
    ("Create a course about SQL and a marketing campaign", [CURRICULUM, MARKETING]),
    ("Create a beginner curriculum about SQL, 30 hours, and a marketing campaign", [CURRICULUM, MARKETING]),
    ("Create a curriculum about digital marketing and a campaign for it", [CURRICULUM, MARKETING]),
    ("Create a curriculum about marketing and a marketing campaign for it", [CURRICULUM, MARKETING]),
    ("أنشئ منهج تدريبي ثم حملة تسويقية", [CURRICULUM, MARKETING]),
    ("أنشئ منهج عن تحليل البيانات وحملة تسويقية", [CURRICULUM, MARKETING]),
    # the other agents are unchanged
    ("Find education tenders in Egypt", [RFP]),
    ("Find tenders about marketing training in Egypt", [RFP]),
    ("ابحث عن مناقصات تعليمية", [RFP]),
    ("Recommend a course for beginners", [REX]),
    ("أريد دورة في البرمجة للمبتدئين", [REX]),
    ("Make a PDF of the curriculum", [CURRICULUM]),
    ("Suggest new courses for our data track", [CURRICULUM]),
    ("Create a curriculum about SQL, 30 hours", [CURRICULUM]),
])
def test_routing_by_deliverable(router, text, plan):
    assert router._make_plan(text) == plan


@pytest.mark.parametrize("text", [
    "Tell me about marketing", "What is marketing?", "Hello there", "yes", "make it 40 hours", "How much is a pizza",
    "I study marketing", "Marketing is fun", "Find me a marketing course to take",
])
def test_the_word_marketing_alone_routes_nothing_to_marketing(router, text):
    assert MARKETING not in router._make_plan(text)


@pytest.mark.parametrize("text", ["Hello there", "yes", "make it 40 hours", "Add a capstone project", "How much is a pizza"])
def test_messages_without_a_request_still_have_no_plan(router, text):
    assert router._make_plan(text) == []


def test_no_explicitly_requested_agent_is_dropped_and_none_is_added(router):
    plan = router._make_plan(f"Generate a question bank for {CID} and create a marketing campaign for the same course")
    assert plan == [DOCUMENT, MARKETING] and CURRICULUM not in plan and RFP not in plan
    assert router._make_plan("Create a curriculum about SQL and statistics") == [CURRICULUM]


def test_routing_does_not_depend_on_a_course_id(router):
    with_id = router._make_plan(f"Write a marketing campaign for {CID}")
    without = router._make_plan("Write a marketing campaign for the SQL course")
    assert with_id == without == [MARKETING]
    assert router._make_plan(f"Create a curriculum about marketing, reference {CID}") == [CURRICULUM]
    assert not hasattr(__import__("agents.orchestrator", fromlist=["x"]), "_CURRICULUM_ID")


def test_the_deliverable_reader_is_in_its_own_module_without_importing_the_orchestrator():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(routing.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.startswith("agents.orchestrator") for m in imported)           # no circular import


# ---- executing a plan: no agent that was not asked for runs ----------------------------------------------------------------------

def make_orchestrator(*names):
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    handlers = {}
    for name in names:
        handlers[name] = MagicMock(return_value={"request_id": "r", "agent_name": name, "status": "success", "result": {"kind": name}})
        orchestrator.register_agent(name, handlers[name])
    return orchestrator, handlers


def test_a_curriculum_only_request_does_not_run_marketing():
    orchestrator, handlers = make_orchestrator(CURRICULUM, DOCUMENT, MARKETING)
    response = orchestrator.handle_request("Create a curriculum about digital marketing for analysts")

    assert [r["agent_name"] for r in response["result"]] == [CURRICULUM]
    handlers[MARKETING].assert_not_called()
    handlers[DOCUMENT].assert_not_called()


def test_two_deliverables_run_both_agents_in_order_with_the_earlier_results_shared():
    orchestrator, handlers = make_orchestrator(CURRICULUM, DOCUMENT, MARKETING)
    response = orchestrator.handle_request(f"Generate a question bank for {CID} and create a marketing campaign for the same course")

    assert [r["agent_name"] for r in response["result"]] == [DOCUMENT, MARKETING]
    handlers[CURRICULUM].assert_not_called()
    previous = handlers[MARKETING].call_args.args[0]["context"]["previous_results"]
    assert [r["agent_name"] for r in previous] == [DOCUMENT]


def test_the_marketing_tab_request_still_routes_to_marketing_alone():
    orchestrator, handlers = make_orchestrator(CURRICULUM, DOCUMENT, MARKETING)
    orchestrator.handle_request("Create a curriculum and a question bank", {"marketing_request": {"curriculum_id": CID}})
    handlers[MARKETING].assert_called_once()
    handlers[CURRICULUM].assert_not_called()
    handlers[DOCUMENT].assert_not_called()


def test_the_document_form_request_still_routes_to_documents_alone():
    orchestrator, handlers = make_orchestrator(CURRICULUM, DOCUMENT, MARKETING)
    orchestrator.handle_request("Write a marketing campaign", {"document_request": {"curriculum_id": CID, "document_type": "course_description"}})
    handlers[DOCUMENT].assert_called_once()
    handlers[MARKETING].assert_not_called()


# ---- continuation --------------------------------------------------------------------------------------------------------------

def test_a_follow_up_continues_the_open_workflow_and_starts_no_other_agent():
    orchestrator, handlers = make_orchestrator(DOCUMENT, MARKETING)
    open_conversation = MagicMock(return_value={"request_id": "r", "agent_name": CURRICULUM, "status": "success", "result": {"kind": "draft"}})
    orchestrator.register_agent(CURRICULUM, open_conversation, continuation_check=lambda probe: probe["user_input"] == "make it 30 hours")

    response = orchestrator.handle_request("make it 30 hours", {"conversation_id": "c1"})

    assert [r["agent_name"] for r in response["result"]] == [CURRICULUM]
    handlers[MARKETING].assert_not_called()
    handlers[DOCUMENT].assert_not_called()


def test_an_explicit_request_still_wins_over_a_continuation():
    orchestrator, handlers = make_orchestrator(CURRICULUM, MARKETING)
    asked = MagicMock(return_value={"request_id": "r", "agent_name": "document_generation", "status": "success", "result": {}})
    orchestrator.register_agent(DOCUMENT, asked, continuation_check=lambda probe: True)

    response = orchestrator.handle_request("Write a marketing campaign for the SQL course", {"conversation_id": "c1"})

    assert [r["agent_name"] for r in response["result"]] == [MARKETING]
    asked.assert_not_called()


def test_an_unrouted_message_without_an_open_conversation_is_not_given_to_any_agent():
    orchestrator, handlers = make_orchestrator(CURRICULUM, DOCUMENT, MARKETING)
    response = orchestrator.handle_request("thanks", {"conversation_id": "c1"})

    assert response["status"] == "error" and response["result"][0]["agent_name"] == "orchestrator"
    assert not any(h.called for h in handlers.values())


def test_the_admin_orchestrator_still_refuses_rex():
    orchestrator, handlers = make_orchestrator(CURRICULUM)
    response = orchestrator.handle_request("Recommend a course for me")
    assert response["status"] == "error" and "REX Assistant" in response["error_message"]
