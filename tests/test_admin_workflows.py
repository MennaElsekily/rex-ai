"""
Admin agents 2-5 through the Central Orchestrator (acceptance, 2026-11-11):
planning of single- and multi-agent requests, each agent getting the right
part of a mixed request, genuine execution of every planned agent, and a
synthesis that keeps every agent's outcome, ids, warnings and errors and
says truthfully whether the steps depended on each other.

Mocked: the agents' handlers (the real agents are covered by their own
tests and by the opt-in real-model runs).
"""

from __future__ import annotations

import pytest

from agents.orchestrator import Orchestrator, synthesize_results
from agents.register_agents import create_admin_orchestrator


def _orchestrator(*handlers):
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    for name, handler in handlers:
        orchestrator.register_agent(name, handler)
    return orchestrator


@pytest.mark.parametrize("request_text, plan", [
    # Workflow 1: a course name of several words before "curriculum" used to lose the curriculum step.
    ("Create a 30-hour Excel for Business Analysis curriculum for beginners, then a course proposal document from it",
     ["curriculum_research", "document_generation"]),
    # Workflow 2
    ("Create a curriculum about SQL for data analysts and then a marketing campaign for it",
     ["curriculum_research", "marketing"]),
    # Workflow 4: a tender search and a curriculum used to run the tender search only.
    ("Find training tenders in Egypt and create a curriculum for a data analytics course",
     ["rfp_research", "curriculum_research"]),
    ("ابحث عن مناقصات تدريب في مصر ثم أنشئ منهجًا عن تحليل البيانات", ["rfp_research", "curriculum_research"]),
    # Single agents stay single.
    ("Find training tenders in Saudi Arabia", ["rfp_research"]),
    ("Generate a course description for the Python curriculum", ["document_generation"]),
    ("Show me the tenders for the data analytics curriculum project", ["rfp_research"]),
])
def test_requests_are_planned_to_the_agents_they_ask_for(request_text, plan):
    assert Orchestrator(excluded_agents=("rex_assistant",))._make_plan(request_text) == plan


def test_a_mixed_request_runs_both_agents_each_on_its_own_part():
    seen = {}

    def rfp(request):
        seen["rfp"] = request["user_input"]
        return {"agent_name": "rfp_research", "status": "success",
                "result": {"message": "Found 3 tenders in Egypt; 2 are new and wait for your review."}}

    def curriculum(request):
        seen["curriculum"] = (request["user_input"], [r["agent_name"] for r in request["context"]["previous_results"]])
        return {"agent_name": "curriculum_research", "status": "success",
                "result": {"message": "Curriculum drafted: Data Analytics", "curriculum_id": "CUR-1",
                           "review_status": "pending_review", "bilingual_companion_id": "CUR-2"}}

    response = _orchestrator(("rfp_research", rfp), ("curriculum_research", curriculum)).handle_request(
        "Find training tenders in Egypt and create a curriculum for a data analytics course")

    assert seen["rfp"] == "Find training tenders in Egypt"
    assert seen["curriculum"] == ("create a curriculum for a data analytics course", ["rfp_research"])
    assert [r["agent_name"] for r in response["result"]] == ["rfp_research", "curriculum_research"]
    assert response["status"] == "success"
    assert response["summary"] == (
        "1. Tender research: Found 3 tenders in Egypt; 2 are new and wait for your review.\n"
        "2. Curriculum research: Curriculum drafted: Data Analytics [CUR-1, pending review; companion CUR-2]\n"
        "These steps ran independently.\n"
        "Done: 2/2")


def test_one_agent_failing_is_reported_and_the_other_still_runs():
    def rfp(request):
        raise RuntimeError("the World Bank API is unreachable")

    def curriculum(request):
        return {"agent_name": "curriculum_research", "status": "success", "result": {"message": "Drafted."}}

    response = _orchestrator(("rfp_research", rfp), ("curriculum_research", curriculum)).handle_request(
        "Find training tenders in Egypt and create a curriculum for a data analytics course")

    assert response["status"] == "error" and "World Bank API is unreachable" in response["error_message"]
    assert response["result"][1]["status"] == "success"
    assert "1. Tender research: Failed: the World Bank API is unreachable" in response["summary"]
    assert response["summary"].endswith("Done: 1/2 · Failed: 1")


def test_the_synthesis_keeps_warnings_ids_and_waiting_steps_in_both_languages():
    results = [{"agent_name": "curriculum_research", "status": "success",
                "result": {"message": "Curriculum drafted: SQL", "curriculum_id": "CUR-9", "review_status": "draft",
                           "warnings": ["LO3 is not assessed."]}},
               {"agent_name": "document_generation", "status": "paused", "result": {"message": "Waiting."}}]

    english = synthesize_results(results, "en")
    arabic = synthesize_results(results, "ar")

    assert "1. Curriculum research: Curriculum drafted: SQL [CUR-9, draft]" in english
    assert "   ⚠ LO3 is not assessed." in english
    assert "2. Document generation: Waiting for an administrator's approval" in english
    assert english.endswith("Step 2 used the curriculum from step 1.\nDone: 1/2 · Waiting: 1")
    assert "[CUR-9, مسودة]" in arabic and "بانتظار الاعتماد: 1" in arabic


def test_the_admin_orchestrator_runs_the_four_admin_agents_and_never_rex():
    orchestrator = create_admin_orchestrator()

    assert {"rfp_research", "curriculum_research", "document_generation", "marketing"} <= set(orchestrator.agents)
    assert "rex_assistant" in orchestrator.excluded_agents
    response = orchestrator.handle_request("Recommend a course for me")
    assert response["status"] == "error" and response["result"][0]["agent_name"] == "orchestrator"


def test_every_admin_action_enters_the_orchestrator():
    from agents.admin_service import AdminService

    calls = []

    class RecordingOrchestrator:
        def handle_request(self, message, context=None):
            calls.append((message, sorted((context or {}).keys())))
            return {"status": "success", "result": [], "summary": ""}

    admin = AdminService(orchestrator_factory=RecordingOrchestrator, passcode="pass-1234")
    token = admin.login("pass-1234")
    admin.ask(token, "Create a curriculum about SQL")
    admin.search_tenders(token, {"countries": ["Egypt"]})
    admin.generate_document(token, "CUR-1", "course_proposal")
    admin.generate_campaign(token, {"curriculum_id": "CUR-1"})

    assert len(calls) == 4
    assert "search" in calls[1][1] and "document_request" in calls[2][1] and "marketing_request" in calls[3][1]


def test_a_course_name_with_for_in_it_is_a_curriculum_request():
    # Real-model run: "Create a 30-hour Excel for Business Analysis curriculum ..." got the agent's help text.
    from agents.curriculum_research import understanding

    text = "Create a 30-hour Excel for Business Analysis curriculum for beginners"
    slots = understanding.request_slots(text)

    assert understanding.is_creation_request(text)
    assert (slots["topic"], slots["total_hours"], slots["level"]) == ("Excel for Business Analysis", 30, "Beginner")


def test_a_chained_step_waits_while_the_curriculum_agent_asks_for_details():
    # Real-model run: "Create a curriculum about SQL ... and then a marketing campaign" -- the curriculum
    # agent asked for the hours and level, and the campaign step was shown as "Failed: no course approved".
    from agents.admin_service import AdminService

    class AskingOrchestrator:
        def handle_request(self, message, context=None):
            return {"status": "error", "error_message": "No course is approved yet.", "result": [
                {"agent_name": "curriculum_research", "status": "success",
                 "result": {"kind": "question", "message": "To draft this curriculum I need: the hours."}},
                {"agent_name": "marketing", "status": "error", "error_message": "No course is approved yet.",
                 "result": {"kind": "not_approved", "message": "No course is approved yet."}}]}

    admin = AdminService(orchestrator_factory=AskingOrchestrator, passcode="pass-1234")
    response = admin.ask(admin.login("pass-1234"), "Create a curriculum about SQL and then a marketing campaign for it")

    assert response["status"] == "success" and response["result"][1]["status"] == "waiting_for_input"
    assert "Failed" not in response["summary"]
    assert "2. Marketing: Waiting for your answer to the curriculum questions." in response["summary"]
