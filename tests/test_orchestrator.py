
from unittest.mock import patch

import pytest

from agents.multilingual import looks_like_arabizi, transliterate_arabizi
from agents.orchestrator import Orchestrator


def mock_agent(request):
    """A temporary fake agent for testing the routing system."""
    return {
        "request_id": request["request_id"],
        "agent_name": request["agent_name"],
        "status": "success",
        "result": f"Mock response from {request['agent_name']}",
        "error_message": None,
    }


def test_routes_tender_request():
    router = Orchestrator()
    router.register_agent("rfp_research", mock_agent)

    response = router.handle_request(
        "Find education tenders in Egypt",
        request_id="test-001",
    )

    assert response["status"] == "success"
    assert response["request_id"] == "test-001"
    assert response["result"][0]["agent_name"] == "rfp_research"


def test_multi_agent_workflow():
    router = Orchestrator()
    router.register_agent("curriculum_research", mock_agent)
    router.register_agent("marketing", mock_agent)

    response = router.handle_request(
        "Create a curriculum and marketing campaign",
        request_id="test-002",
    )

    assert response["status"] == "success"
    assert len(response["result"]) == 2
    assert response["result"][0]["agent_name"] == "curriculum_research"
    assert response["result"][1]["agent_name"] == "marketing"


def test_missing_agent_returns_error():
    router = Orchestrator()

    response = router.handle_request(
        "Find education tenders",
        request_id="test-003",
    )

    assert response["status"] == "error"
    assert "not implemented" in response["error_message"]


def test_unknown_request_returns_error():
    router = Orchestrator()

    response = router.handle_request(
        "Hello there",
        request_id="test-004",
    )

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == "orchestrator"


def test_agent_exception_is_handled():
    router = Orchestrator()

    def broken_agent(request):
        raise RuntimeError("Simulated failure")

    router.register_agent("rfp_research", broken_agent)

    response = router.handle_request(
        "Find education tenders",
        request_id="test-005",
    )

    assert response["status"] == "error"
    assert "Simulated failure" in response["error_message"]


def test_previous_results_are_passed_to_next_agent():
    """
    Confirms the shared-workflow-state mechanism: when a plan runs
    multiple agents, each later agent's context must include every
    earlier agent's result via context["previous_results"].
    """
    router = Orchestrator()
    captured_contexts = []

    def curriculum_agent(request):
        return {
            "request_id": request["request_id"],
            "agent_name": "curriculum_research",
            "status": "success",
            "result": {"course": "AI Foundations"},
            "error_message": None,
        }

    def marketing_agent(request):
        captured_contexts.append(request["context"])
        return {
            "request_id": request["request_id"],
            "agent_name": "marketing",
            "status": "success",
            "result": "Campaign created",
            "error_message": None,
        }

    router.register_agent("curriculum_research", curriculum_agent)
    router.register_agent("marketing", marketing_agent)

    response = router.handle_request(
        "Create a curriculum and marketing campaign",
        request_id="test-006",
    )

    assert response["status"] == "success"
    assert len(captured_contexts) == 1

    previous_results = captured_contexts[0]["previous_results"]
    assert len(previous_results) == 1
    assert previous_results[0]["agent_name"] == "curriculum_research"
    assert previous_results[0]["result"] == {"course": "AI Foundations"}


# --------------------------------------------------------------------
# Language detection (pure, no agents involved)
# --------------------------------------------------------------------

EN_COURSE = "I want to learn programming. Which course do you recommend?"
MSA_COURSE = "أريد تعلم البرمجة، ما الدورة التي تنصحني بها؟"
EGYPTIAN_COURSE = "أنا عايز آخد كورس في البرمجة، تنصحني بإيه؟"
ARABIZI_COURSE_1 = "Ana 3ayez akhod course fel programming."
ARABIZI_COURSE_2 = "Ana 3ayez at3alem programming, tensa7ny b course eh?"


@pytest.mark.parametrize("text, expected", [
    (EN_COURSE, "en"),
    (MSA_COURSE, "ar"),
    (EGYPTIAN_COURSE, "ar"),
    (ARABIZI_COURSE_1, "ar"),
    (ARABIZI_COURSE_2, "ar"),
    ("ezzay a3raf el mawa3eed?", "ar"),
    # Mixed Arabic script + English words: Arabic.
    ("أنا عايز أتعلم Python عشان أشتغل Software Developer.", "ar"),
    # Short / ambiguous input falls back safely.
    ("Hello", "en"),
    ("ok", "en"),
    ("5", "en"),
    ("أهلاً", "ar"),
    # English with numbers or tech tokens is not Arabizi.
    ("I need 3 courses for 2025 on k8s, b2b sales and web3, 2nd level", "en"),
    ("Is the s3bucket course available?", "en"),
    # One weak marker (the name "Ana") is not enough.
    ("Ana wants to know the course price", "en"),
])
def test_detect_language(text, expected):
    assert Orchestrator.detect_language(text) == expected


def test_arabizi_needs_two_weak_signals_without_a_strong_one():
    assert not looks_like_arabizi("Ana is here")
    assert looks_like_arabizi("ana fel beit")


def test_transliteration_is_a_matching_aid_only():
    text = "Ana 3ayez akhod course"

    assert transliterate_arabizi(text) == "انا عايز اخد كورس"
    assert text == "Ana 3ayez akhod course"  # input never mutated
    assert transliterate_arabizi("Find tenders in Egypt") == "Find tenders in Egypt"


# --------------------------------------------------------------------
# Intent routing through handle_request (mock agents only)
# --------------------------------------------------------------------

def _router_with_mock_agents():
    router = Orchestrator()
    for name in ("rex_assistant", "rfp_research", "curriculum_research"):
        router.register_agent(name, mock_agent)
    return router


@pytest.mark.parametrize("text, expected_agent", [
    (EN_COURSE, "rex_assistant"),
    (MSA_COURSE, "rex_assistant"),
    (EGYPTIAN_COURSE, "rex_assistant"),
    (ARABIZI_COURSE_1, "rex_assistant"),
    (ARABIZI_COURSE_2, "rex_assistant"),
    ("Find education and training tenders in Egypt", "rfp_research"),
    ("ابحث عن مناقصات تعليمية في مصر", "rfp_research"),
    ("عايز مناقصات تدريب في السعودية", "rfp_research"),
    ("Create a curriculum about data analysis", "curriculum_research"),
    ("أنشئ منهج تدريبي عن تحليل البيانات", "curriculum_research"),
])
def test_intent_selects_the_correct_agent(text, expected_agent):
    response = _router_with_mock_agents().handle_request(text, request_id="r")

    assert response["status"] == "success"
    assert [item["agent_name"] for item in response["result"]] == [expected_agent]


@pytest.mark.parametrize("text, expected_plan", [
    # More specific intents keep priority over the course rule.
    ("عايز كورس عن المناقصات، تنصحني بإيه؟", ["rfp_research"]),
    ("أريد تعلم كيف أصمم منهج، ما الدورة المناسبة؟", ["curriculum_research"]),
    # A desire verb alone is not a course request (e.g. a draft edit).
    ("عايز الدورة تكون 40 ساعة", []),
    ("Make a good decision about the course", []),
])
def test_course_rule_respects_existing_priorities(text, expected_plan):
    assert Orchestrator()._make_plan(text) == expected_plan


# --------------------------------------------------------------------
# End-to-end pipeline order and data passed to the agent
# --------------------------------------------------------------------

def test_language_is_detected_before_routing():
    router = _router_with_mock_agents()
    calls = []
    real_detect = Orchestrator.detect_language
    real_plan = router._make_plan

    def spy_detect(text):
        calls.append("detect")
        return real_detect(text)

    def spy_plan(text):
        calls.append("route")
        return real_plan(text)

    with patch.object(Orchestrator, "detect_language", staticmethod(spy_detect)), \
            patch.object(router, "_make_plan", spy_plan):
        router.handle_request(ARABIZI_COURSE_1)

    assert calls == ["detect", "route"]


def test_agent_receives_original_query_and_detected_language():
    router = Orchestrator()
    received = []

    def capturing_agent(request):
        received.append(request)
        return mock_agent(request)

    router.register_agent("rex_assistant", capturing_agent)

    response = router.handle_request(
        ARABIZI_COURSE_2,
        context={"conversation_id": "c1"},
        request_id="r-arabizi",
    )

    request = received[0]
    assert request["user_input"] == ARABIZI_COURSE_2  # not transliterated
    assert request["context"]["language"] == "ar"
    assert request["context"]["conversation_id"] == "c1"
    assert request["request_id"] == "r-arabizi"

    assert response["request_id"] == "r-arabizi"
    assert response["agent_name"] == "orchestrator"
    assert response["status"] == "success"
    assert response["error_message"] is None
    assert response["result"][0]["agent_name"] == "rex_assistant"


@pytest.mark.parametrize("text", [
    "What are the entry requirements for the data analytics programme?",
    "How much are the course fees?",
    "What is the duration of the Flutter program?",
    "Who are the instructors?",
    "ما هي رسوم الدورة؟",
    "ما هي متطلبات الالتحاق بالبرنامج؟",
])
def test_course_information_questions_route_to_rex(text):
    assert Orchestrator()._make_plan(text) == ["rex_assistant"]


def test_arabic_request_for_unregistered_agent_still_errors():
    router = Orchestrator()  # nothing registered

    response = router.handle_request(EGYPTIAN_COURSE, request_id="r")

    assert response["status"] == "error"
    assert "not implemented" in response["error_message"]
