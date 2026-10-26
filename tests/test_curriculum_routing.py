"""
Orchestrator routing for curriculum requests, priority against the other
agents, and the optional continuation hook used for draft refinements.
"""

import pytest

from agents.curriculum_research.request_parser import CREATION_PHRASES
from agents.orchestrator import Orchestrator

CURRICULUM = ["curriculum_research"]


@pytest.fixture(scope="module")
def router():
    return Orchestrator()


def ok(request):
    return {
        "request_id": request["request_id"],
        "agent_name": request["agent_name"],
        "status": "success",
        "result": {"handled_by": request["agent_name"]},
        "error_message": None,
    }


# --------------------------------------------------------------------
# Curriculum requests reach the Curriculum Research Agent
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # search
    "Find a curriculum about data analysis",
    "I need a training curriculum for artificial intelligence",
    "Show me a syllabus for project management",
    "show me python curricula",
    "Find curricula about cybersecurity",
    # creation
    "Create a curriculum about Python",
    "Create a course about Python",
    "Design a course about SQL",
    "Design a Python course for beginners",
    "Create a training course about Excel",
    "Build a course on cybersecurity",
    "Develop a new course about project management for managers",
    # accreditation (routed so the agent can answer honestly)
    "Do an accreditation gap analysis for my AI curriculum",
    "Accreditation alignment for my program",
])
def test_english_curriculum_requests_route_to_curriculum(router, text):
    assert router._make_plan(text) == CURRICULUM


@pytest.mark.parametrize("text", [
    # search
    "أريد منهج تدريبي لتحليل البيانات",
    "ابحث عن مناهج تدريبية",
    "تصميم المناهج التدريبية",
    "أريد مناهج في الذكاء الاصطناعي",
    # creation
    "أنشئ منهج عن تحليل البيانات",
    "إنشاء دورة عن البرمجة",
    "انشاء دورة عن البرمجة",
    "أنشئ دورة عن البرمجة",
    "صمم دورة عن الأمن السيبراني",
    "صمّم دورة عن الأمن السيبراني",
    "إعداد دورة عن إدارة المشاريع",
])
def test_arabic_curriculum_requests_route_to_curriculum(router, text):
    assert router._make_plan(text) == CURRICULUM


@pytest.mark.parametrize("phrase", CREATION_PHRASES)
def test_every_phrase_the_agent_supports_is_routed_to_it(router, phrase):
    # Contract between the agent's creation phrases and the router.
    assert router._make_plan(f"{phrase} about data analysis") == CURRICULUM


# --------------------------------------------------------------------
# Priority and conflicts with other agents
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Find education tenders in Egypt", ["rfp_research"]),
    ("Find tenders for curriculum development in Egypt", ["rfp_research"]),
    ("Find a tender to design a course", ["rfp_research"]),
    ("ابحث عن مناقصات لتصميم منهج", ["rfp_research"]),
    ("Can you recommend a course for me?", ["rex_assistant"]),
    ("Recommend a course for data analysis", ["rex_assistant"]),
    ("Which course should I take?", ["rex_assistant"]),
    ("I want a training course", ["rex_assistant"]),
    ("Make a course recommendation", ["rex_assistant"]),
    ("أريد دورة في البرمجة للمبتدئين", ["rex_assistant"]),
    ("Generate a course participant guide", ["document_generation"]),
    ("Create a course question bank", ["document_generation"]),
    ("Write a marketing campaign", ["marketing"]),
])
def test_other_agents_keep_their_requests(router, text, expected):
    assert router._make_plan(text) == expected


def test_course_proposal_is_not_taken_by_curriculum(router):
    assert router._make_plan("Create a course proposal") != CURRICULUM


@pytest.mark.parametrize("text", [
    "Create a curriculum and marketing campaign",
    "Create a course about SQL and a marketing campaign",
    "أنشئ منهج تدريبي ثم حملة تسويقية",
])
def test_curriculum_plus_marketing_workflow_is_preserved(router, text):
    assert router._make_plan(text) == ["curriculum_research", "marketing"]


@pytest.mark.parametrize("text", [
    "Hello there",
    "yes",
    "make it 40 hours",
    "Add a capstone project",
    "How much is a pizza",
    "Make a good decision about the course",
])
def test_messages_without_keywords_still_have_no_plan(router, text):
    assert router._make_plan(text) == []


# --------------------------------------------------------------------
# Continuation hook (only consulted when no keyword matched)
# --------------------------------------------------------------------

def make_router():
    router = Orchestrator()
    for name in ("curriculum_research", "rfp_research", "rex_assistant"):
        router.register_agent(name, ok)
    return router


def test_continuation_claims_a_message_with_no_keywords():
    router = make_router()
    seen = []

    def check(probe):
        seen.append(probe)
        return True

    router.register_agent("curriculum_research", ok, continuation_check=check)

    response = router.handle_request(
        "make it 40 hours", context={"conversation_id": "c1"}
    )

    assert response["status"] == "success"
    assert response["result"][0]["agent_name"] == "curriculum_research"
    assert seen[0]["user_input"] == "make it 40 hours"
    assert seen[0]["context"]["conversation_id"] == "c1"
    assert seen[0]["context"]["language"] == "en"


def test_continuation_receives_the_detected_language():
    router = make_router()
    seen = []
    router.register_agent(
        "curriculum_research", ok,
        continuation_check=lambda probe: seen.append(probe) or True,
    )

    router.handle_request("اجعلها 40 ساعة", context={"conversation_id": "c1"})

    assert seen[0]["context"]["language"] == "ar"


def test_a_false_check_changes_nothing():
    router = make_router()
    router.register_agent(
        "curriculum_research", ok, continuation_check=lambda probe: False
    )

    response = router.handle_request("make it 40 hours")

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == "orchestrator"


def test_keyword_matches_always_win_over_a_continuation():
    router = make_router()
    asked = []
    router.register_agent(
        "curriculum_research", ok,
        continuation_check=lambda probe: asked.append(1) or True,
    )

    tenders = router.handle_request("Find education tenders in Egypt")
    recommend = router.handle_request("Recommend a course for me")

    assert tenders["result"][0]["agent_name"] == "rfp_research"
    assert recommend["result"][0]["agent_name"] == "rex_assistant"
    assert asked == []  # not even consulted


def test_a_failing_check_is_ignored():
    router = make_router()

    def broken(probe):
        raise RuntimeError("store unavailable")

    router.register_agent("curriculum_research", ok, continuation_check=broken)

    response = router.handle_request("make it 40 hours")

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == "orchestrator"


def test_registering_without_a_check_clears_the_old_one():
    router = make_router()
    router.register_agent(
        "curriculum_research", ok, continuation_check=lambda probe: True
    )
    router.register_agent("curriculum_research", ok)

    assert router.handle_request("make it 40 hours")["status"] == "error"


def test_existing_two_argument_registration_still_works():
    router = Orchestrator()
    router.register_agent("rfp_research", ok)

    assert router.handle_request("Find education tenders")["status"] == "success"


def test_unknown_agent_names_are_still_rejected():
    with pytest.raises(ValueError):
        Orchestrator().register_agent(
            "made_up", ok, continuation_check=lambda probe: True
        )
