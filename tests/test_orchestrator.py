
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
