"""
The intended five-agent architecture, pinned:

    User  -> UI -> REX Assistant (Agent 1) -> Knowledge base / RAG -> REX -> User
    Admin -> UI / Ask Agents / floating assistant -> AdminService -> Central Orchestrator
          -> RFP (2) / Curriculum (3) / Documents (4) / Marketing (5) -> Orchestrator -> Admin

Agent 1 never passes through the admin orchestrator; every admin agent
request does; and the orchestrator really coordinates (order, context
from earlier steps, partial failures) and synthesizes one account.
"""

import ast
from pathlib import Path

from agents import register_agents
from agents.orchestrator import Orchestrator, synthesize_results

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# --------------------------------------------------------------------
# Agent 1: its own path to the knowledge base, never the orchestrator
# --------------------------------------------------------------------

def test_the_user_service_never_imports_the_orchestrator_or_admin_side():
    imports = _imports(ROOT / "agents" / "user_service.py")
    assert not any("orchestrator" in name or "admin_service" in name or "register_agents" in name
                   for name in imports), imports


def test_rex_assistant_reaches_the_knowledge_base_directly():
    package = ROOT / "agents" / "rex_assistant"
    imports = set().union(*(_imports(p) for p in package.rglob("*.py")))
    assert any(name.startswith("knowledge_base") for name in imports), imports
    assert not any("orchestrator" in name for name in imports), imports


def test_the_admin_orchestrator_cannot_route_to_rex_assistant(monkeypatch):
    built = {}

    class Spy(Orchestrator):
        def __init__(self, excluded_agents=()):
            super().__init__(excluded_agents=excluded_agents)
            built["excluded"] = tuple(excluded_agents)

    monkeypatch.setattr(register_agents, "Orchestrator", Spy)
    for name in ("rfp_agent", "curriculum_agent", "document_agent", "marketing_agent"):
        if hasattr(register_agents, name):
            monkeypatch.setattr(register_agents, name, None, raising=False)
    try:
        register_agents.create_admin_orchestrator()
    except Exception:
        pass   # building the real agents is not needed: the exclusion is set in the constructor
    assert built.get("excluded") == (register_agents.REX_ASSISTANT_AGENT_NAME,)


# --------------------------------------------------------------------
# The orchestrator coordinates and synthesizes
# --------------------------------------------------------------------

def _orchestrator(*handlers):
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    for name, handler in handlers:
        orchestrator.register_agent(name, handler)
    return orchestrator


def test_a_chained_request_runs_in_order_and_each_agent_sees_the_earlier_results():
    seen = {}

    def curriculum(request):
        seen["curriculum"] = list(request["context"]["previous_results"])
        return {"agent_name": "curriculum_research", "status": "success",
                "result": {"kind": "curriculum", "message": "Curriculum drafted: Data Analysis\nID: CUR-1"}}

    def document(request):
        seen["document"] = [r["agent_name"] for r in request["context"]["previous_results"]]
        return {"agent_name": "document_generation", "status": "success",
                "result": {"message": "Course proposal ready."}}

    def marketing(request):
        seen["marketing"] = [r["agent_name"] for r in request["context"]["previous_results"]]
        return {"agent_name": "marketing", "status": "success", "result": {"message": "Campaign ready."}}

    orchestrator = _orchestrator(("curriculum_research", curriculum), ("document_generation", document),
                                 ("marketing", marketing))

    response = orchestrator.handle_request(
        "Create a data analysis curriculum, then a professional course proposal document from it, "
        "then a marketing campaign for the same course")

    assert [r["agent_name"] for r in response["result"]] == ["curriculum_research", "document_generation",
                                                              "marketing"]
    assert seen["curriculum"] == []
    assert seen["document"] == ["curriculum_research"]
    assert seen["marketing"] == ["curriculum_research", "document_generation"]
    # The closing lines say what actually happened (it used to claim "each step used the results of
    # the steps before it" even when the steps were independent).
    assert response["summary"] == ("1. Curriculum research: Curriculum drafted: Data Analysis\n"
                                   "2. Document generation: Course proposal ready.\n"
                                   "3. Marketing: Campaign ready.\n"
                                   "Steps 2, 3 used the curriculum from step 1.\n"
                                   "Done: 3/3")


def test_a_failing_agent_does_not_stop_the_others_and_is_named_in_the_synthesis():
    def curriculum(request):
        return {"agent_name": "curriculum_research", "status": "success", "result": {"message": "Drafted."}}

    def document(request):
        raise RuntimeError("the document service is down")

    orchestrator = _orchestrator(("curriculum_research", curriculum), ("document_generation", document))

    response = orchestrator.handle_request("Create a curriculum about SQL and then a course proposal document from it")

    assert response["status"] == "error" and len(response["result"]) == 2
    assert response["result"][0]["status"] == "success"
    assert "2. Document generation: Failed: the document service is down" in response["summary"]


def test_the_synthesis_explains_a_step_that_waits_for_approval_in_both_languages():
    results = [{"agent_name": "curriculum_research", "status": "success", "result": {"message": "Drafted."}},
               {"agent_name": "marketing", "status": "paused", "result": {"message": "Waiting."}}]

    assert "2. Marketing: Waiting for an administrator's approval" in synthesize_results(results, "en")
    assert "2. التسويق: بانتظار اعتماد المسؤول" in synthesize_results(results, "ar")


def test_admin_requests_and_the_floating_assistant_go_through_admin_service_ask():
    """Both chat surfaces call the one admin_call("ask", ...) path; there is no second chat system."""
    source = (ROOT / "rex_ai_app.py").read_text(encoding="utf-8")
    assert source.count('admin_call("ask"') == 1
    assert "agents_tab(key_suffix=\"_floating\")" in source
    assert "handle_request(" not in source   # the UI never talks to an orchestrator directly
