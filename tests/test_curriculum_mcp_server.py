"""
agents/curriculum_mcp_server.py through a real MCP subprocess
(CurriculumMcpClient): real stdio transport and tool schemas, with a
temporary index, database, documents and frameworks folders, research
disabled and no Tavily key (no network).
"""

import json

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.curriculum_research.mcp_client import CurriculumMcpClient
from agents.mcp_tool_client import McpToolError
from tests.curriculum_fixtures import FRAMEWORK, build_test_index, valid_curriculum

TOOLS = {"list_academy_courses", "get_course_details", "check_course_overlap", "catalogue_gap_report",
         "search_curriculum_sources", "list_frameworks", "get_framework_requirements", "validate_curriculum",
         "compare_with_framework", "save_curriculum_draft", "get_curriculum", "list_curricula",
         "render_curriculum_pdf"}


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("curriculum_server")
    build_test_index(root / "index")
    (root / "frameworks").mkdir()
    (root / "frameworks" / "test-framework.json").write_text(json.dumps(FRAMEWORK), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def client(paths):
    client = CurriculumMcpClient(env={
        "CURRICULUM_INDEX_DIR": str(paths / "index"), "CURRICULUM_DB_PATH": str(paths / "curricula.sqlite3"),
        "CURRICULUM_DOCUMENTS_DIR": str(paths / "docs"), "CURRICULUM_FRAMEWORKS_DIR": str(paths / "frameworks"),
        "CURRICULUM_RESEARCH_DOMAINS": "", "TAVILY_API_KEY": "",
    }, startup_timeout=120)
    yield client
    client.close()


def test_the_server_offers_exactly_the_curriculum_tools(client):
    tools = {tool["name"]: tool for tool in client.list_tools()}

    assert set(tools) == TOOLS  # no approve / request-changes tool
    for tool in tools.values():
        assert tool["description"] and tool["input_schema"]["type"] == "object"
    assert tools["validate_curriculum"]["input_schema"]["required"] == ["curriculum", "requested_hours"]


def test_courses_come_from_the_configured_index(client):
    result = client.list_academy_courses(department="Data Analytics")

    assert [c["course_id"] for c in result["courses"]] == ["DA-101", "DA-201"]
    assert client.get_course_details("DA-201")["course"]["prerequisite_courses"] == ["DA-101"]


def test_research_is_reported_as_not_configured(client):
    assert client.search_curriculum_sources("cloud computing curriculum")["status"] == "not_configured"


def test_save_and_read_through_the_server(client, paths):
    saved = client.save_curriculum_draft(valid_curriculum(), topic="machine learning")

    assert (saved["status"], saved["review_status"]) == ("ok", "pending_review") and saved["validation"]["valid"]
    record = client.get_curriculum(saved["curriculum_id"])["curriculum"]
    assert record["status"] == "pending_review"
    # Stored in the configured temporary database.
    assert CurriculumStore(paths / "curricula.sqlite3").get(saved["curriculum_id"])["title"] == record["title"]


def test_frameworks_and_validation_through_the_server(client):
    assert client.get_framework_requirements("test-framework")["framework"]["name"] == FRAMEWORK["name"]
    assert client.get_framework_requirements("missing")["status"] == "not_found"
    result = client.validate_curriculum(valid_curriculum(hours=40), 45)
    assert result["valid"] is False and "hours_total" in {e["code"] for e in result["errors"]}


def test_bad_arguments_are_reported_not_crashing(client):
    assert client.get_curriculum("../../etc/passwd")["status"] == "invalid_request"
    with pytest.raises(McpToolError, match="reported an error"):
        client.call_tool("validate_curriculum", {"curriculum": "not an object", "requested_hours": 40})
    assert client.list_frameworks()["status"] == "ok"  # the session survives
