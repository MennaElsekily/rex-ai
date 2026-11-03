"""
agents/marketing_mcp_server.py through a real MCP subprocess (MarketingMcpClient): real stdio transport and tool schemas,
with temporary curriculum and marketing databases. The server offers no way to approve, export, publish or send.
"""

import copy

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.marketing.marketing_agent import MarketingAgent
from agents.marketing.marketing_store import MarketingStore
from agents.marketing.marketing_tools import MarketingToolbox
from agents.marketing.mcp_client import MarketingMcpClient
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import ADMIN, fake_llm, make_campaign

TOOLS = {"list_approved_courses", "get_approved_course", "save_campaign_draft", "get_campaign", "list_campaigns"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("marketing_server")
    curricula = CurriculumStore(root / "curricula.sqlite3")
    approved = curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    curricula.approve(approved["curriculum_id"], 1, ADMIN)
    pending = curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    # a campaign written in-process (a separate database file) to hand to the server's save tool
    writer = MarketingToolbox(store=MarketingStore(root / "writer.sqlite3"), curricula=curricula)
    made = make_campaign(writer, approved["curriculum_id"])["result"]
    content = writer.store.get(made["campaign_id"])["content"]
    curricula.close()
    return {"root": root, "approved": approved["curriculum_id"], "pending": pending["curriculum_id"], "content": content}


@pytest.fixture(scope="module")
def client(world):
    root = world["root"]
    client = MarketingMcpClient(env={"MARKETING_DB_PATH": str(root / "marketing.sqlite3"),
                                     "CURRICULUM_DB_PATH": str(root / "curricula.sqlite3")}, startup_timeout=120)
    yield client
    client.close()


def test_the_server_offers_exactly_the_marketing_tools(client):
    tools = {tool["name"]: tool for tool in client.list_tools()}

    assert set(tools) == TOOLS                       # no approve, export, publish, send or file tool
    for tool in tools.values():
        assert tool["description"] and tool["input_schema"]["type"] == "object"
    assert tools["save_campaign_draft"]["input_schema"]["required"] == ["content"]
    assert tools["get_approved_course"]["input_schema"]["required"] == ["curriculum_id"]
    assert not any(word in name for name in tools for word in ("approve_", "export", "publish", "send", "email", "whatsapp"))


def test_only_approved_courses_are_listed_and_fetched(client, world):
    listed = client.list_approved_courses()
    assert listed["status"] == "ok" and [c["curriculum_id"] for c in listed["courses"]] == [world["approved"]]
    assert client.get_approved_course(world["approved"])["course"]["version"] == 1
    refused = client.get_approved_course(world["pending"])
    assert refused["status"] == "not_approved" and "course" not in refused
    assert client.get_approved_course("CUR-aaaaaaaaaaaa")["status"] == "not_found"


def test_the_server_validates_what_it_saves_and_can_never_approve(client, world):
    saved = client.save_campaign_draft(copy.deepcopy(world["content"]))
    assert saved["status"] == "ok" and saved["review_status"] == "pending_review" and saved["campaign_id"].startswith("MKT-")

    bad = copy.deepcopy(world["content"])
    bad["channels"]["whatsapp"]["variations"][0]["message"] = "Join the best course for only 99 dollars, guaranteed!"
    kept = client.save_campaign_draft(bad)
    assert kept["review_status"] == "draft" and not kept["validation"]["valid"]

    unapproved = copy.deepcopy(world["content"])
    unapproved["curriculum_id"] = world["pending"]
    assert client.save_campaign_draft(unapproved)["status"] == "not_approved"

    got = client.get_campaign(saved["campaign_id"])
    assert got["status"] == "ok" and got["campaign"]["version_status"] == "pending_review" and got["campaign"]["approved_version"] is None
    assert {c["campaign_id"] for c in client.list_campaigns()["campaigns"]} >= {saved["campaign_id"], kept["campaign_id"]}
    assert [c["campaign_id"] for c in client.list_campaigns("approved")["campaigns"]] == []


def test_the_agent_works_through_the_real_server(client, world, tmp_path):
    agent = MarketingAgent(tools=client, llm=fake_llm())
    response = agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "language": "en", "marketing_request": {"curriculum_id": world["approved"], "channels": ["email"], "variations": 1}}})

    assert response["status"] == "success" and response["result"]["review_status"] == "pending_review"
    assert client.get_campaign(response["result"]["campaign_id"])["campaign"]["content"]["variations"] == 1
    refused = agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "language": "en", "marketing_request": {"curriculum_id": world["pending"]}}})
    assert refused["result"]["kind"] == "not_approved"
