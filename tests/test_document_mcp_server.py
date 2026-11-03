"""
agents/document_mcp_server.py through a real MCP subprocess
(DocumentMcpClient): real stdio transport and tool schemas, with temporary
curriculum and document databases and PDF folder.
"""

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation import templates
from agents.document_generation.composer import compose
from agents.document_generation.mcp_client import DocumentMcpClient
from agents.mcp_tool_client import McpToolError
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, parts_answers

TOOLS = {"list_document_types", "list_approved_curricula", "get_approved_curriculum", "save_document_draft",
         "render_document_pdf", "get_document", "list_documents"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("document_server")
    curricula = CurriculumStore(root / "curricula.sqlite3")
    approved = curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    curricula.approve(approved["curriculum_id"], 1, ADMIN)
    pending = curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    curricula.close()
    return {"root": root, "approved": approved["curriculum_id"], "pending": pending["curriculum_id"]}


@pytest.fixture(scope="module")
def client(world):
    root = world["root"]
    client = DocumentMcpClient(env={"DOCUMENT_DB_PATH": str(root / "documents.sqlite3"),
                                    "DOCUMENT_OUTPUT_DIR": str(root / "pdfs"),
                                    "CURRICULUM_DB_PATH": str(root / "curricula.sqlite3")}, startup_timeout=120)
    yield client
    client.close()


def test_the_server_offers_exactly_the_document_tools(client):
    tools = {tool["name"]: tool for tool in client.list_tools()}

    assert set(tools) == TOOLS   # no approve tool, no file-path tool, no file-content tool
    for tool in tools.values():
        assert tool["description"] and tool["input_schema"]["type"] == "object"
    assert tools["save_document_draft"]["input_schema"]["required"] == ["content"]
    assert tools["get_approved_curriculum"]["input_schema"]["required"] == ["curriculum_id"]


def test_types_and_approved_curricula_come_from_the_configured_stores(client, world):
    types = client.list_document_types()
    assert [t["document_type"] for t in types["supported"]] == templates.supported_types()
    assert {t["document_type"] for t in types["unsupported"]} == {"curriculum_pdf"}

    assert [c["curriculum_id"] for c in client.list_approved_curricula()["curricula"]] == [world["approved"]]
    assert client.get_approved_curriculum(world["pending"])["status"] == "not_approved"
    assert client.get_approved_curriculum("bad id")["status"] == "invalid_request"


def test_save_render_and_read_a_document_through_the_server(client, world):
    record = client.get_approved_curriculum(world["approved"])["curriculum"]
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]))

    saved = client.save_document_draft(content)
    assert saved["status"] == "ok" and saved["review_status"] == "pending_review"
    rendered = client.render_document_pdf(saved["document_id"], saved["version"])
    assert rendered["status"] == "ok" and rendered["document"]["pages"] >= 2
    fetched = client.get_document(saved["document_id"])
    assert fetched["document"]["filename"] == f"{saved['document_id']}_v1_en.pdf"
    assert [d["document_id"] for d in client.list_documents(status="pending_review")["documents"]] == [saved["document_id"]]


def test_the_server_validates_what_it_is_given(client, world):
    record = client.get_approved_curriculum(world["approved"])["curriculum"]
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]))

    tampered = {**content, "curriculum_id": world["pending"]}
    assert client.save_document_draft(tampered)["status"] == "not_approved"
    content["sections"][1]["blocks"][0]["text"] = "Fully compliant with the national standard."
    result = client.save_document_draft(content)
    assert result["review_status"] == "draft" and result["validation"]["errors"][0]["code"] == "accreditation_claim"
    with pytest.raises(McpToolError):   # a malformed call is rejected by MCP itself
        client.call_tool("save_document_draft", {"content": "not an object"})


def test_the_server_pages_the_approved_curricula(client, world):
    page = client.list_approved_curricula(limit=1, offset=0)
    assert (page["total"], page["offset"], page["has_more"], len(page["curricula"])) == (1, 0, False, 1)
    assert client.list_approved_curricula(limit=1, offset=1)["curricula"] == []


def test_the_server_checks_the_extra_fields_itself(client, world):
    record = client.get_approved_curriculum(world["approved"])["curriculum"]
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]))
    for extra, code in (({"price": "5"}, "extra_unknown"), ({"organization": "x" * 400}, "extra_too_long"),
                        ({"training_format": " padded "}, "extra_invalid")):
        result = client.save_document_draft({**content, "extra": extra})
        assert result["review_status"] == "draft"
        assert code in [e["code"] for e in result["validation"]["errors"]]
