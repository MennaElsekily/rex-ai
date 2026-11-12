"""
Opt-in end-to-end check of the real runtime path, skipped by default: set
REX_REAL_MODEL_TESTS=1 (needs Ollama and the local embedding, reranker and
translation models; several minutes).

The academy catalogue (knowledge_base/documents) is ingested with the real
embedding model into a temporary index. REX then runs with its real MCP
client, which starts the real RAG MCP server as a subprocess on that index
(REX_VECTOR_DB_DIR): language detection, Arabic->English translation,
query embedding, ChromaDB similarity search and reranking all happen in the
server, and REX's real model writes the answers. The project's index is
never opened.
"""

import os
import re
import tempfile
from pathlib import Path

import pytest

from agents.rex_assistant.interface import RexAssistantAgent
from agents.rex_assistant.lead_store import JSONLLeadStore
from agents.rex_assistant.mcp_client import RagMcpClient
from knowledge_base import pipeline
from knowledge_base.vector_store import VectorStore
from tests.test_rex_grounding import DOCUMENTS, _assert_grounded

pytestmark = pytest.mark.skipif(os.environ.get("REX_REAL_MODEL_TESTS") != "1",
                                reason="Real-model test: set REX_REAL_MODEL_TESTS=1 (needs Ollama).")

PROFILE_EN = {"background": "Computer Engineer", "career_goal": "Data Analytics", "skill_level": "Beginner",
              "training_preference": "Online or offline", "budget": "4000", "available_time": "3 days per week",
              "interests": "skip"}
PROFILE_AR = {"background": "مهندس كمبيوتر", "career_goal": "تحليل البيانات", "skill_level": "مبتدئ",
              "training_preference": "أونلاين أو حضوري", "budget": "4000", "available_time": "3 أيام في الأسبوع",
              "interests": "تخطي"}


class RecordingClient(RagMcpClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    def call_tool(self, tool_name, arguments):
        result = super().call_tool(tool_name, arguments)
        self.calls.append((tool_name, dict(arguments), result))
        return result


@pytest.fixture(scope="module")
def real_index(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real_mcp")
    store = VectorStore(persist_directory=directory / "db")
    summary = pipeline.ingest_directory(DOCUMENTS, vector_store=store, metadata_path=directory / "meta.json",
                                        manifest_path=directory / "manifest.json")
    assert not summary["files_failed"]
    return store


@pytest.fixture
def rex(real_index, monkeypatch):
    monkeypatch.setenv("REX_VECTOR_DB_DIR", real_index.persist_directory)
    client = RecordingClient()
    agent = RexAssistantAgent(lead_store=JSONLLeadStore(Path(tempfile.mkdtemp()) / "leads.jsonl"), mcp_client=client)
    yield agent, client
    agent.close()


def _converse(agent, messages, profile=None):
    state, response, queue = None, None, list(messages)
    while queue:
        response = agent.handle_message(queue.pop(0), state=state)
        state, result = response["state"], response["result"]
        if profile and result.get("pending_field"):
            queue.insert(0, profile[result["pending_field"]])
    return response


def _searches(client):
    return [result for name, _, result in client.calls if name == "search_knowledge_base"]


@pytest.mark.parametrize("message, profile, language", [
    ("I am a beginner interested in data analytics. Recommend suitable courses.", PROFILE_EN, "en"),
    ("أنا مبتدئ ومهتم بتحليل البيانات. رشح لي دورات مناسبة.", PROFILE_AR, "ar"),
], ids=["en", "ar"])
def test_recommendation_runs_through_the_real_mcp_server(rex, message, profile, language):
    agent, client = rex
    result = _converse(agent, [message], profile)["result"]
    answer = result["answer"]
    print(f"\n--- real MCP recommendation ({language}) ---\n{answer}")

    search = _searches(client)[-1]
    assert search["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2" and search["embedding_dimensions"] == 384
    assert search["collection"] == "rex_knowledge" and search["chunks"]
    if language == "ar":
        assert search["translation"] == "translated"
        assert not re.search(r"[؀-ۿ]", search["embedded_text"])          # English was embedded
        assert re.search(r"[؀-ۿ]", answer.split("**")[0])                # the answer is Arabic
    else:
        assert search["translation"] == "not_needed"
    records = client.get_course_records()["records"]
    shown = _assert_grounded(answer, records)
    retrieved = {c["course_id"] for c in search["chunks"] if c.get("course_id")}
    assert {course_id for course_id, _ in shown} <= retrieved                       # evidence-backed only
    assert not re.search(r"Data-led|Basic Computer Literacy", answer)


def test_complete_catalogue_and_category_through_the_real_mcp_server(rex):
    agent, client = rex
    everything = _converse(agent, ["What courses does the academy offer?"])["result"]["answer"]
    category = _converse(agent, ["What Data Analytics courses does the academy offer?"])["result"]["answer"]
    arabic = _converse(agent, ["ما هي جميع الدورات التي تقدمها الأكاديمية؟"])["result"]["answer"]

    assert len(re.findall(r"^\d+\. \*\*", everything, re.MULTILINE)) == 14
    assert sorted(re.findall(r"\(([A-Z]+-\d+)\) —", category)) == ["DA-101", "DA-201", "DA-301"]
    assert arabic.startswith("يضم كتالوج الأكاديمية 14 دورة")
    assert not _searches(client)                                                    # no top-k search for a listing


def test_a_specific_course_and_its_arabic_follow_up_through_the_real_mcp_server(rex):
    agent, client = rex
    english = _converse(agent, ["Tell me more about SQL and Power BI for Data Analysts.", "What are the fees?"])
    arabic = _converse(agent, ["ما هي متطلبات الالتحاق بدورة Excel for Business Analysis؟", "كم عدد ساعاتها؟"])

    assert english["result"]["selected_course"]["course_id"] == "DA-201"
    assert "7800" in english["result"]["answer"]
    assert arabic["result"]["selected_course"]["course_id"] == "DA-101"
    assert "24" in arabic["result"]["answer"]
    scoped = [args for name, args, _ in client.calls if name == "search_knowledge_base" and args.get("course_ids")]
    assert ["DA-201"] in [a["course_ids"] for a in scoped] and ["DA-101"] in [a["course_ids"] for a in scoped]
