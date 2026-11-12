"""
Opt-in check with the REAL local model and the real retrieval models,
skipped by default: set REX_REAL_MODEL_TESTS=1 (needs Ollama; several
minutes). The academy catalogue (knowledge_base/documents) is ingested
with the real embedding model into a temporary store -- the project's
index is never opened -- and REX runs the live-failure learner profile
end to end (Agent 1 -> RAG tools -> verified records -> recommendation),
in English and Arabic.
"""

import os
import re

import pytest

from agents.rex_assistant.interface import RexAssistantAgent
from knowledge_base import pipeline
from knowledge_base.evidence import EvidenceRetriever
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import VectorStore
from tests.test_rex_assistant import FakeLeadStore
from tests.test_rex_grounding import (
    DOCUMENTS, EXACT_PROFILE_AR, EXACT_PROFILE_EN, InProcessRag, _assert_grounded, _converse,
)

pytestmark = pytest.mark.skipif(os.environ.get("REX_REAL_MODEL_TESTS") != "1",
                                reason="Real-model test: set REX_REAL_MODEL_TESTS=1 (needs Ollama).")


@pytest.fixture(autouse=True)
def translator():
    """The real translator (overrides tests/test_rex_grounding.py's stand-in)."""
    yield


@pytest.fixture(scope="module")
def real_rag(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real_catalogue")
    store = VectorStore(persist_directory=directory / "db")
    summary = pipeline.ingest_directory(DOCUMENTS, vector_store=store, metadata_path=directory / "meta.json",
                                        manifest_path=directory / "manifest.json")
    assert not summary["files_failed"]
    rag = InProcessRag(store)
    rag.evidence = EvidenceRetriever(Retriever(vector_store=store))   # real embeddings and reranker
    return rag


@pytest.mark.parametrize("messages", [EXACT_PROFILE_EN, EXACT_PROFILE_AR], ids=["en", "ar"])
def test_real_model_recommends_only_verified_catalogue_courses(real_rag, messages):
    agent = RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=real_rag)
    result = _converse(agent, messages)["result"]
    answer = result["answer"]
    print(f"\n--- real-model recommendation ({messages[1]}) ---\n{answer}")

    records = real_rag.get_course_records()["records"]
    shown = _assert_grounded(answer, records)
    by_id = {r["course_id"]: r for r in records}
    assert result["profile"]["training_preference"] == "no_preference"
    if shown:
        for course_id, _ in shown:   # each fact line is the record's
            record = by_id[course_id]
            block = answer.split(f"({course_id})", 1)[1]
            assert f"{int(record['fee'])} EGP" in block and record["delivery_mode"] in block
        assert not re.search(r"Data-led|Basic Computer Literacy", answer)
    else:
        assert "No suitable course" in answer or "لم أجد دورة مناسبة" in answer
