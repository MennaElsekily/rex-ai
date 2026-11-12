"""
REX <-> RAG over MCP: the retrieval contract, bilingual query handling,
the three learner interaction modes and catalogue enumeration.

Data: the academy catalogue in knowledge_base/documents, ingested into a
temporary store (tests/test_rex_grounding.py's catalogue_store fixture).
The project's index is never opened.

Mocked here: the chat model, and -- except where a test says otherwise --
the embedding search (a keyword stand-in with the same result shape) and
the Arabic->English translator. The order-of-operations tests run the RAG
service's real retrieval code (knowledge_base.retriever.Retriever) with a
spy on the embedding call; the MCP test starts the REAL RAG MCP server as a
subprocess against a temporary index. Real-model runs are opt-in:
tests/test_rex_real_model_grounding.py and docs/MANUAL_TEST_PLAN.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.rex_assistant.interface import RexAssistantAgent
from knowledge_base import evidence as evidence_module, language, retriever as retriever_module
from knowledge_base.evidence import EvidenceRetriever
from knowledge_base.vector_store import VectorStore
from tests.test_rex_assistant import FakeLeadStore
from tests.test_rex_grounding import (  # noqa: F401  (catalogue_store is a fixture)
    EXACT_PROFILE_EN, InProcessRag, ScriptedLLM, _assert_grounded, _converse, _course_list, _fake_embed,
    _shown_courses, catalogue_store,
)

ROOT = Path(__file__).resolve().parents[1]


class Translator:
    """Arabic -> English stand-in; records what it was asked to translate."""

    def __init__(self, replies=None, fail=False):
        self.replies, self.fail, self.calls = replies or {}, fail, []

    def translate_to_english(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("translation model unavailable")
        return self.replies.get(text, "data analytics courses for beginners")


@pytest.fixture(autouse=True)
def translator():
    fake = Translator()
    language._translator_singleton = fake
    yield fake
    language._translator_singleton = None


def _agent(store, choice="NONE", profile_json="{}"):
    llm = ScriptedLLM(choice, profile_json)
    rag = InProcessRag(store)
    return RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=rag), llm, rag


def _say(agent, messages, state=None):
    response = None
    for text in messages:
        response = agent.handle_message(text, state=state)
        state = response["state"]
    return response


def _run(store, messages, choice="NONE"):
    agent, llm, rag = _agent(store, choice)
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _converse(agent, messages)
    return response, llm, rag


# --------------------------------------------------------------------
# Order of operations inside RAG: detect -> translate -> embed -> search
# --------------------------------------------------------------------

class SpyCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [float(len(set(q.casefold().split()) & set(t.casefold().split()))) for q, t in pairs]


@pytest.fixture
def real_retrieval(tmp_path, monkeypatch):
    """RAG's real Retriever on a temporary store; the embedding call is recorded (16-dim fake vectors)."""
    embedded = []

    def embed(texts, *args, **kwargs):
        embedded.extend(texts)
        return _fake_embed(texts)

    monkeypatch.setattr(retriever_module, "embed_texts", embed)
    monkeypatch.setattr(retriever_module, "_load_cross_encoder", lambda: SpyCrossEncoder())
    store = VectorStore(persist_directory=tmp_path / "db")
    chunks = [{"text": t, "chunk_id": n, "source": "catalogue.pdf", "page": 1, "section": "x"}
              for n, t in enumerate(["Course ID: DA-101\nExcel for Business Analysis\nEntry requirements: Basic computer skills",
                                     "Course ID: DA-201\nSQL and Power BI for Data Analysts\nFee (EGP): 7800"], start=1)]
    store.add_chunks(chunks, _fake_embed([c["text"] for c in chunks]))
    embedded.clear()
    return EvidenceRetriever(retriever_module.Retriever(vector_store=store, min_relevance_score=None)), embedded, store


def test_an_english_query_is_embedded_as_written_without_translation(real_retrieval, translator):
    evidence, embedded, store = real_retrieval

    result = evidence.search("What are the entry requirements of Excel for Business Analysis?")

    assert translator.calls == []
    assert embedded == ["What are the entry requirements of Excel for Business Analysis?"]
    assert result["translation"] == "not_needed" and result["query_language"] == "en"
    assert result["embedded_text"] == embedded[0]
    assert result["chunks"] and result["collection"] == store.collection_name


def test_an_arabic_query_is_translated_before_embedding_and_the_original_is_kept(real_retrieval, translator):
    evidence, embedded, store = real_retrieval
    translator.replies = {"ما هي متطلبات الالتحاق؟": "What are the entry requirements?"}

    result = evidence.search("ما هي متطلبات الالتحاق؟")

    assert translator.calls == ["ما هي متطلبات الالتحاق؟"]             # translated first...
    assert embedded == ["What are the entry requirements?"]             # ...then only the English is embedded
    assert result["question"] == "ما هي متطلبات الالتحاق؟"             # the original is returned unchanged
    assert result["query_language"] == "ar" and result["translation"] == "translated"
    assert result["embedded_text"] == "What are the entry requirements?"


def test_a_mixed_query_keeps_its_course_code_and_name_through_translation(real_retrieval, translator):
    evidence, embedded, _ = real_retrieval
    translator.replies = {"ما هي رسوم دورة ؟": "What are the fees of the course?"}

    result = evidence.search("ما هي رسوم دورة DA-201 SQL؟")

    assert translator.calls == ["ما هي رسوم دورة ؟"]                   # only the Arabic part is translated
    assert "DA-201" in embedded[0] and "SQL" in embedded[0]
    assert result["query_language"] == "mixed"


def test_a_failed_translation_retrieves_nothing_and_rex_says_so(real_retrieval, translator, catalogue_store):
    evidence, embedded, _ = real_retrieval
    translator.fail = True

    result = evidence.search("ما هي متطلبات الالتحاق؟")

    assert result["status"] == "translation_failed" and result["chunks"] == [] and embedded == []
    response, _, _ = _run(catalogue_store, ["ما هي سياسة الاسترداد؟"])
    assert response["result"]["answer"].startswith("عذرًا، لم أتمكن من تجهيز سؤالك")


def test_query_and_document_embeddings_share_model_and_dimensions(real_retrieval):
    evidence, _, store = real_retrieval
    stored = store.collection.get(limit=1, include=["embeddings"])["embeddings"][0]

    result = evidence.search("Excel for Business Analysis")

    assert result["embedding_dimensions"] == len(stored) == 16
    assert result["embedding_model"] == retriever_module.DEFAULT_MODEL_NAME


def test_a_query_against_an_index_built_with_another_model_fails_clearly(real_retrieval, monkeypatch):
    evidence, _, store = real_retrieval
    store.collection.update(ids=store.collection.get()["ids"],
                            metadatas=[{"embedding_model": "another/model", "chunk_id": n, "source": "catalogue.pdf"}
                                       for n in (1, 2)])

    with pytest.raises(ValueError, match="Rebuild it"):
        evidence.search("Excel for Business Analysis")


def test_every_search_response_carries_the_trace_fields(real_retrieval):
    evidence, _, _ = real_retrieval

    result = evidence.search("Excel for Business Analysis", top_k=1)

    for key in ("question", "query_language", "translation", "embedded_text", "embedding_model",
                "embedding_dimensions", "collection", "top_k", "chunks", "courses", "limitations", "status"):
        assert key in result
    chunk = result["chunks"][0]
    for key in ("chunk_id", "source", "page", "course_id", "course_name", "relevance_score", "distance", "text"):
        assert key in chunk
    assert len(result["chunks"]) == 1 and "not a complete list" in result["limitations"][0]


# --------------------------------------------------------------------
# The real RAG MCP server (subprocess, stdio) on a temporary index
# --------------------------------------------------------------------

def test_the_real_rag_mcp_server_serves_the_catalogue_tools(catalogue_store, monkeypatch):
    from agents.rex_assistant.mcp_client import RagMcpClient

    monkeypatch.setenv("REX_VECTOR_DB_DIR", catalogue_store.persist_directory)   # forwarded to the server
    client = RagMcpClient()
    try:
        tools = {tool["name"] for tool in client.list_tools()}
        listing = client.get_course_records()
        topic = client.get_course_records(query="What Data Analytics courses does the academy offer?")
        courses = client.list_courses()
    finally:
        client.close()

    assert {"search_knowledge_base", "list_courses", "get_course_records"} <= tools
    assert listing["status"] == "ok" and listing["complete"] is True
    assert {r["course_id"] for r in listing["records"]} == set(_course_list())        # the whole catalogue
    assert listing["total_in_catalogue"] == 14
    assert {r["course_id"] for r in topic["records"]} == {"DA-101", "DA-201", "DA-301"} and topic["complete"] is False
    assert {c["course_id"] for c in courses["courses"]} == set(_course_list())


# --------------------------------------------------------------------
# Mode 3: complete catalogue discovery (not a top-k search)
# --------------------------------------------------------------------

def test_what_courses_does_the_academy_offer_lists_every_verified_course(catalogue_store):
    response, llm, rag = _run(catalogue_store, ["What courses does the academy offer?"])
    answer = response["result"]["answer"]

    listed = re.findall(r"^\d+\. \*\*(.+?)\*\* \(([A-Z]+-\d+)\)", answer, re.MULTILINE)
    expected = _course_list()
    assert sorted(cid for _, cid in listed) == sorted(expected)                       # all 14, once each
    assert all(expected[cid][0] == name for name, cid in listed)                       # exact names
    assert answer.startswith("The academy catalogue lists 14 courses:")
    assert rag.searches == [] and llm.selection_prompts == []                          # no top-k search, no model
    assert "- Level" not in answer and "Beginner" in answer and "3200 EGP" in answer


def test_a_category_lists_only_the_matching_verified_courses(catalogue_store):
    response, _, _ = _run(catalogue_store, ["What Data Analytics courses does the academy offer?"])
    answer = response["result"]["answer"]

    listed = re.findall(r"\(([A-Z]+-\d+)\) —", answer)
    assert sorted(listed) == ["DA-101", "DA-201", "DA-301"]
    assert answer.startswith("3 of the catalogue's 14 courses match your request (topic: data, analytics):")


def test_a_level_narrows_the_listing(catalogue_store):
    response, _, _ = _run(catalogue_store, ["What beginner courses do you offer?"])

    listed = re.findall(r"\(([A-Z]+-\d+)\) —", response["result"]["answer"])
    assert sorted(listed) == sorted(cid for cid, v in _course_list().items()
                                    if cid in ("AI-101", "DA-101", "MKT-101", "MOB-101", "SW-101"))


def test_an_arabic_catalogue_request_is_translated_by_rag_and_answered_in_arabic(catalogue_store, translator):
    translator.replies = {"ما هي جميع الدورات التي تقدمها الأكاديمية؟": "What are all the courses offered by the academy?"}

    response, _, _ = _run(catalogue_store, ["ما هي جميع الدورات التي تقدمها الأكاديمية؟"])
    answer = response["result"]["answer"]

    assert answer.startswith("يضم كتالوج الأكاديمية 14 دورة:")
    assert len(re.findall(r"^\d+\. \*\*", answer, re.MULTILINE)) == 14
    assert translator.calls == ["ما هي جميع الدورات التي تقدمها الأكاديمية؟"]


def test_an_unknown_topic_is_reported_not_invented(catalogue_store, monkeypatch):
    monkeypatch.setattr(EvidenceRetriever, "_courses_found_by_search", lambda self, q, r: [])

    response, _, _ = _run(catalogue_store, ["What Kubernetes courses do you offer?"])

    assert response["result"]["answer"].startswith("No course in the current academy catalogue matches your request")
    assert "Kubernetes" not in _shown_courses(response["result"]["answer"])


# --------------------------------------------------------------------
# Mode 1: a specific course, with follow-ups; mode switching
# --------------------------------------------------------------------

def test_a_named_course_question_is_answered_from_that_courses_passages(catalogue_store):
    agent, llm, rag = _agent(catalogue_store)
    llm.choice = "unused"
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _converse(agent, ["Tell me more about SQL and Power BI for Data Analysts.",
                                     "What are its prerequisites?"])

    assert response["result"]["selected_course"]["course_id"] == "DA-201"
    last = rag.searches[-1]
    assert last["question"] == "What are its prerequisites?"                      # the original question...
    assert "DA-201 SQL and Power BI for Data Analysts" in last["retrieval_query"]   # ...searched with its course


def test_catalogue_then_recommendation_then_the_first_course(catalogue_store):
    agent, llm, rag = _agent(catalogue_store, choice="DA-101 :: It starts your Data Analytics path.")
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        listing = _say(agent, ["What courses does the academy offer?"])
        state = listing["state"]
        first_listed = state["course_focus"][0]["course_id"]
        recommendation = _say(agent, ["Recommend beginner courses for me", *EXACT_PROFILE_EN[1:]], state)
        follow_up = _say(agent, ["no", "Tell me more about the first course"], recommendation["state"])

    assert first_listed == "AI-101"                                                # the listing order (by department)
    assert "**Excel for Business Analysis** (DA-101)" in recommendation["result"]["answer"]
    assert follow_up["result"]["selected_course"]["course_id"] == "DA-101"        # the first RECOMMENDED course


def test_an_unknown_course_is_not_invented(catalogue_store):
    response, _, _ = _run(catalogue_store, ["Tell me more about the Kubernetes Administration course."])
    answer = response["result"]["answer"]

    assert response["result"]["selected_course"] is None
    assert "Kubernetes Administration" not in [name for _, name in _shown_courses(answer)]


# --------------------------------------------------------------------
# Mode 2 grounding: only courses the similarity search retrieved
# --------------------------------------------------------------------

def test_a_course_no_retrieved_passage_supports_cannot_be_recommended(catalogue_store, monkeypatch):
    marketing_only = {"status": "ok", "question": "q", "retrieval_query": "q", "query_language": "en",
                      "translation": "not_needed", "chunks": [
                          {"reference": 1, "text": "Digital Marketing Fundamentals ...", "course_id": "MKT-101",
                           "course_name": "Digital Marketing Fundamentals", "source": "catalogue_mkt.pdf"}],
                      "courses": [], "error": None}
    monkeypatch.setattr(InProcessRag, "search_knowledge_base", lambda self, *a, **k: marketing_only)

    response, llm, _ = _run(catalogue_store, EXACT_PROFILE_EN, choice="DA-101 :: Data analytics.")

    assert "DA-101" not in llm.selection_prompts[0].split("RETRIEVED")[0]          # not even offered to the model
    assert "Excel for Business Analysis" not in response["result"]["answer"]


def test_empty_retrieval_never_produces_a_course(catalogue_store, monkeypatch):
    nothing = {"status": "no_results", "question": "q", "chunks": [], "courses": [], "error": None}
    monkeypatch.setattr(InProcessRag, "search_knowledge_base", lambda self, *a, **k: nothing)

    response, llm, _ = _run(catalogue_store, EXACT_PROFILE_EN, choice="DA-101 :: Data analytics.")

    assert response["result"]["answer"].startswith("No suitable course was found")
    assert llm.selection_prompts == []


def test_the_model_sees_the_retrieved_passages_of_each_candidate(catalogue_store):
    response, llm, rag = _run(catalogue_store, EXACT_PROFILE_EN, choice="DA-101 :: It fits.")

    prompt = llm.selection_prompts[0]
    assert "RETRIEVED CATALOGUE PASSAGES" in prompt
    retrieved = rag.searches[-1]
    assert retrieved["top_k"] == 12
    first = next(c for c in retrieved["chunks"] if c["course_id"])
    assert f"({first['course_id']} {first['course_name']})" in prompt


# --------------------------------------------------------------------
# Architecture: REX reaches the knowledge base through MCP only
# --------------------------------------------------------------------

def test_rex_has_no_retrieval_pipeline_of_its_own():
    sources = {p.name: p.read_text("utf-8") for p in (ROOT / "agents" / "rex_assistant").glob("*.py")}
    imports = [line.strip() for text in sources.values() for line in text.splitlines()
               if line.strip().startswith(("import ", "from "))]
    forbidden = ("chromadb", "sentence_transformers", "knowledge_base.vector_store", "knowledge_base.retriever",
                 "knowledge_base.evidence", "knowledge_base.course_records", "knowledge_base.embedding_service",
                 "knowledge_base.pipeline", "knowledge_base.translator", "agents.orchestrator", "admin_service")
    assert not [i for i in imports if any(f in i for f in forbidden)]
    # Nor does REX build database filters: it passes course ids to the RAG tool.
    assert not any(re.search(r"field_course_id|\$in\b|\$or\b", text) for text in sources.values())


def test_an_arabic_follow_up_with_an_attached_pronoun_stays_on_the_course(catalogue_store, translator):
    # Real-model trace (Test H): "كم عدد ساعاتها؟" ("how many hours is IT?") after a question about
    # Excel for Business Analysis was searched without the course and answered about SW-101.
    agent, llm, rag = _agent(catalogue_store)
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _say(agent, ["ما هي متطلبات الالتحاق بدورة Excel for Business Analysis؟", "كم عدد ساعاتها؟"])

    assert response["result"]["selected_course"]["course_id"] == "DA-101"
    assert rag.searches[-1]["question"] == "كم عدد ساعاتها؟"
    assert "DA-101 Excel for Business Analysis" in rag.searches[-1]["retrieval_query"]


def test_an_arabic_profile_is_searched_as_one_arabic_sentence():
    # Live-index trace: English labels around Arabic values were embedded as
    # ":: Data analysis;: data analysis;: beginner;:;: Courses suitable for ...".
    from agents.rex_assistant.graph import _recommendation_query

    query = _recommendation_query({"background": "مهندس كمبيوتر", "career_goal": "تحليل البيانات",
                                   "skill_level": "beginner", "training_preference": "no_preference"})

    assert query == "دورات مناسبة لمتعلم هدفه المهني: تحليل البيانات، خلفيته: مهندس كمبيوتر، مستواه: مبتدئ"
    assert not re.search(r"[A-Za-z]", query)


def test_a_category_prefers_the_department_that_names_it(tmp_path):
    # Live-index trace: "Data Analytics" also listed AI programmes whose descriptions mention data and analysts.
    from tests.test_rex_grounding import LAYOUT_CHUNKS, _layout_metadata

    store = VectorStore(persist_directory=tmp_path / "db")
    chunks = [{"text": text, **_layout_metadata(f, cid, page, section)} for f, cid, page, section, text in LAYOUT_CHUNKS]
    ai = "programme_layout_ai.pdf"
    chunks += [
        {"text": "Page 3\n\nPROGRAMME 04 · AI-201", **_layout_metadata(ai, 14, 3, "Page 3")},
        {"text": "Programme summary\n\nHands-on machine learning for learners who already code in Python, from "
                 "data preparation to model selection.", **_layout_metadata(ai, 17, 3, "Programme summary")},
        {"text": "Intended audience\n\nDevelopers, analysts and graduates who know Python and want practical "
                 "machine-learning skills.", **_layout_metadata(ai, 18, 3, "Intended audience")},
        {"text": "Record and provenance\n\nDepartment: Artificial Intelligence | Skill level: Intermediate | "
                 "Programme: Machine Learning Practitioner", **_layout_metadata(ai, 53, 4, "Record")},
    ]
    store.add_chunks(chunks, _fake_embed([c["text"] for c in chunks]))

    result = EvidenceRetriever(retriever_module.Retriever(vector_store=store)).course_records(
        query="What Data Analytics courses does the academy offer?")

    assert {r["course_id"] for r in result["records"]} == {"DA-101", "DA-201"}
    assert result["total_in_catalogue"] == 4 and result["match_method"] == "topic"


def test_a_skill_level_is_never_stored_as_the_background(catalogue_store):
    # Real-model trace (Arabic): from "أنا مبتدئ ومهتم بتحليل البيانات" the model extracted
    # background = "مبتدئ" ("beginner"), so the real background was never asked.
    agent, llm, _ = _agent(catalogue_store, profile_json='{"background": "مبتدئ", "skill_level": "مبتدئ", '
                                                          '"interests": "تحليل البيانات"}')
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _say(agent, ["أنا مبتدئ ومهتم بتحليل البيانات. رشح لي دورات مناسبة."])

    profile = response["result"]["profile"]
    assert "background" not in profile and profile["skill_level"] == "beginner"
    assert response["result"]["pending_field"] == "background"


def test_a_course_question_also_gets_the_courses_catalogue_record(tmp_path):
    # Live-index trace: the "<n> guided learning hours" passage was retrieved, but the model
    # said the hours were not stated. The verified record (from the same documents) states them.
    from tests.test_rex_grounding import LAYOUT_CHUNKS, _layout_metadata

    store = VectorStore(persist_directory=tmp_path / "db")
    chunks = [{"text": text, **_layout_metadata(f, cid, page, section)} for f, cid, page, section, text in LAYOUT_CHUNKS]
    store.add_chunks(chunks, _fake_embed([c["text"] for c in chunks]))
    agent, llm, rag = _agent(store)
    prompts = []
    llm.invoke_original = llm.invoke

    def recording_invoke(messages):
        prompts.append(messages[-1].content)
        return llm.invoke_original(messages)

    llm.invoke = recording_invoke
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _say(agent, ["Tell me more about Excel for Business Analysis.", "How long does it take?"])

    assert response["result"]["selected_course"]["course_id"] == "DA-101"
    prompt = prompts[-1]
    assert "Catalogue record of DA-101 Excel for Business Analysis" in prompt
    assert "Duration: 24 hours" in prompt and "Fee: not stated in the catalogue" in prompt


def test_a_russian_word_in_an_arabic_answer_is_removed():
    # Real-model trace: "الدورة DA-101 Excel for Business Analysis تدريبية مدة занятияها 24 ساعة."
    from agents.rex_assistant.graph import generate_answer
    from tests.test_rex_answer_generation import FakeLLM, _evidence

    llm = FakeLLM("مدة занятияها 24 ساعة [1].")
    result = generate_answer(llm, _evidence(), "ar")

    assert len(llm.calls) == 2                                   # asked once more, with the language reminder
    assert "занятия" not in result["answer"] and "24 ساعة" in result["answer"]


def test_an_index_holding_documents_missing_from_the_documents_folder_is_refused(tmp_path):
    # An obsolete index came back with a git branch switch and was served as the catalogue.
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "catalogue_da_data_analytics.pdf").write_bytes(b"%PDF")
    store = VectorStore(persist_directory=tmp_path / "db")
    chunks = [{"text": "Course ID: DA-101\nExcel for Business Analysis", "chunk_id": 1,
               "source": "knowledge_base/documents/catalogue_da_data_analytics.pdf",
               "filename": "catalogue_da_data_analytics.pdf"},
              {"text": "PROGRAMME 01 · XX-101", "chunk_id": 1, "source": "knowledge_base/documents/old_catalogue.pdf",
               "filename": "old_catalogue.pdf"}]
    store.add_chunks(chunks, _fake_embed([c["text"] for c in chunks]))
    evidence = EvidenceRetriever(retriever_module.Retriever(vector_store=store), documents_dir=documents)

    assert evidence.stale_sources() == ["old_catalogue.pdf"]
    for result in (evidence.search("What courses are there?"), evidence.course_records(), evidence.list_courses()):
        assert result["status"] == "stale_index" and "python -m knowledge_base.pipeline" in result["error"]

    (documents / "old_catalogue.pdf").write_bytes(b"%PDF")     # the folder and the index agree again
    store.add_chunks(chunks[:1], _fake_embed([chunks[0]["text"]]))  # (any index write refreshes the check)
    assert evidence.stale_sources() == []
