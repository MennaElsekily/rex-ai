"""
Agent 1 (REX) recommendation grounding, end to end on real catalogue data:
REX graph -> knowledge-base tools (in-process, same functions as the RAG
MCP server) -> verified course records -> recommendation.

Live finding (2026-11-11), learner profile: Computer Engineer, goal Data
Analytics, Beginner, online or offline, budget 4000, 3 days per week. REX
recommended "Intermediate Data-led Labs (70 hours)" and "Basic Computer
Literacy" twice -- none of them courses. The index then in use (an
obsolete demonstration catalogue, since removed) headed each programme
"PROGRAMME 01 · <code>" and had no course fields, so REX had no course
list, and the model turned retrieved fragments into course names: a
"guided learning hours / level / labs" row and the "Entry requirements"
passages of two programmes.

Data used (the academy catalogue in knowledge_base/documents is the only
source of course facts):
- that catalogue, ingested by the real pipeline into a temporary store
  (deterministic fake embeddings);
- the same programme-record layout as the failure, filled with facts of
  that catalogue's own courses (DA-101, DA-201, MKT-301), so the layout
  stays covered.
Mocked: the chat model, the embedding search (a keyword stand-in with the
same result shape) and the Arabic->English query translator. The project
index is never opened.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.rex_assistant import discovery
from agents.rex_assistant.interface import RexAssistantAgent
from knowledge_base import chunker, language, pipeline
from knowledge_base.course_records import course_map
from knowledge_base.evidence import EvidenceRetriever, safe_course_records, safe_list_courses, safe_search
from knowledge_base.vector_store import VectorStore
from tests.test_rex_assistant import FakeLeadStore, FakeMessage

DOCUMENTS = Path(__file__).resolve().parents[1] / "knowledge_base" / "documents"
EXACT_PROFILE_EN = ["I want course recommendations", "Computer Engineer", "Data Analytics", "Beginner",
                    "Online or offline", "4000", "3 days per week"]
EXACT_PROFILE_AR = ["أريد ترشيح دورات تناسبني", "مهندس كمبيوتر", "تحليل البيانات", "مبتدئ",
                    "أونلاين أو حضوري", "4000", "3 أيام في الأسبوع"]
# What the model wrote in the live failure, plus a repeat, an invented id and a real course.
FABRICATED_REPLY = """Intermediate Data-led Labs :: Hands-on labs for 70 hours.
Basic Computer Literacy :: Basic computer literacy is a prerequisite for the other courses.
Basic Computer Literacy :: You need this first.
XX-999 :: An invented course.
DA-101 :: It teaches the spreadsheet analysis your Data Analytics goal starts with. You already know computers.
DA-101 :: Listed twice.
DA-201 :: It is the next step after DA-101."""
_HEADER = re.compile(r"^\*\*(?P<name>[^*]+)\*\* \((?P<id>[^)]+)\)$", re.MULTILINE)


def _fake_embed(texts, *args, **kwargs):
    return [[byte / 255 for byte in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


class KeywordRetriever:
    """Stands in for the embedding + reranker retriever: same result shape, word-overlap ranking."""

    def __init__(self, vector_store, top_k=5):
        self.vector_store, self.top_k = vector_store, top_k

    def retrieve(self, query, retrieval_query=None, metadata_filter=None):
        words = {w for w in re.findall(r"[a-z]{4,}", (retrieval_query or query).casefold())}
        data = self.vector_store.collection.get(include=["metadatas", "documents"], where=metadata_filter)
        scored = []
        for metadata, text in zip(data["metadatas"], data["documents"]):
            score = sum(1 for w in words if w in text.casefold())
            if score:
                scored.append((score, -int(metadata.get("chunk_id") or 0), metadata, text))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [{"text": text, "metadata": metadata, "source": metadata.get("source"), "page": metadata.get("page"),
                 "section": metadata.get("section"), "chunk_id": metadata.get("chunk_id"), "distance": 0.0,
                 "relevance_score": float(score)} for score, _, metadata, text in scored[:self.top_k]]


class InProcessRag:
    """The RAG MCP server's three tools (agents/rag_mcp_server.py), in-process."""

    def __init__(self, store):
        self.evidence = EvidenceRetriever(KeywordRetriever(store))
        self.searches = []

    def search_knowledge_base(self, question, retrieval_query=None, metadata_filter=None, course_ids=None, top_k=None):
        result = safe_search(self.evidence, question, retrieval_query=retrieval_query, metadata_filter=metadata_filter,
                             course_ids=course_ids, top_k=top_k)
        self.searches.append(result)
        return result

    def list_courses(self):
        return safe_list_courses(self.evidence)

    def get_course_records(self, query=None, level=None):
        return safe_course_records(self.evidence, query=query, level=level)

    def close(self):
        pass


class ScriptedLLM:
    """Profile extraction -> `profile_json`; choosing from the catalogue -> `choice`."""

    def __init__(self, choice, profile_json="{}"):
        self.choice, self.profile_json = choice, profile_json
        self.selection_prompts = []

    def invoke(self, messages):
        last = messages[-1].content
        if "Extract any of these fields" in last:
            facts = self.profile_json(last) if callable(self.profile_json) else self.profile_json
            return FakeMessage(facts)
        if "CATALOGUE COURSES" in last:
            self.selection_prompts.append(last)
            return FakeMessage(self.choice)
        return FakeMessage("General reply.")


class EnglishTranslator:
    def translate_to_english(self, text):
        return "courses for a beginner in data analytics, computer engineer background"


@pytest.fixture(autouse=True)
def translator():
    language._translator_singleton = EnglishTranslator()
    yield
    language._translator_singleton = None


@pytest.fixture(scope="module")
def catalogue_store(tmp_path_factory):
    """The real academy catalogue (knowledge_base/documents), ingested into a temporary store."""
    directory = tmp_path_factory.mktemp("catalogue")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "embed_texts", _fake_embed)
        mp.setattr(chunker, "embed_texts", _fake_embed)
        store = VectorStore(persist_directory=directory / "db")
        summary = pipeline.ingest_directory(DOCUMENTS, vector_store=store, metadata_path=directory / "meta.json",
                                            manifest_path=directory / "manifest.json")
    assert not summary["files_failed"] and summary["chunks_stored"] > 0
    return store


def _course_list():
    """{id: (name, delivery, fee)} from knowledge_base/documents/course_list.md."""
    lines = re.findall(r"^- ([A-Z]+-\d+) (.+?): (\w+), (\d+) EGP$", (DOCUMENTS / "course_list.md").read_text("utf-8"),
                       re.MULTILINE)
    return {cid: (name, delivery, float(fee)) for cid, name, delivery, fee in lines}


def _converse(agent, messages):
    state, response = None, None
    for text in messages:
        response = agent.handle_message(text, state=state)
        state = response["state"]
    return response


def _recommend(store, choice, messages=EXACT_PROFILE_EN, profile_json="{}"):
    llm = ScriptedLLM(choice, profile_json)
    rag = InProcessRag(store)
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        response = _converse(RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=rag), messages)
    return response, llm, rag


def _shown_courses(answer):
    return [(m.group("id"), m.group("name")) for m in _HEADER.finditer(answer)]


def _assert_grounded(answer, records):
    """Every course shown is a catalogue record, by its exact id and name, once."""
    by_id = {r["course_id"]: r for r in records}
    shown = _shown_courses(answer)
    for course_id, name in shown:
        assert course_id in by_id, f"{course_id} is not in the catalogue"
        assert name == by_id[course_id]["course_name"], f"{name!r} is not the catalogue name of {course_id}"
    assert len({cid for cid, _ in shown}) == len(shown), "a course is listed twice"
    return shown


# --------------------------------------------------------------------
# The catalogue records the knowledge base returns
# --------------------------------------------------------------------

def test_the_verified_records_are_exactly_the_academy_catalogue(catalogue_store):
    records = InProcessRag(catalogue_store).get_course_records()["records"]
    expected = _course_list()

    assert len(expected) == 14
    assert {r["course_id"]: r["course_name"] for r in records} == {cid: v[0] for cid, v in expected.items()}
    for record in records:
        name, delivery, fee = expected[record["course_id"]]
        assert record["delivery_mode"].startswith(delivery) and record["fee"] == fee
        assert record["fee_currency"] == "EGP" and record["level"] in ("Beginner", "Intermediate", "Advanced")


def test_prerequisite_courses_are_only_those_the_catalogue_lists(catalogue_store):
    records = {r["course_id"]: r for r in InProcessRag(catalogue_store).get_course_records()["records"]}

    assert records["DA-101"]["prerequisite_courses"] == [] and records["DA-101"]["prerequisites_text"] == "None"
    assert records["DA-201"]["prerequisite_courses"] == ["DA-101"]
    assert all(set(r["prerequisite_courses"]) <= set(records) for r in records.values())


# --------------------------------------------------------------------
# The exact learner profile, end to end
# --------------------------------------------------------------------

def test_exact_profile_recommendation_shows_only_catalogue_courses_with_catalogue_facts(catalogue_store):
    response, llm, rag = _recommend(catalogue_store, FABRICATED_REPLY)
    result = response["result"]
    answer = result["answer"]
    records = rag.get_course_records()["records"]

    assert result["profile"]["training_preference"] == "no_preference"   # "Online or offline" is either
    assert result["profile"]["career_goal"] == "Data Analytics"
    shown = _assert_grounded(answer, records)
    assert shown == [("DA-101", "Excel for Business Analysis")]   # DA-201 is Intermediate: a beginner course fits
    assert "Data-led" not in answer and "Basic Computer Literacy" not in answer and "XX-999" not in answer
    block = answer.split("**Excel for Business Analysis** (DA-101)")[1]
    assert "- Level: Beginner\n" in block
    assert "- Duration: 24 hours\n" in block
    assert "- Delivery: Offline (on campus, Cairo)\n" in block            # no preference: nothing to compare
    assert "- Fee: 3200 EGP (within your budget of 4000)\n" in block
    assert "- Entry requirements: Basic computer skills. No prior Excel experience needed.\n" in block
    assert "- Prerequisite courses: None\n" in block
    why = re.search(r"- Why it fits you: (.*)", block).group(1)
    assert "prerequisite" not in why.casefold() and why.startswith("It teaches the spreadsheet analysis")
    assert result["sources"] and all(s["course_id"] == "DA-101" for s in result["sources"])
    assert "CATALOGUE COURSES" in llm.selection_prompts[0] and "DA-101 | Excel for Business Analysis" in llm.selection_prompts[0]


def test_exact_profile_in_arabic_follows_the_same_rules(catalogue_store):
    reply = "DA-101 :: تعلّمك هذه الدورة تحليل البيانات في الجداول، وهي بداية مناسبة لهدفك.\nDA-101 :: مكرر"
    response, _, rag = _recommend(catalogue_store, reply, messages=EXACT_PROFILE_AR)
    answer = response["result"]["answer"]

    assert response["result"]["profile"]["training_preference"] == "no_preference"
    assert response["result"]["profile"]["career_goal"] == "تحليل البيانات"
    assert _assert_grounded(answer, rag.get_course_records()["records"]) == [("DA-101", "Excel for Business Analysis")]
    assert "- المستوى: Beginner" in answer
    assert "- الرسوم: 3200 EGP (ضمن ميزانيتك: 4000)" in answer
    assert "- متطلبات الالتحاق: Basic computer skills. No prior Excel experience needed." in answer
    assert answer.startswith("بناءً على ملفك")


def test_an_intermediate_course_for_a_beginner_states_the_catalogue_entry_requirements(catalogue_store):
    response, _, rag = _recommend(catalogue_store, "DA-201 :: It covers SQL and dashboards for your goal.")
    answer = response["result"]["answer"]

    assert _assert_grounded(answer, rag.get_course_records()["records"]) == [("DA-201", "SQL and Power BI for Data Analysts")]
    assert "- Level: Intermediate (above your beginner level: check the entry requirements below" in answer
    assert "- Entry requirements: Confident Excel use (DA-101 or equivalent). No SQL experience needed." in answer
    assert "- Prerequisite courses: DA-101 Excel for Business Analysis" in answer   # stated by the catalogue
    assert "- Fee: 7800 EGP (above your budget of 4000)" in answer


def test_fragments_only_give_the_retrieved_catalogue_courses_never_the_fragments(catalogue_store):
    reply = "Intermediate Data-led Labs :: Labs.\nBasic Computer Literacy :: Literacy."
    response, _, rag = _recommend(catalogue_store, reply)
    answer = response["result"]["answer"]

    shown = _assert_grounded(answer, rag.get_course_records()["records"])
    retrieved = {c["course_id"] for s in rag.searches for c in s["chunks"] if c.get("course_id")}
    assert {cid for cid, _ in shown} <= retrieved
    assert "Data-led" not in answer and "Literacy" not in answer.split("- Entry")[0]


def test_when_no_course_fits_rex_says_so(catalogue_store):
    response, _, _ = _recommend(catalogue_store, "NONE")

    assert response["result"]["answer"].startswith(
        "No suitable course was found in the current academy catalogue for your profile.")
    assert response["result"]["consent_pending"] is False


def test_without_verified_course_records_nothing_is_recommended(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "db")
    store.add_chunks([{"text": "Entry requirements\n\nBasic computer literacy.", "chunk_id": 1, "source": "x.pdf",
                       "page": 1, "section": "Entry requirements"}], _fake_embed(["x"]))

    response, llm, _ = _recommend(store, FABRICATED_REPLY)

    assert response["result"]["answer"].startswith("I can't recommend a specific course right now")
    assert llm.selection_prompts == []   # the model is not asked to choose from nothing


def test_a_short_reply_fills_only_the_question_asked():
    # Live finding (Arabic run): extraction copied "computer engineer" into career_goal too,
    # so the career goal was never asked.
    def extraction(prompt):
        if prompt.rstrip().endswith("User message: Computer Engineer"):
            return '{"background": "computer engineer", "career_goal": "computer engineer"}'
        return "{}"

    llm = ScriptedLLM("NONE", profile_json=extraction)
    with patch("agents.rex_assistant.graph._get_llm", return_value=llm):
        agent = RexAssistantAgent(lead_store=FakeLeadStore(), mcp_client=MagicMock())
        response = _converse(agent, ["I want course recommendations", "Computer Engineer"])

    assert response["result"]["profile"] == {"background": "Computer Engineer"}
    assert response["result"]["pending_field"] == "career_goal"


@pytest.mark.parametrize("reply, expected", [
    ("Online or offline", "no_preference"), ("أونلاين أو حضوري", "no_preference"), ("either", "no_preference"),
    ("online", "online"), ("online only, not offline", "online"), ("offline, no online please", "offline"),
    ("حضوري", "offline"), ("both", "hybrid"),
])
def test_training_preference_replies(reply, expected):
    assert discovery.normalize_answer("training_preference", reply) == expected


# --------------------------------------------------------------------
# The programme-record layout (the layout of the failure), with facts of
# the academy catalogue's own courses (knowledge_base/documents)
# --------------------------------------------------------------------

DA_FILE = "programme_layout_data_analytics.pdf"
MK_FILE = "programme_layout_digital_marketing.pdf"
# (file, chunk id, page, section, text). Delivery and fees are deliberately
# not stated in this layout: they must come out as "not stated".
LAYOUT_CHUNKS = [
    (DA_FILE, 2, 1, "Data Analytics", "Data Analytics\n\nANDALUSIA ACADEMY · COURSE CATALOGUE · DATA ANALYTICS"),
    (DA_FILE, 14, 3, "Page 3", "Page 3\n\nPROGRAMME 01 · DA-101"),
    (DA_FILE, 16, 3, "24 guided learning hours", "24 guided learning hours\n\n4 weeks Beginner Excel labs."),
    (DA_FILE, 18, 3, "Intended audience",
     "Intended audience\n\nAccountants, administrative staff, students and anyone who reports numbers at work."),
    (DA_FILE, 19, 3, "Entry requirements", "Entry requirements\n\nBasic computer skills. No prior Excel experience needed."),
    (DA_FILE, 53, 4, "Record and provenance",
     "Record and provenance\n\nDepartment: Data Analytics | Skill level: Beginner | "
     "Programme: Excel for Business Analysis | Source file: programme_layout_data_analytics.pdf."),
    (DA_FILE, 55, 5, "Page 5", "Page 5\n\nPROGRAMME 02 · DA-201"),
    (DA_FILE, 57, 5, "48 guided learning hours", "48 guided learning hours\n\n8 weeks"),
    (DA_FILE, 60, 5, "Intended audience",
     "Intended audience\n\nAnalysts, accountants and Excel users who want to move into data analysis."),
    (DA_FILE, 61, 5, "Entry requirements",
     "Entry requirements\n\nConfident Excel use (DA-101 or equivalent). No SQL experience needed."),
    (DA_FILE, 94, 6, "Record and provenance",
     "Record and provenance\n\nDepartment: Data Analytics | Skill level: Intermediate | "
     "Programme: SQL and Power BI for Data Analysts | Source file: programme_layout_data_analytics.pdf."),
    (MK_FILE, 61, 5, "Page 5", "Page 5\n\nPROGRAMME 03 · MKT-301"),
    (MK_FILE, 63, 5, "48 guided learning hours",
     "48 guided learning hours\n\n8 weeks Advanced Dashboards, experiments and attribution labs."),
    (MK_FILE, 65, 5, "Intended audience",
     "Intended audience\n\nMarketers and analysts responsible for campaign performance and reporting."),
    (MK_FILE, 66, 5, "Entry requirements",
     "Entry requirements\n\nExperience running campaigns and basic spreadsheet skills. MKT-201 or DA-101 recommended."),
    (MK_FILE, 108, 6, "Record and provenance",
     "Record and provenance\n\nDepartment: Digital Marketing | Skill level: Advanced | "
     "Programme: Marketing Analytics and Campaign Optimization | Source file: programme_layout_digital_marketing.pdf."),
]


def _layout_metadata(file, chunk_id, page, section):
    return {"document_id": f"knowledge_base/test_layout/{file}", "source": f"knowledge_base/test_layout/{file}",
            "filename": file, "chunk_id": chunk_id, "page": page, "section": section}


@pytest.fixture
def layout_store(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "db")
    chunks = [{"text": text, **_layout_metadata(file, cid, page, section)}
              for file, cid, page, section, text in LAYOUT_CHUNKS]
    store.add_chunks(chunks, _fake_embed([c["text"] for c in chunks]))
    return store


def test_programme_records_in_running_text_become_verified_records():
    records, owners = course_map([_layout_metadata(*c[:4]) for c in LAYOUT_CHUNKS], [c[4] for c in LAYOUT_CHUNKS])
    by_id = {r["course_id"]: r for r in records}

    assert set(by_id) == {"DA-101", "DA-201", "MKT-301"}
    assert by_id["DA-101"]["course_name"] == "Excel for Business Analysis"
    assert by_id["DA-101"]["level"] == "Beginner" and by_id["DA-101"]["total_hours"] == 24
    assert by_id["DA-101"]["entry_requirements"] == "Basic computer skills. No prior Excel experience needed."
    assert by_id["DA-201"]["level"] == "Intermediate" and by_id["DA-201"]["total_hours"] == 48
    assert by_id["MKT-301"]["course_name"] == "Marketing Analytics and Campaign Optimization"
    assert by_id["MKT-301"]["total_hours"] == 48
    # Not stated in this layout: never filled in.
    assert all(r["delivery_mode"] is None and r["fee"] is None and r["prerequisite_courses"] == [] for r in records)
    # The kinds of fragments REX once turned into "courses" belong to their programmes.
    assert owners[(f"knowledge_base/test_layout/{MK_FILE}", "63")]["course_id"] == "MKT-301"
    assert owners[(f"knowledge_base/test_layout/{DA_FILE}", "19")]["course_id"] == "DA-101"
    assert (f"knowledge_base/test_layout/{DA_FILE}", "2") not in owners   # the cover belongs to no programme


def test_a_programme_heading_without_a_stated_name_is_not_a_course():
    records, _ = course_map([_layout_metadata(DA_FILE, 1, 1, "Page 1"),
                             _layout_metadata(DA_FILE, 2, 1, "Entry requirements")],
                            ["Page 1\n\nPROGRAMME 09 · DA-901", "Entry requirements\n\nBasic computer skills."])

    assert records == []


def test_search_results_name_the_programme_each_passage_belongs_to(layout_store):
    result = InProcessRag(layout_store).search_knowledge_base("entry requirements", "basic computer skills")

    labelled = {c["text"]: (c["course_id"], c["course_name"]) for c in result["chunks"]}
    assert labelled["Entry requirements\n\nBasic computer skills. No prior Excel experience needed."] == (
        "DA-101", "Excel for Business Analysis")


def test_the_reported_failure_cannot_recur_on_a_programme_record_catalogue(layout_store):
    reply = ("**Intermediate Data-led Labs** (70 hours)\n- Why it fits you: hands-on.\n\n"
             "**Basic Computer Literacy**\n- Why it fits you: a prerequisite.\n\n"
             "**Basic Computer Literacy**\n- Why it fits you: again.\n\n"
             "DA-101 :: It teaches spreadsheet analysis for your Data Analytics goal.")
    response, _, rag = _recommend(layout_store, reply)
    answer = response["result"]["answer"]

    shown = _assert_grounded(answer, rag.get_course_records()["records"])
    assert shown == [("DA-101", "Excel for Business Analysis")]
    assert "Data-led" not in answer and "Basic Computer Literacy" not in answer
    assert "- Delivery: online or offline is not stated in the catalogue\n" in answer
    assert "- Fee: not stated in the catalogue (so it cannot be checked against your budget of 4000)" in answer
    assert "- Level: Beginner\n" in answer and "- Duration: 24 hours" in answer


def test_rex_reaches_the_records_through_the_rag_tools_only():
    import ast

    # Read, not imported: importing the server opens the project's index.
    server = ast.parse((Path(__file__).resolve().parents[1] / "agents" / "rag_mcp_server.py").read_text("utf-8"))
    tools = {node.name for node in server.body if isinstance(node, ast.FunctionDef)
             and any("mcp.tool" in ast.unparse(d) for d in node.decorator_list)}
    assert {"search_knowledge_base", "list_courses", "get_course_records"} <= tools
    imports = [line.strip() for p in (Path(__file__).resolve().parents[1] / "agents" / "rex_assistant").glob("*.py")
               for line in p.read_text("utf-8").splitlines() if line.strip().startswith(("import ", "from "))]
    # Agent 1 talks to the knowledge base directly, never through the admin orchestrator...
    assert not [i for i in imports if "orchestrator" in i or "admin_service" in i]
    # ...and only through the RAG tools: no retrieval internals are imported.
    assert not [i for i in imports if "knowledge_base.evidence" in i or "knowledge_base.course_records" in i
                or "knowledge_base.retriever" in i or "knowledge_base.vector_store" in i]


# --------------------------------------------------------------------
# The model's "why it fits" sentence (live findings with qwen2.5:3b)
# --------------------------------------------------------------------

def test_a_reason_crediting_the_course_with_another_courses_topics_is_removed(catalogue_store):
    # Real-model run: Excel for Business Analysis "will help you ... by learning SQL and Power BI"
    # -- the topics of DA-201 SQL and Power BI for Data Analysts.
    reply = ("DA-101 :: As a Computer Engineer, this course will help you move into data analysis by learning SQL "
             "and Power BI. It builds the pivot tables and charts your Data Analytics goal starts with.")
    response, _, _ = _recommend(catalogue_store, reply)
    answer = response["result"]["answer"]

    assert "SQL" not in answer.split("- Prerequisite")[0] and "Power BI" not in answer
    assert "- Why it fits you: It builds the pivot tables and charts your Data Analytics goal starts with." in answer


def test_a_looping_reason_is_dropped_and_the_course_still_shown_from_its_record(catalogue_store):
    # Real-model run (Arabic): the reason looped "أ Divider" hundreds of times.
    reply = "DA-101 :: هذا الدورة يلبي احتياجاتك " + "أ Divider " * 200
    response, _, _ = _recommend(catalogue_store, reply, messages=EXACT_PROFILE_AR)
    answer = response["result"]["answer"]

    assert "Divider" not in answer and "لماذا تناسبك" not in answer
    assert "**Excel for Business Analysis** (DA-101)" in answer and "- الرسوم: 3200 EGP" in answer
    assert len(answer) < 2000
