"""
The rebuilt Agent 3 end to end through its LangGraph workflow
(curriculum_agent.py -> graph.py) with a stand-in model (FakeLLM) and the
real tools in-process on a temporary index, database and folders. No
real index, database, model or website is used.

Covers Workflow A (generation, follow-up questions, catalogue facts,
bounded repair, drafts on failure), revisions with version history,
Workflow B (suggestions), show/list, approval attempts, research limited
to approved domains, English and Arabic, and failure handling.
"""

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research import curriculum_agent, llm as llm_module
from agents.curriculum_research.curriculum_agent import CurriculumAgent
from agents.curriculum_research.llm import CurriculumModelError, structured
from agents.curriculum_research.schemas import RequestSlots
from agents.rfp_research.api_clients.tavily import WebPage, WebSearchResult
from tests.curriculum_fakes import (
    FakeLLM,
    NoModel,
    canned_answers,
    details_answer,
    extras_answer,
    in_process_tools,
    outline_answer,
)
from tests.curriculum_fixtures import DATA_STATUS

CONTEXT = {"conversation_id": "admin-1"}


@pytest.fixture
def tools(tmp_path):
    return in_process_tools(tmp_path)


@pytest.fixture
def conversations(tmp_path):
    return ConversationStore(tmp_path / "conversations.sqlite3")


def make_agent(tools, conversations, answers):
    fake = FakeLLM(answers)
    return CurriculumAgent(conversation_store=conversations, tools=tools, llm=fake), fake


def ask(agent, text, context=CONTEXT):
    return agent.process_request({"request_id": "r", "user_input": text, "context": context})


def stored(tools, curriculum_id, version=None):
    return tools.store.get(curriculum_id, version)


# --------------------------------------------------------------------
# Workflow A -- generation
# --------------------------------------------------------------------

def test_english_request_generates_validates_and_saves_for_review(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers())

    response = ask(agent, "Create an intermediate curriculum about cloud computing for IT staff, 40 hours")

    assert response["status"] == "success" and response["agent_name"] == "curriculum_research"
    result = response["result"]
    assert result["kind"] == "curriculum" and result["review_status"] == "pending_review"
    assert result["validation"]["valid"] and result["version"] == 1
    record = stored(tools, result["curriculum_id"])
    content = record["content"]
    assert content["language"] == "en" and content["total_hours"] == 40 and content["level"] == "Intermediate"
    assert sum(m["hours"] for m in content["modules"]) == 40
    # Three bounded structured calls; the prompts carry the request's values.
    assert fake.steps() == ["CurriculumOutline", "ModuleDetailList", "CurriculumExtras"]
    outline_prompt = fake.prompts("CurriculumOutline")[0]
    assert "Total hours: 40" in outline_prompt and "Intermediate" in outline_prompt
    assert "Write every text value in English" in outline_prompt
    # The reply is readable and honest about approval and research.
    message = result["message"]
    assert result["curriculum_id"] in message and "pending review" in message
    assert "only an administrator can approve" in message
    assert "No external research sources were used" in message


def test_arabic_request_gets_an_arabic_curriculum_and_reply(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers("ar"))

    response = ask(agent, "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين")

    result = response["result"]
    assert response["status"] == "success" and result["validation"]["valid"]
    assert stored(tools, result["curriculum_id"])["content"]["language"] == "ar"
    assert "Write every text value in Arabic" in fake.prompts("CurriculumOutline")[0]
    assert "تم إعداد مسودة المنهج" in result["message"] and "بانتظار المراجعة" in result["message"]


def test_explicit_output_language_overrides_the_request_language(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers("en"))

    response = ask(agent, "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين باللغة الإنجليزية")

    assert stored(tools, response["result"]["curriculum_id"])["content"]["language"] == "en"
    assert "تم إعداد مسودة المنهج" in response["result"]["message"]  # the reply stays in Arabic


def test_missing_essentials_are_asked_once_together_then_remembered(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers(hours=(10, 10, 10)))

    first = ask(agent, "Create a curriculum about Kubernetes")
    assert first["result"]["kind"] == "question"
    assert first["result"]["missing"] == ["total_hours", "level"]
    assert "How many training hours" in first["result"]["message"] and "Topic: Kubernetes" in first["result"]["message"]
    assert fake.calls == []  # nothing generated, nothing saved
    assert tools.store.list() == []

    second = ask(agent, "30 hours")
    assert second["result"]["kind"] == "question" and second["result"]["missing"] == ["level"]

    third = ask(agent, "use defaults")
    content = stored(tools, third["result"]["curriculum_id"])["content"]
    assert content["level"] == "Beginner" and content["total_hours"] == 30
    assert any("default Beginner level" in a for a in content["assumptions"])
    assert "Topic: Kubernetes" in fake.prompts("CurriculumOutline")[0] or "Kubernetes" in fake.prompts(
        "CurriculumOutline")[0]


def test_catalogue_course_facts_fill_the_request_and_are_labelled(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers(hours=(10, 15, 15)))

    response = ask(agent, "Create a curriculum based on DA-201")

    result = response["result"]
    assert result["kind"] == "curriculum"  # hours and level came from the catalogue: no question
    content = stored(tools, result["curriculum_id"])["content"]
    assert (content["total_hours"], content["level"]) == (40, "Intermediate")
    assert {e["course_id"] for e in content["academy_evidence"]} == {"DA-201"}
    assert DATA_STATUS in content["data_notice"]
    prompt = fake.prompts("CurriculumOutline")[0]
    assert "DA-201" in prompt and "SQL for Analysts" in prompt and DATA_STATUS in prompt
    assert "taken from catalogue course DA-201" in result["message"]


def test_a_similar_topic_is_related_not_the_same_course(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers())

    response = ask(agent, "Create a beginner curriculum about Cloud Computing, 40 hours")

    content = stored(tools, response["result"]["curriculum_id"])["content"]
    assert content["academy_evidence"] == []  # not the catalogue's CL-101, only related to it
    assert "Related existing course CL-101" in fake.prompts("CurriculumOutline")[0]


def test_unmeasurable_outcomes_are_repaired_once(tools, conversations):
    outline = outline_answer()
    outline["learning_outcomes"][0]["statement"] = "Understand cloud service models"
    repaired = {"learning_outcomes": [{"id": "LO1", "statement": "Compare cloud service models"}]}
    agent, fake = make_agent(tools, conversations, canned_answers(CurriculumOutline=outline, RepairPatch=repaired))

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")

    assert fake.steps().count("RepairPatch") == 1
    assert "outcome_not_measurable" in fake.prompts("RepairPatch")[0]
    assert response["result"]["review_status"] == "pending_review"
    assert stored(tools, response["result"]["curriculum_id"])["content"]["learning_outcomes"][0]["statement"] \
        == "Compare cloud service models"


def test_a_curriculum_still_invalid_after_one_repair_is_saved_as_a_draft(tools, conversations):
    outline = outline_answer()
    outline["learning_outcomes"][0]["statement"] = "Know the cloud"
    agent, fake = make_agent(tools, conversations, canned_answers(CurriculumOutline=outline,
                                                                  RepairPatch={"learning_outcomes": []}))

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")

    result = response["result"]
    assert fake.steps().count("RepairPatch") == 1  # bounded: one repair pass only
    assert result["review_status"] == "draft" and not result["validation"]["valid"]
    assert "saved as a draft" in result["message"] and "LO1" in result["message"]


def test_wrong_module_hours_are_rescaled_without_changing_the_requested_total(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers(hours=(10, 10, 10)))

    response = ask(agent, "Create a beginner curriculum about cloud computing, 45 hours")

    content = stored(tools, response["result"]["curriculum_id"])["content"]
    assert [m["hours"] for m in content["modules"]] == [15, 15, 15] and content["total_hours"] == 45
    assert "RepairPatch" not in fake.steps()  # arithmetic is fixed deterministically, not by the model
    assert any("rescaled proportionally" in a for a in content["assumptions"])
    assert response["result"]["review_status"] == "pending_review"


def test_accreditation_claims_keep_the_curriculum_a_draft(tools, conversations):
    outline = outline_answer()
    outline["overview"] = "This course is accredited by NCAAA."
    agent, _ = make_agent(tools, conversations, canned_answers(CurriculumOutline=outline, RepairPatch={}))

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")

    assert response["result"]["review_status"] == "draft"
    assert "accreditation_claim" in {e["code"] for e in response["result"]["validation"]["errors"]}


def test_no_capstone_request_is_respected(tools, conversations):
    agent, fake = make_agent(tools, conversations, canned_answers())

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours, without a capstone")

    content = stored(tools, response["result"]["curriculum_id"])["content"]
    assert content["capstone"] is None and content["capstone_omitted_reason"]
    assert "NO capstone" in fake.prompts("CurriculumExtras")[0]


def test_model_failure_is_reported_and_nothing_is_saved(tools, conversations):
    agent, _ = make_agent(tools, conversations, {"CurriculumOutline": ConnectionError("Ollama is not running")})

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")

    assert response["status"] == "error" and "language model" in response["error_message"]
    assert tools.store.list() == []


def test_topic_found_by_the_model_only_when_it_is_in_the_message(tools, conversations):
    # The rules find no topic here ("that helps ..."), so the model is asked.
    text = "Build a course that helps managers with budgeting"
    agent, fake = make_agent(tools, conversations, {"RequestSlots": {"topic": "budgeting", "total_hours": 99}})

    response = ask(agent, text)
    assert fake.steps() == ["RequestSlots"] and response["result"]["kind"] == "question"
    assert response["result"]["missing"] == ["total_hours", "level"]  # 99 hours was not in the message
    assert "Topic: budgeting" in response["result"]["message"]

    agent2, _ = make_agent(tools, conversations, {"RequestSlots": {"topic": "blockchain"}})
    response = ask(agent2, text, context={"conversation_id": "admin-2"})
    assert response["result"]["missing"][0] == "topic"  # an invented topic is ignored


# --------------------------------------------------------------------
# Revisions
# --------------------------------------------------------------------

@pytest.fixture
def drafted(tools, conversations):
    def run(extra=None, language="en"):
        answers = canned_answers(language, hours=(10, 15, 15))
        answers.update(extra or {})
        agent, fake = make_agent(tools, conversations, answers)
        text = ("Create a beginner curriculum about cloud computing, 40 hours" if language == "en"
                else "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين")
        first = ask(agent, text)
        return agent, fake, first["result"]["curriculum_id"]
    return run


def test_hours_revision_creates_a_new_version_and_keeps_the_old_one(tools, drafted):
    agent, fake, curriculum_id = drafted()

    response = ask(agent, "make it 30 hours")

    result = response["result"]
    assert (result["curriculum_id"], result["version"]) == (curriculum_id, 2)
    assert sum(m["hours"] for m in stored(tools, curriculum_id)["content"]["modules"]) == 30
    assert stored(tools, curriculum_id, 1)["content"]["total_hours"] == 40
    assert [v["version"] for v in tools.store.versions(curriculum_id)] == [1, 2]
    assert "40 to 30" in stored(tools, curriculum_id)["change_summary"]
    assert fake.steps().count("CurriculumOutline") == 1  # no regeneration for an hours change


def test_adding_a_module_keeps_the_total_hours(tools, drafted):
    new_module = {"title": "Cost management", "topics": ["Budgets"], "description": "Controlling cloud costs.",
                  "activities": [{"type": "exercise", "description": "Set a budget alert"}], "outcome_ids": ["LO1"]}
    agent, fake, curriculum_id = drafted({"NewModule": new_module})

    response = ask(agent, "add a module on cost management")

    content = stored(tools, curriculum_id)["content"]
    assert [m["title"] for m in content["modules"]][-1] == "Cost management" and content["modules"][-1]["id"] == "M4"
    assert content["total_hours"] == 40 and sum(m["hours"] for m in content["modules"]) == 40
    assert "cost management" in fake.prompts("NewModule")[0]
    assert response["result"]["version"] == 2


def test_removing_a_module_and_then_the_capstone(tools, drafted):
    agent, _, curriculum_id = drafted({"RepairPatch": [{}, {}]})

    ask(agent, "remove module 3")
    content = stored(tools, curriculum_id)["content"]
    assert len(content["modules"]) == 2 and content["total_hours"] == 40

    response = ask(agent, "remove the capstone")
    content = stored(tools, curriculum_id)["content"]
    assert content["capstone"] is None and response["result"]["version"] == 3


def test_adding_a_capstone_when_one_exists_changes_nothing(tools, drafted):
    agent, _, curriculum_id = drafted()

    response = ask(agent, "add a capstone project")

    assert response["result"]["kind"] == "no_change"
    assert len(tools.store.versions(curriculum_id)) == 1


def test_level_change_regenerates_from_the_current_curriculum(tools, drafted):
    agent, fake, curriculum_id = drafted({"CurriculumOutline": [outline_answer(), outline_answer(level="Advanced")],
                                          "ModuleDetailList": [details_answer()] * 2,
                                          "CurriculumExtras": [extras_answer()] * 2})

    response = ask(agent, "make it advanced")

    assert response["result"]["version"] == 2
    assert stored(tools, curriculum_id)["content"]["level"] == "Advanced"
    prompt = fake.prompts("CurriculumOutline")[1]
    assert "Requested change: Rewrite it for the Advanced level." in prompt
    assert "Current curriculum: Cloud Computing Essentials" in prompt


def test_translation_regenerates_in_the_other_language(tools, drafted):
    agent, fake, curriculum_id = drafted({"CurriculumOutline": [outline_answer(), outline_answer("ar")],
                                          "ModuleDetailList": [details_answer(), details_answer("ar")],
                                          "CurriculumExtras": [extras_answer(), extras_answer("ar")]})

    response = ask(agent, "translate it to Arabic")

    assert response["result"]["validation"]["valid"]
    assert stored(tools, curriculum_id)["content"]["language"] == "ar"
    assert stored(tools, curriculum_id, 1)["content"]["language"] == "en"
    assert "Write every text value in Arabic" in fake.prompts("CurriculumOutline")[1]


def test_arabic_revision_of_an_arabic_curriculum(tools, drafted):
    agent, _, curriculum_id = drafted(language="ar")

    response = ask(agent, "اجعلها 20 ساعة")

    assert response["result"]["version"] == 2 and "تم تعديل المنهج" in response["result"]["message"]
    assert stored(tools, curriculum_id)["content"]["total_hours"] == 20


def test_an_impossible_revision_is_refused_without_a_new_version(tools, drafted):
    agent, _, curriculum_id = drafted()

    response = ask(agent, "make it 2 hours")

    assert response["status"] == "error" and "cannot cover 3 modules" in response["error_message"]
    assert len(tools.store.versions(curriculum_id)) == 1


def test_follow_up_detection_needs_this_conversations_state(tools, drafted, conversations):
    agent, _, _ = drafted()

    assert agent.wants_follow_up({"user_input": "make it 30 hours", "context": CONTEXT})
    assert agent.wants_follow_up({"user_input": "translate it to Arabic", "context": CONTEXT})
    assert not agent.wants_follow_up({"user_input": "what courses do you offer?", "context": CONTEXT})
    assert not agent.wants_follow_up({"user_input": "make it 30 hours", "context": {"conversation_id": "other"}})


# --------------------------------------------------------------------
# Workflow B -- suggestions
# --------------------------------------------------------------------

def _suggestions():
    return {"suggestions": [
        {"title": "Advanced Data Analytics with Python", "department": "Data Analytics", "level": "Advanced",
         "target_audience": "Analysts who completed DA-201", "prerequisite_course_ids": ["DA-201", "XX-999"],
         "outcomes": ["Build predictive models"], "proposed_hours": 36, "modules_outline": ["Pandas", "Modelling"],
         "rationale": "The department has no Advanced course.", "gap_ids": ["G1", "G99"], "source_ids": ["SRC-fake"],
         "confidence": "medium"},
        {"title": "SQL for Analysts", "department": "Data Analytics", "level": "Intermediate",
         "target_audience": "Analysts", "prerequisite_course_ids": [], "outcomes": ["Write SQL queries with joins"],
         "proposed_hours": 40,
         "modules_outline": ["SELECT and joins", "Aggregations"], "rationale": "Popular topic.", "gap_ids": [],
         "source_ids": [], "confidence": "high"},
    ]}


def test_suggestions_cite_only_real_gaps_and_flag_duplicates(tools, conversations):
    agent, fake = make_agent(tools, conversations, {"SuggestionList": _suggestions()})

    response = ask(agent, "Suggest new courses for the Data Analytics department")

    result = response["result"]
    assert result["kind"] == "suggestions"
    first, second = result["suggestions"]
    gap_ids = {g["id"] for g in result["gaps"]}
    assert first["gap_ids"] == [g for g in ["G1"] if g in gap_ids] and "G99" not in first["gap_ids"]
    assert first["source_ids"] == [] and first["prerequisite_course_ids"] == ["DA-201"]
    assert first["basis"] == "catalogue_gap"
    # The second repeats an existing course: flagged, listed last, confidence lowered.
    assert second["duplicate"] and second["overlaps"][0]["course_id"] == "DA-201"
    assert second["basis"] == "model_proposal" and second["confidence"] == "low"
    prompt = fake.prompts("SuggestionList")[0]
    assert "DA-101" in prompt and "No Advanced course in Data Analytics" in prompt and DATA_STATUS in prompt
    assert "SQL for Analysts" in prompt
    message = result["message"]
    assert DATA_STATUS in message and "not verified academy or market needs" in message
    assert "likely duplicate" in message and "generate suggestion 1" in message


def test_a_suggestion_can_be_turned_into_a_curriculum(tools, conversations):
    answers = canned_answers(hours=(12, 12, 12))
    answers["SuggestionList"] = _suggestions()
    agent, fake = make_agent(tools, conversations, answers)
    ask(agent, "Suggest new courses for the Data Analytics department")

    response = ask(agent, "generate suggestion 1")

    record = stored(tools, response["result"]["curriculum_id"])
    assert record["origin"] == "suggestion" and record["content"]["total_hours"] == 36
    assert record["content"]["level"] == "Advanced"
    assert "Topic: Advanced Data Analytics with Python" in fake.prompts("CurriculumOutline")[0]
    assert any("catalogue suggestion" in a for a in record["content"]["assumptions"])
    assert "There is no suggestion 7" in ask(agent, "generate suggestion 7")["result"]["message"]


def test_arabic_suggestion_request(tools, conversations):
    agent, fake = make_agent(tools, conversations, {"SuggestionList": {"suggestions": []}})

    response = ask(agent, "اقترح دورات جديدة للأكاديمية")

    assert response["result"]["kind"] == "suggestions" and "اقتراحات دورات" in response["result"]["message"]
    assert "Write every text value in Arabic" in fake.prompts("SuggestionList")[0]


# --------------------------------------------------------------------
# Research (approved domains only)
# --------------------------------------------------------------------

def test_research_sources_from_approved_domains_are_cited(tmp_path, conversations):
    def search(query, **kwargs):
        assert kwargs["include_domains"] == ["nist.gov"]
        return WebSearchResult(status="success", message="ok", query=query, pages=[
            WebPage(title="NICE Framework", url="https://www.nist.gov/nice", content="Cloud security roles",
                    source_domain="nist.gov"),
            WebPage(title="Blog", url="https://blog.example/cloud", content="Opinion", source_domain="blog.example")])

    tools = in_process_tools(tmp_path, research_domains=["nist.gov"], search=search)
    probe = tools.search_curriculum_sources("cloud security")
    source_id = probe["sources"][0]["source_id"]
    agent, fake = make_agent(tools, conversations, canned_answers(
        CurriculumExtras=extras_answer(source_ids=(source_id, "SRC-invented"))))

    response = ask(agent, "Create a beginner curriculum about cloud security, 40 hours")

    content = stored(tools, response["result"]["curriculum_id"])["content"]
    assert [r["url"] for r in content["references"]] == ["https://www.nist.gov/nice"]  # the blog was discarded
    assert content["competencies"][0]["source_ids"] == [source_id]  # the invented id was removed
    assert source_id in fake.prompts("CurriculumOutline")[0]
    assert "citation(s) to sources that were not retrieved" in response["result"]["message"]
    assert response["result"]["validation"]["valid"]


# --------------------------------------------------------------------
# Show, approval attempts, other replies
# --------------------------------------------------------------------

def test_show_and_list(tools, drafted):
    agent, _, curriculum_id = drafted()

    shown = ask(agent, "show the curriculum")
    assert shown["result"]["curriculum_id"] == curriculum_id and "Stored curriculum" in shown["result"]["message"]
    listed = ask(agent, "list curricula")
    assert listed["result"]["kind"] == "list" and curriculum_id in listed["result"]["message"]
    by_id = ask(agent, f"status of {curriculum_id}", context={"conversation_id": "someone-else"})
    assert by_id["result"]["curriculum_id"] == curriculum_id
    missing = ask(agent, "show CUR-000000000000")
    assert missing["status"] == "error"


@pytest.mark.parametrize("text", ["Approve the curriculum", "اعتمد المنهج"])
def test_the_agent_never_approves(tools, drafted, text):
    agent, _, curriculum_id = drafted()

    response = ask(agent, text)

    assert response["result"]["kind"] == "approve_attempt"
    assert ("Only an administrator" in response["result"]["message"]
            or "يعتمد المسؤول فقط" in response["result"]["message"])
    assert stored(tools, curriculum_id)["status"] == "pending_review"
    trail = tools.store.audit_trail(curriculum_id)
    assert len(trail) == 1 and trail[0]["new_status"] == "pending_review"  # only the agent's save


def test_alignment_without_frameworks_is_unable_to_determine(tools, conversations):
    agent, fake = make_agent(tools, conversations, {})

    response = ask(agent, "Align the curriculum with the Saudi NQF")

    assert response["result"]["alignment_status"] == "unable_to_determine"
    assert "unable to determine" in response["result"]["message"] and fake.calls == []


def test_sending_is_refused_and_unclear_messages_get_help(tools, conversations):
    agent, fake = make_agent(tools, conversations, {})

    assert ask(agent, "email the curriculum to the dean")["result"]["kind"] == "unsupported"
    help_reply = ask(agent, "hello")
    assert help_reply["result"]["kind"] == "help" and "Curriculum Agent" in help_reply["result"]["message"]
    assert fake.calls == []


def test_empty_input_and_tool_failures_are_errors_not_crashes(conversations):
    class BrokenTools:
        def __getattr__(self, name):
            def fail(**_):
                raise RuntimeError("server unavailable")
            return fail

    agent = CurriculumAgent(conversation_store=conversations, tools=BrokenTools(), llm=FakeLLM({}))
    assert ask(agent, "   ")["status"] == "error"
    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")
    assert response["status"] == "error" and "server unavailable" in response["error_message"]


# --------------------------------------------------------------------
# Defaults and the model helper
# --------------------------------------------------------------------

def test_tests_never_reach_the_real_model_or_mcp_server():
    agent = CurriculumAgent()
    assert isinstance(agent.llm, NoModel)
    assert type(agent.tools).__name__ == "CurriculumToolbox"  # in-process, temporary storage
    assert agent.tools.store.db_path.name == "curricula.sqlite3" and "curriculum_agent" in str(agent.tools.store.db_path)


def test_the_model_is_configurable_and_defaults_to_rex_model(monkeypatch):
    from agents.config import rex_model
    monkeypatch.delenv("CURRICULUM_LLM_MODEL", raising=False)
    monkeypatch.setattr(llm_module, "get_setting", lambda name, default: default)
    assert llm_module.curriculum_model() == rex_model()
    monkeypatch.setattr(llm_module, "get_setting", lambda name, default: "qwen3.5:4b")
    assert llm_module.curriculum_model() == "qwen3.5:4b"


def test_model_calls_are_bounded(monkeypatch):
    """A looping model cannot hang a request: output tokens and wait time are capped."""
    monkeypatch.setattr(llm_module, "_supports_thinking", lambda model: False)  # no Ollama call in tests
    chat = llm_module.get_llm.__wrapped__("any-model")
    assert chat.num_predict == llm_module.MAX_OUTPUT_TOKENS and chat.temperature == 0
    assert chat.client_kwargs["timeout"] == llm_module.CALL_TIMEOUT_SECONDS


def test_a_timeout_is_not_retried():
    class Timeout(Exception):
        pass

    Timeout.__name__ = "ReadTimeout"
    fake = FakeLLM({"RequestSlots": [Timeout("timed out"), {"topic": "SQL"}]})
    with pytest.raises(CurriculumModelError):
        structured(fake, RequestSlots, [], "test")
    assert len(fake.calls) == 1


def test_structured_calls_retry_once_then_fail_clearly():
    fake = FakeLLM({"RequestSlots": [ValueError("bad json"), {"topic": "SQL"}]})
    assert structured(fake, RequestSlots, [], "test").topic == "SQL"

    failing = FakeLLM({"RequestSlots": [ValueError("bad"), ValueError("bad again")]})
    with pytest.raises(CurriculumModelError, match="test step"):
        structured(failing, RequestSlots, [], "test")
    assert len(failing.calls) == 2


def test_the_model_must_write_every_needed_section():
    """
    Regression: with defaults, the JSON schema sent to Ollama made these
    fields optional and qwen2.5:3b left out the learning outcomes,
    objectives and capstone (§17). They are required, with minimum sizes.
    """
    from agents.curriculum_research.schemas import CurriculumExtras, CurriculumOutline, ModuleDetailList

    outline = CurriculumOutline.model_json_schema()
    assert {"objectives", "learning_outcomes", "modules", "prerequisites"} <= set(outline["required"])
    assert outline["properties"]["learning_outcomes"]["minItems"] == 3
    extras = CurriculumExtras.model_json_schema()
    assert {"assessments", "capstone", "tools_resources", "competencies"} <= set(extras["required"])
    detail = ModuleDetailList.model_json_schema()["$defs"]["ModuleDetail"]
    assert {"activities", "outcome_ids"} <= set(detail["required"])
    # Bounded, so a run-on answer is cut while decoding instead of breaking the JSON.
    assessment = extras["$defs"]["AssessmentItem"]["properties"]
    assert assessment["description"]["maxLength"] == 700 and extras["properties"]["assessments"]["maxItems"] == 6
    assert outline["properties"]["modules"]["maxItems"] == 12


def test_the_default_factories_are_patched_in_tests():
    assert curriculum_agent.default_llm().__class__ is NoModel


# --------------------------------------------------------------------
# Kept from the old agent's reviewed behaviours (Phase 5 cleanup)
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Create a beginner curriculum about export management, 20 hours",
                                  "Create a beginner curriculum about translation, 20 hours",
                                  "أنشئ منهج عن الترجمة لمدة 20 ساعة للمبتدئين"])
def test_topics_that_sound_like_actions_still_create_a_curriculum(tools, conversations, text):
    language = "ar" if "منهج" in text else "en"
    agent, fake = make_agent(tools, conversations, canned_answers(language, hours=(5, 5, 10)))

    response = ask(agent, text)

    assert response["result"]["kind"] == "curriculum" and fake.steps()[0] == "CurriculumOutline"


def test_an_edit_that_mentions_creating_is_still_an_edit(tools, drafted):
    new_module = {"title": "Designing a course", "topics": ["Course design"], "description": "How to design.",
                  "activities": [{"type": "exercise", "description": "Draft an outline"}], "outcome_ids": ["LO1"]}
    agent, _, curriculum_id = drafted({"NewModule": new_module})

    response = ask(agent, "add a module on how to design a course")

    assert response["result"]["curriculum_id"] == curriculum_id and response["result"]["version"] == 2


def test_make_a_curriculum_still_creates_a_new_one_even_with_a_draft(tools, drafted):
    agent, _, curriculum_id = drafted({"CurriculumOutline": [outline_answer(), outline_answer()],
                                       "ModuleDetailList": [details_answer()] * 2,
                                       "CurriculumExtras": [extras_answer()] * 2})

    response = ask(agent, "make a beginner curriculum about SQL, 40 hours")

    assert response["result"]["curriculum_id"] != curriculum_id and response["result"]["version"] == 1


def test_numbering_labels_written_by_the_model_are_removed(tools, conversations):
    """Regression (Phase 6 live check): qwen2.5:3b prefixed "module1:" to the title, audience and modules."""
    outline = outline_answer("ar")
    outline["title"] = "module1: " + outline["title"]
    outline["target_audience"] = "module1: " + outline["target_audience"]
    outline["modules"][1]["title"] = "الوحدة 2: " + outline["modules"][1]["title"]
    agent, _ = make_agent(tools, conversations, canned_answers("ar", CurriculumOutline=outline))

    response = ask(agent, "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين")

    content = stored(tools, response["result"]["curriculum_id"])["content"]
    assert content["title"] == "أساسيات الحوسبة السحابية" and content["target_audience"] == "موظفو تقنية المعلومات"
    assert content["modules"][1]["title"] == "الخوادم الافتراضية"
