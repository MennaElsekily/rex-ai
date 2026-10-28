"""
Framework alignment (agents/curriculum_research/alignment.py and the
graph's align routes): only approved framework files, one result per
requirement, evidence required for covered/partial, gaps with
recommendations, saved as a new version with a PDF, cleared by revisions,
never presented as accreditation. Stand-in model, temporary storage.
"""

import json

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research.alignment import match_framework
from agents.curriculum_research.curriculum_agent import CurriculumAgent
from tests.curriculum_fakes import FakeLLM, canned_answers, in_process_tools
from tests.curriculum_fixtures import FRAMEWORK

CONTEXT = {"conversation_id": "admin-align"}


def judgement(*items):
    return {"items": [dict(zip(("requirement_id", "status", "element_ids", "explanation", "recommendation"), item))
                      for item in items]}


GOOD = judgement(("R1", "covered", ["LO1", "M1"], "M1 teaches it and A1 assesses it.", None),
                 ("R2", "covered", [], "Probably covered.", None),           # no evidence -> downgraded
                 ("R9", "covered", ["LO1"], "Not a requirement.", None),     # unknown requirement -> ignored
                 ("R2", "not_covered", [], "Second answer for R2.", None))   # duplicate -> first one kept
# R3 is not judged at all -> unable to determine.


@pytest.fixture
def frameworks_dir(tmp_path):
    directory = tmp_path / "frameworks"
    directory.mkdir()
    (directory / "test-framework.json").write_text(json.dumps(FRAMEWORK), encoding="utf-8")
    return directory


@pytest.fixture
def tools(tmp_path, frameworks_dir):
    return in_process_tools(tmp_path, frameworks_dir=frameworks_dir)


def make_agent(tools, tmp_path, answers):
    fake = FakeLLM(answers)
    return CurriculumAgent(conversation_store=ConversationStore(tmp_path / "c.sqlite3"), tools=tools, llm=fake), fake


def ask(agent, text):
    return agent.process_request({"request_id": "r", "user_input": text, "context": CONTEXT})


def draft(agent):
    return ask(agent, "Create a beginner curriculum about cloud computing, 40 hours")["result"]["curriculum_id"]


def test_alignment_is_evidence_based_and_saved_as_a_new_version(tools, tmp_path):
    agent, fake = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement=GOOD))
    curriculum_id = draft(agent)

    response = ask(agent, "Align it with the test framework")

    result = response["result"]
    assert (result["curriculum_id"], result["version"]) == (curriculum_id, 2)
    record = tools.store.get(curriculum_id)
    alignment = record["content"]["alignment"]
    by_id = {i["requirement_id"]: i for i in alignment["items"]}
    assert list(by_id) == ["R1", "R2", "R3"]  # every requirement, nothing invented
    assert by_id["R1"]["status"] == "covered" and by_id["R1"]["element_ids"] == ["LO1", "M1"]
    assert by_id["R2"]["status"] == "unable_to_determine" and "without valid evidence" in by_id["R2"]["explanation"]
    assert by_id["R3"]["status"] == "unable_to_determine"
    assert alignment["summary"] == {"covered": 1, "partially_covered": 0, "not_covered": 0, "unable_to_determine": 2}
    assert alignment["framework"]["source_url"] == FRAMEWORK["source_url"]
    assert record["framework_id"] == "test-framework" and record["alignment"] == alignment
    # The content itself is unchanged; the validator checked the alignment against the framework.
    assert tools.store.get(curriculum_id, 1)["content"]["modules"] == record["content"]["modules"]
    assert result["validation"]["valid"] and result["review_status"] == "pending_review"
    # The model saw the requirements and the word-overlap hints.
    prompt = fake.prompts("AlignmentJudgement")[0]
    assert "R1: Prepare and clean a dataset for analysis" in prompt and "possible evidence" in prompt
    assert "Alignment with Test Competency Framework (1.0)" in result["message"]
    assert "not accreditation" in result["message"] and result["document"]["pages"] >= 2


def test_gaps_carry_recommendations(tools, tmp_path):
    answer = judgement(("R1", "covered", ["LO1"], "Taught.", "ignored for covered"),
                       ("R2", "partially_covered", ["LO2"], "Only evaluated briefly.", "Add an evaluation lab."),
                       ("R3", "not_covered", [], "No deployment content.", "Add a deployment module."))
    agent, _ = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement=answer))
    curriculum_id = draft(agent)

    response = ask(agent, "Compare the curriculum with the test framework")

    alignment = tools.store.get(curriculum_id)["content"]["alignment"]
    assert alignment["gaps"] == ["R2", "R3"]
    assert alignment["items"][0]["recommendation"] is None
    assert "R3 (not covered): Add a deployment module." in response["result"]["message"]


def test_a_compliance_claim_in_the_alignment_keeps_it_a_draft(tools, tmp_path):
    answer = judgement(("R1", "covered", ["LO1"], "The course is fully compliant with the framework.", None))
    agent, _ = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement=answer))
    draft(agent)

    response = ask(agent, "Align it with the test framework")

    assert response["result"]["review_status"] == "draft"
    assert "accreditation_claim" in {e["code"] for e in response["result"]["validation"]["errors"]}


def test_generation_that_names_a_framework_is_aligned_before_saving(tools, tmp_path):
    agent, fake = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement=GOOD))

    response = ask(agent, "Create a beginner curriculum about cloud computing, 40 hours, aligned with the test framework")

    record = tools.store.get(response["result"]["curriculum_id"])
    assert record["version"] == 1 and record["content"]["alignment"]["framework"]["framework_id"] == "test-framework"
    assert fake.steps()[-1] == "AlignmentJudgement"


def test_a_plain_topic_word_does_not_trigger_alignment(tmp_path):
    directory = tmp_path / "fw"
    directory.mkdir()
    nice = dict(FRAMEWORK, framework_id="nice-cybersecurity", name="NICE Workforce Framework for Cybersecurity")
    (directory / "nice-cybersecurity.json").write_text(json.dumps(nice), encoding="utf-8")
    tools = in_process_tools(tmp_path, frameworks_dir=directory)
    agent, fake = make_agent(tools, tmp_path, canned_answers())

    ask(agent, "Create a beginner cybersecurity curriculum, 40 hours")

    assert "AlignmentJudgement" not in fake.steps()


def test_unknown_or_unclear_frameworks_and_missing_curricula(tools, tmp_path, frameworks_dir):
    agent, fake = make_agent(tools, tmp_path, canned_answers())
    assert "Which curriculum?" in ask(agent, "Align it with the test framework")["result"]["message"]

    other = dict(FRAMEWORK, framework_id="other-body", name="Other Body Skills Model")
    (frameworks_dir / "other-body.json").write_text(json.dumps(other), encoding="utf-8")
    draft(agent)
    reply = ask(agent, "Check this curriculum against the framework")["result"]["message"]
    assert "Which approved framework" in reply and "test-framework" in reply and "other-body" in reply
    assert "AlignmentJudgement" not in fake.steps()

    generated = ask(agent, "Create a beginner curriculum about SQL, 30 hours, aligned with the ESCO framework")
    assert "No approved framework document matches" in generated["result"]["message"]
    assert generated["result"]["curriculum"].get("alignment") is None


def test_a_revision_clears_the_alignment(tools, tmp_path):
    agent, _ = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement=GOOD))
    curriculum_id = draft(agent)
    ask(agent, "Align it with the test framework")

    response = ask(agent, "make it 30 hours")

    record = tools.store.get(curriculum_id)
    assert record["version"] == 3 and record["content"]["alignment"] is None
    assert "ask to align it again" in response["result"]["message"]


def test_arabic_alignment_request(tools, tmp_path):
    answers = canned_answers("ar", AlignmentJudgement=judgement(("R1", "covered", ["LO1"], "مغطى في الوحدة الأولى.", None)))
    agent, fake = make_agent(tools, tmp_path, answers)
    agent.process_request({"request_id": "r", "user_input": "أنشئ منهج عن الحوسبة السحابية لمدة 40 ساعة للمبتدئين",
                           "context": CONTEXT})

    response = ask(agent, "قارن المنهج مع معايير test-framework")

    assert "أضيفت المواءمة" in response["result"]["message"]
    assert "Write explanation and recommendation in Arabic" in fake.prompts("AlignmentJudgement")[0]


def test_large_frameworks_are_judged_in_batches(tools, tmp_path, frameworks_dir):
    big = dict(FRAMEWORK, requirements=[{"id": f"K{n:03d}", "text": f"Knowledge of topic {n}"} for n in range(30)])
    (frameworks_dir / "test-framework.json").write_text(json.dumps(big), encoding="utf-8")
    agent, fake = make_agent(tools, tmp_path, canned_answers(AlignmentJudgement={"items": []}))
    curriculum_id = draft(agent)

    ask(agent, "Align it with the test framework")

    assert fake.steps().count("AlignmentJudgement") == 3  # 12 + 12 + 6
    items = tools.store.get(curriculum_id)["content"]["alignment"]["items"]
    assert len(items) == 30 and {i["status"] for i in items} == {"unable_to_determine"}


@pytest.mark.parametrize("text, expected", [
    ("align with test-framework", "test-framework"),
    ("compare it with the Test Competency Framework", "test-framework"),
    ("check it against the NICE framework", "nice-cybersecurity"),
    ("align it with the framework", None),
    ("align it with the national skills framework", None),
])
def test_match_framework(text, expected):
    frameworks = [{"framework_id": "test-framework", "name": "Test Competency Framework"},
                  {"framework_id": "nice-cybersecurity", "name": "NICE Workforce Framework for Cybersecurity"}]
    assert match_framework(text, frameworks) == expected
