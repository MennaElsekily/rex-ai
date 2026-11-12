"""
agents/curriculum_research/safe_repair.py: the deterministic repairs for
validation problems with one safe, explainable fix -- and the cases it
must leave alone so the curriculum honestly stays a draft. The validation
rules themselves are never changed; every repaired curriculum is
validated again by validation.py.
"""

from agents.curriculum_research.curriculum_agent import CurriculumAgent
from agents.curriculum_research.safe_repair import apply_safe_fixes
from agents.curriculum_research.validation import validate_curriculum
from tests.curriculum_fakes import FakeLLM, bilingual_answers, in_process_tools, outline_answer
from tests.curriculum_fixtures import valid_curriculum


def _codes(curriculum):
    return [e["message"] for e in validate_curriculum(curriculum, curriculum["total_hours"])["errors"]]


def test_an_outcome_no_module_develops_is_linked_to_the_module_sharing_its_topic():
    """The exact problem seen in the app: "LO8 is not developed by any module"."""
    curriculum = valid_curriculum("en")
    curriculum["learning_outcomes"].append({"id": "LO3", "statement": "Evaluate a classification model's accuracy"})
    curriculum["assessments"][0]["outcome_ids"].append("LO3")
    assert "LO3 is not developed by any module." in _codes(curriculum)

    notes = apply_safe_fixes(curriculum)

    assert "LO3" in curriculum["modules"][1]["outcome_ids"]   # "Models and evaluation" / "Classification"
    assert "LO3" not in curriculum["modules"][0]["outcome_ids"]
    assert notes == ["LO3 was linked to module M2 (shared topic words: classification, model)."]
    assert validate_curriculum(curriculum, curriculum["total_hours"])["valid"]


def test_an_outcome_sharing_no_topic_word_with_any_module_is_left_for_the_admin():
    curriculum = valid_curriculum("en")
    curriculum["learning_outcomes"].append({"id": "LO3", "statement": "Negotiate supplier contracts"})
    curriculum["assessments"][0]["outcome_ids"].append("LO3")

    notes = apply_safe_fixes(curriculum)

    assert notes == [] and all("LO3" not in m["outcome_ids"] for m in curriculum["modules"])
    assert "LO3 is not developed by any module." in _codes(curriculum)   # still reported, never hidden


def test_an_outcome_no_assessment_covers_is_linked_but_the_capstone_counts_as_assessing():
    curriculum = valid_curriculum("en")
    curriculum["assessments"][0]["outcome_ids"] = ["LO1"]   # LO2 is still demonstrated by the capstone
    assert apply_safe_fixes(curriculum) == []

    curriculum["capstone"]["outcome_ids"] = ["LO1"]
    notes = apply_safe_fixes(curriculum)
    assert curriculum["assessments"][0]["outcome_ids"] == ["LO1", "LO2"]
    assert notes and "LO2 was linked to assessment A1" in notes[0]


def test_a_module_that_develops_no_outcome_is_linked_to_its_closest_outcome():
    curriculum = valid_curriculum("en")
    curriculum["modules"][1]["outcome_ids"] = []
    curriculum["learning_outcomes"][1]["statement"] = "Evaluate a classification model"

    apply_safe_fixes(curriculum)

    assert curriculum["modules"][1]["outcome_ids"] == ["LO2"]
    assert validate_curriculum(curriculum, curriculum["total_hours"])["valid"]


def test_vague_english_verbs_get_the_standard_measurable_verb():
    curriculum = valid_curriculum("en")
    curriculum["learning_outcomes"][0]["statement"] = "Understand how a dataset is prepared"
    curriculum["learning_outcomes"][1]["statement"] = "Learners will be able to know model evaluation metrics"

    notes = apply_safe_fixes(curriculum)

    assert curriculum["learning_outcomes"][0]["statement"] == "Explain how a dataset is prepared"
    assert curriculum["learning_outcomes"][1]["statement"] == "Learners will be able to identify model evaluation metrics"
    assert len(notes) == 2 and validate_curriculum(curriculum, curriculum["total_hours"])["valid"]


def test_vague_arabic_verbs_get_the_standard_measurable_verb():
    curriculum = valid_curriculum("ar")
    curriculum["learning_outcomes"][0]["statement"] = "فهم أساسيات تجهيز البيانات"
    curriculum["learning_outcomes"][1]["statement"] = "يعرف مقاييس تقييم النماذج"

    notes = apply_safe_fixes(curriculum)

    assert curriculum["learning_outcomes"][0]["statement"] == "شرح أساسيات تجهيز البيانات"
    assert curriculum["learning_outcomes"][1]["statement"] == "يحدد مقاييس تقييم النماذج"
    assert len(notes) == 2 and all("استُبدل الفعل" in n for n in notes)


def test_an_outcome_with_no_verb_at_all_is_not_rewritten():
    curriculum = valid_curriculum("en")
    curriculum["learning_outcomes"][0]["statement"] = "The basics of data"

    assert apply_safe_fixes(curriculum) == []
    assert curriculum["learning_outcomes"][0]["statement"] == "The basics of data"


def test_a_valid_curriculum_is_never_changed():
    curriculum = valid_curriculum("en")
    before = repr(curriculum)

    assert apply_safe_fixes(curriculum) == []
    assert repr(curriculum) == before


def test_end_to_end_an_unlinked_outcome_the_model_left_becomes_pending_review(tmp_path):
    """Generate -> validate -> one model repair (which fixes nothing) -> safe fixes -> validate -> pending review."""
    tools = in_process_tools(tmp_path)
    answers = bilingual_answers("en")
    outline = outline_answer()
    outline["learning_outcomes"].append({"id": "LO4", "statement": "Apply identity and access controls in the cloud"})
    answers["CurriculumOutline"] = outline
    answers["RepairPatch"] = {"learning_outcomes": []}
    agent = CurriculumAgent(tools=tools, llm=FakeLLM(answers))

    response = agent.process_request({"request_id": "r1", "context": {}, "user_input":
                                      "Create a beginner 40-hour curriculum on cloud computing for IT support staff"})

    result = response["result"]
    content = tools.store.get(result["curriculum_id"])["content"]
    security = next(m for m in content["modules"] if m["title"] == "Cloud security")
    assert "LO4" in security["outcome_ids"]
    assert result["review_status"] == "pending_review", result["validation"]
    assert "LO4 was linked to module" in result["message"]
    agent.close()


def test_sharing_only_the_outcomes_own_leading_verb_is_no_evidence():
    """Seen on a real Arabic draft: the only "shared word" was the action verb يطبق itself."""
    curriculum = valid_curriculum("ar")
    curriculum["learning_outcomes"].append({"id": "LO3", "statement": "يطبق مبادئ التفاوض مع الموردين"})
    curriculum["modules"][0]["description"] = "يطبق المتدرب تقنيات التنظيف على مجموعات البيانات."

    notes = apply_safe_fixes(curriculum)

    assert not any("LO3" in n for n in notes)
    assert all("LO3" not in m["outcome_ids"] for m in curriculum["modules"])


def test_the_arabic_verb_create_gets_its_accepted_form():
    """Also seen live: "يخلق" (creates) is not in the measurable list; "ينشئ" is."""
    curriculum = valid_curriculum("ar")
    curriculum["learning_outcomes"][0]["statement"] = "يخلق محتوى تسويقيًا بسيطًا"

    apply_safe_fixes(curriculum)

    assert curriculum["learning_outcomes"][0]["statement"] == "ينشئ محتوى تسويقيًا بسيطًا"
