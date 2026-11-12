"""
Stand-ins and helpers for Agent 4's tests: the real toolbox on temporary
storage, canned model answers for the narrative parts, and a helper that
saves and approves a curriculum the way the Curriculum Agent and an
administrator do (never writing the approved status directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation.document_store import DocumentStore
from agents.document_generation.document_tools import DocumentToolbox
from tests.curriculum_fakes import FakeLLM
from tests.curriculum_fixtures import valid_curriculum

ADMIN = "admin-session-test0001"


def in_process_document_tools(root: Path) -> DocumentToolbox:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return DocumentToolbox(store=DocumentStore(root / "documents.sqlite3"),
                           curricula=CurriculumStore(root / "curricula.sqlite3"), documents_dir=root / "pdfs")


def save_curriculum(toolbox: DocumentToolbox, content: dict | None = None, valid: bool = True,
                    curriculum_id: str | None = None) -> dict:
    """Save as the Curriculum Agent does: pending_review (valid) or draft."""
    return toolbox.curricula.save_version(content or valid_curriculum(), {"valid": valid, "errors": []}, [],
                                          curriculum_id=curriculum_id)


def approved_curriculum(toolbox: DocumentToolbox, content: dict | None = None) -> str:
    saved = save_curriculum(toolbox, content)
    toolbox.curricula.approve(saved["curriculum_id"], saved["version"], ADMIN)
    return saved["curriculum_id"]


_GUIDE_TOPIC = {"en": {"M1": "cleaning data", "M2": "classification"}, "ar": {"M1": "تنظيف البيانات", "M2": "التصنيف"}}


def guide_teaching(language: str, module_id: str) -> dict:
    """A participant-guide explanation and example that teach the module's topic (long enough to pass the checks)."""
    topic = _GUIDE_TOPIC[language].get(module_id) or _GUIDE_TOPIC[language]["M1"]
    if language == "ar":
        return {"explanation": f"يتناول هذا الجزء موضوع {topic} ويشرح معناه خطوة بعد خطوة. يبدأ المتعلم بفهم الفكرة الأساسية "
                               f"في {topic} ثم يتعرف على سبب أهميتها في العمل مع البيانات. بعد ذلك يرى كيف تُطبق هذه الفكرة "
                               "على أمثلة بسيطة، ويتدرب على ملاحظة النتائج وتفسيرها بلغة واضحة قبل الانتقال إلى الجزء التالي.",
                "example": f"مثال عملي: يطبق المتعلم {topic} على مجموعة صغيرة من البيانات ويقارن النتيجة قبل التطبيق وبعده."}
    return {"explanation": f"This part explains {topic} step by step. Learners first meet the core idea behind {topic} and why it "
                           "matters when working with data. They then see how the idea is applied to a small, realistic example, "
                           "practise noticing the results, and learn to describe what happened in plain words before they move on "
                           "to the next part of the course.",
            "example": f"Worked example: a learner applies {topic} to a small dataset and compares the result before and after."}


def parts_answers(language: str = "en", module_ids=("M1", "M2")) -> dict[str, Any]:
    ar = language == "ar"
    return {
        "DescriptionParts": {"summary": "ملخص الدورة" if ar else "A practical introduction to machine learning.",
                             "assessment_approach": "اختبار عملي" if ar else "Learners complete one practical test."},
        "ProposalParts": {"executive_summary": "ملخص تنفيذي" if ar else "A practical machine learning course.",
                          "background_need": "حاجة" if ar else "Employers expect data and modelling skills.",
                          "methodology": "منهجية" if ar else "Hands-on exercises and labs.",
                          "expected_deliverables": ["دفتر ملاحظات"] if ar else ["A trained and evaluated model"]},
        "GuideParts": {"introduction": "مرحبًا" if ar else "Welcome to the course.",
                       "assessment_instructions": "تعليمات" if ar else "Complete the practical test on your own.",
                       "modules": [{"module_id": m, **guide_teaching(language, m),
                                    "key_concepts": ["مفهوم 1", "مفهوم 2"] if ar else ["Concept one", "Concept two"],
                                    "exercise": "تمرين" if ar else "Work through the lab."} for m in module_ids]},
        "CapstoneParts": {"introduction": "مقدمة المشروع" if ar else "Build an end-to-end model.",
                          "evaluation_guidance": "إرشادات" if ar else "Prepare the notebook and presentation."},
    }


def fake_llm(language: str = "en", **overrides: Any) -> FakeLLM:
    answers = {**parts_answers(language), **overrides}
    return FakeLLM({name: [answer] * 6 if not isinstance(answer, list) else answer for name, answer in answers.items()})


# --------------------------------------------------------------------
# Phase 2: documents written one unit at a time (instructor guide, presentation outline, question bank, rubrics).
# The fake model reads the module / assessment id from the prompt, so every unit gets valid, grounded content.
# --------------------------------------------------------------------

import re

NEW_TYPES = ("instructor_guide", "presentation_outline", "question_bank", "assessment_rubric")

# What each module of tests.curriculum_fixtures.valid_curriculum is about (words that appear in the curriculum).
_TOPICS = {
    "en": {"M1": ("cleaning data", "dataset"), "M2": ("classification", "classifier")},
    "ar": {"M1": ("تنظيف البيانات", "البيانات"), "M2": ("تصنيف", "نموذج")},
}


def _unit_id(prompt: str, label: str) -> str:
    match = re.search(label + r": (\S+)", prompt)
    assert match, f"no '{label}' in the prompt"
    return match.group(1)


def teaching_answer(language="en"):
    ar = language == "ar"

    def answer(prompt):
        mid = _unit_id(prompt, "Module id")
        topic = _TOPICS[language][mid][0]
        return {"module_id": mid,
                "teaching_methods": [f"شرح مباشر لـ {topic}", "تطبيق عملي موجّه"] if ar else [f"Short explanation of {topic}", "Guided practice"],
                "instructor_activities": ["يعرض مثالًا" if ar else "Demonstrate a worked example"],
                "instructor_notes": f"ركّز على {topic}." if ar else f"Emphasise {topic} and check understanding often.",
                "discussion_questions": [f"لماذا يهم {topic}؟", f"أين تطبق {topic}؟"] if ar else
                [f"Why does {topic} matter?", f"Where would you apply {topic}?"]}

    return answer


def slides_answer(language="en"):
    ar = language == "ar"

    def answer(prompt):
        mid = _unit_id(prompt, "Module id")
        topic = _TOPICS[language][mid][0]
        return {"module_id": mid,
                "key_points": [f"نقطة {i} عن {topic}" for i in (1, 2, 3)] if ar else [f"Point {i} about {topic}" for i in (1, 2, 3)],
                "example": f"مثال على {topic}" if ar else f"An example of {topic}",
                "discussion_prompt": f"ناقش {topic}" if ar else f"Discuss {topic}",
                "visual": "مخطط توضيحي" if ar else "A simple diagram"}

    return answer


def questions_answer(language="en"):
    ar = language == "ar"
    bank = {
        "en": {"M1": ["Which step cleans a dataset before modelling?", "What does cleaning data prepare a dataset for?",
                      "Cleaning data is part of preparing a dataset.", "Explain why a dataset is cleaned before modelling."],
               "M2": ["Which task does a classification model perform?", "How is a classifier evaluated after training?",
                      "A classifier is trained and then evaluated.", "Describe how you would evaluate a classification model."]},
        "ar": {"M1": ["ما الخطوة التي تنظيف البيانات قبل بناء النموذج؟", "لأي غرض يُجرى تنظيف البيانات؟",
                      "تنظيف البيانات جزء من تجهيز البيانات.", "اشرح لماذا يجري تنظيف البيانات قبل النمذجة."],
               "M2": ["ما المهمة التي ينفذها نموذج التصنيف؟", "كيف يُقيَّم نموذج التصنيف بعد التدريب؟",
                      "يُدرَّب نموذج التصنيف ثم يُقيَّم.", "صف كيف تقيّم نموذج التصنيف."]},
    }

    def options(mid, which):
        if ar:
            return [f"الخيار الصحيح {mid}{which}", f"خيار آخر {mid}{which}أ", f"خيار ثالث {mid}{which}ب", f"خيار رابع {mid}{which}ج"]
        return [f"The correct option {mid}{which}", f"Another option {mid}{which}a", f"A third option {mid}{which}b",
                f"A fourth option {mid}{which}c"]

    def answer(prompt):
        mid = _unit_id(prompt, "Module id")
        q = bank[language][mid]
        why = "لأنه الصحيح" if ar else "Because it is the right one."
        return {"module_id": mid,
                "multiple_choice": [{"question": q[0], "options": options(mid, "1"), "correct_index": 0, "rationale": why},
                                    {"question": q[1], "options": options(mid, "2"), "correct_index": 1 if False else 0, "rationale": why}],
                "true_false": [{"statement": q[2], "answer": True, "rationale": why}],
                "short_answer": [{"question": q[3], "model_answer": "إجابة نموذجية" if ar else "A model answer.", "rationale": why}]}

    return answer


def rubric_answer(language="en"):
    ar = language == "ar"

    def answer(prompt):
        rid = _unit_id(prompt, "Assessment id")
        outcomes = re.findall(r"^- (LO\d+):", prompt, re.MULTILINE)
        levels = ("بداية", "نامٍ", "متمكن", "متقدم") if ar else ("barely shows", "partly shows", "clearly shows", "expertly shows")
        return {"assessment_id": rid, "instructions": "قيّم العمل وفق البنود" if ar else "Score the work against each criterion.",
                "criteria": [{"outcome_id": oid,
                              **{name: (f"عمل {lv} للمخرج {oid} رقم {i}" if ar else f"The work {lv} outcome {oid} (level {i}).")
                                 for i, (name, lv) in enumerate(zip(("beginning", "developing", "proficient", "advanced"), levels), 1)}}
                             for oid in dict.fromkeys(outcomes)]}

    return answer


def new_type_answers(language="en", repeat=20):
    """FakeLLM answers for every unit and frame schema of the four new types."""
    frames = {
        "InstructorFrame": {"preparation": "تحضير المدرب" if language == "ar" else "Prepare the datasets and the lab.",
                            "assessment_guidance": "إرشادات التقييم" if language == "ar" else "Run the practical test and mark it."},
        "SlidesFrame": {"closing_summary": "ملخص الدورة" if language == "ar" else "A short summary of the course."},
    }
    units_ = {"ModuleTeaching": teaching_answer(language), "ModuleSlides": slides_answer(language),
              "ModuleQuestions": questions_answer(language), "AssessmentRubric": rubric_answer(language)}
    return {name: [value] * repeat for name, value in {**frames, **units_}.items()}


def container_answer(doc_type: str, language: str, curriculum: dict) -> dict:
    """The assembled model text of a new-type document (what generation produces), for composing without a model."""
    from agents.document_generation import templates, units as unit_helpers

    template = templates.TEMPLATES[doc_type]
    maker = {"instructor_guide": teaching_answer, "presentation_outline": slides_answer,
             "question_bank": questions_answer, "assessment_rubric": rubric_answer}[doc_type](language)
    out = {}
    if template.frame_schema is not None:
        out.update(new_type_answers(language)[template.frame_schema.__name__][0])
    key = "Assessment id" if template.unit_field == "rubrics" else "Module id"
    items = []
    for uid in unit_helpers.unit_ids(template.unit_field, curriculum):
        prompt = f"{key}: {uid}\n" + "".join(f"- {c['outcome_id']}: {c['statement']}\n" for c in (unit_helpers.rubric_spec(curriculum, uid) or {}).get("criteria", []))
        items.append(maker(prompt))
    out[template.unit_field] = items
    return out
