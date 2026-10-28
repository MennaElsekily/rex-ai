"""
Stand-ins for Agent 3's language model and tools in tests.

FakeLLM mimics the part of a LangChain chat model the graph uses:
``llm.with_structured_output(Schema).invoke(messages)``. Answers are
given per schema name as a value, a list (one per call, in order) or a
function of the prompt text; every call is recorded. ``canned_answers``
gives answers that build a valid curriculum in English or Arabic.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from agents.curriculum_research.academy_courses import AcademyCatalogue
from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.curriculum_research.curriculum_tools import CurriculumToolbox
from agents.curriculum_research.schemas import (
    CapstoneItem,
    CurriculumExtras,
    CurriculumOutline,
    ModuleDetailList,
    NewModule,
    RepairPatch,
    RequestSlots,
    SuggestionList,
)
from tests.curriculum_fixtures import build_test_index


class NoModel:
    """The default model in tests: any call fails loudly (no real Ollama)."""

    def with_structured_output(self, schema):
        raise AssertionError("A test reached the language model without a FakeLLM.")


class FakeLLM:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = {k: (list(v) if isinstance(v, list) else v) for k, v in answers.items()}
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema):
        fake = self

        class Runnable:
            def invoke(self, messages):
                prompt = "\n".join(getattr(m, "content", str(m)) for m in messages)
                fake.calls.append((schema.__name__, prompt))
                answer = fake.answers.get(schema.__name__)
                if isinstance(answer, list):
                    if not answer:
                        raise AssertionError(f"No more fake answers for {schema.__name__}.")
                    answer = answer.pop(0)
                if callable(answer) and not isinstance(answer, type):
                    answer = answer(prompt)
                if isinstance(answer, Exception):
                    raise answer
                if answer is None:
                    raise AssertionError(f"No fake answer for {schema.__name__}.")
                return schema.model_validate(copy.deepcopy(answer)) if isinstance(answer, dict) else answer

        return Runnable()

    def steps(self) -> list[str]:
        return [name for name, _ in self.calls]

    def prompts(self, schema_name: str) -> list[str]:
        return [prompt for name, prompt in self.calls if name == schema_name]


def _texts(language: str) -> dict[str, Any]:
    if language == "ar":
        return {
            "title": "أساسيات الحوسبة السحابية", "overview": "دورة عملية تعرّف المتدربين بخدمات الحوسبة السحابية.",
            "rationale": "تحتاج المؤسسات إلى موظفين يديرون الخدمات السحابية بأمان.",
            "objectives": ["تقديم نماذج الخدمات السحابية", "تدريب المتعلمين على نشر الخدمات"],
            "audience": "موظفو تقنية المعلومات", "outcomes": ["يقارن نماذج الخدمات السحابية",
                                                           "ينشر خادمًا افتراضيًا", "يطبق ضوابط الأمان الأساسية"],
            "modules": [("مقدمة في السحابة", ["نماذج الخدمات"]), ("الخوادم الافتراضية", ["إنشاء خادم"]),
                        ("الأمان السحابي", ["إدارة الهوية"])],
            "description": "شرح عملي للموضوعات مع تطبيقات.", "activity": ("مختبر", "تمرين عملي موجّه"),
            "assessment": ("اختبار عملي", "تنفيذ مهمة سحابية كاملة"),
            "capstone": ("مشروع ختامي", "نشر تطبيق بسيط على السحابة", ["تقرير", "عرض تقديمي"]),
            "tools": ["AWS Free Tier", "متصفح ويب"], "competency": "إدارة الخدمات السحابية",
            "assumption": "يتوفر للمتدربين حساب سحابي مجاني.", "question": "هل تفضّل الأكاديمية مزودًا سحابيًا معينًا؟",
        }
    return {
        "title": "Cloud Computing Essentials", "overview": "A practical course introducing cloud services.",
        "rationale": "Organisations need staff who can run cloud services safely.",
        "objectives": ["Introduce cloud service models", "Practise deploying services"],
        "audience": "IT support staff", "outcomes": ["Compare cloud service models", "Deploy a virtual server",
                                                      "Apply basic cloud security controls"],
        "modules": [("Cloud foundations", ["Service models"]), ("Virtual servers", ["Launching a server"]),
                    ("Cloud security", ["Identity and access"])],
        "description": "Guided explanation of the topics with practice.", "activity": ("lab", "Guided hands-on lab"),
        "assessment": ("practical test", "Complete a cloud task end to end"),
        "capstone": ("Capstone project", "Deploy a simple application to the cloud", ["Report", "Presentation"]),
        "tools": ["AWS Free Tier", "Web browser"], "competency": "Cloud service administration",
        "assumption": "Learners have a free cloud account.", "question": "Does the academy prefer one cloud provider?",
    }


def outline_answer(language: str = "en", hours: tuple[int, ...] = (10, 15, 15), level: str = "Beginner") -> dict:
    t = _texts(language)
    return {"title": t["title"], "overview": t["overview"], "rationale": t["rationale"], "objectives": t["objectives"],
            "target_audience": t["audience"], "prerequisites": [], "level": level,
            "learning_outcomes": [{"id": f"LO{i}", "statement": s} for i, s in enumerate(t["outcomes"], start=1)],
            "modules": [{"id": f"M{i}", "title": title, "hours": h, "topics": topics}
                        for i, ((title, topics), h) in enumerate(zip(t["modules"], hours), start=1)]}


def details_answer(language: str = "en", modules: int = 3) -> dict:
    t = _texts(language)
    return {"modules": [{"module_id": f"M{i}", "description": t["description"],
                         "activities": [{"type": t["activity"][0], "description": t["activity"][1]}],
                         "outcome_ids": [f"LO{min(i, 3)}"]} for i in range(1, modules + 1)]}


def extras_answer(language: str = "en", source_ids: tuple[str, ...] = ()) -> dict:
    t = _texts(language)
    return {"assessments": [{"id": "A1", "type": t["assessment"][0], "description": t["assessment"][1],
                             "outcome_ids": ["LO1", "LO2", "LO3"]}],
            "capstone": {"title": t["capstone"][0], "description": t["capstone"][1],
                         "outcome_ids": ["LO2", "LO3"], "deliverables": t["capstone"][2]},
            "tools_resources": t["tools"], "competencies": [{"name": t["competency"], "source_ids": list(source_ids)}],
            "assumptions": [t["assumption"]], "open_questions": [t["question"]]}


def canned_answers(language: str = "en", hours: tuple[int, ...] = (10, 15, 15), **extra: Any) -> dict[str, Any]:
    """Answers for one valid generation (outline, module details, extras)."""
    answers = {"CurriculumOutline": outline_answer(language, hours),
               "ModuleDetailList": details_answer(language, len(hours)),
               "CurriculumExtras": extras_answer(language)}
    answers.update(extra)
    return answers


def in_process_tools(tmp_path, research_domains=None, search: Callable | None = None,
                     frameworks_dir=None) -> CurriculumToolbox:
    """The real toolbox on a temporary index, database and folders (no real index, no network)."""
    def fake_evidence(question):
        return {"status": "ok", "chunks": []}

    catalogue = AcademyCatalogue(vector_store=build_test_index(tmp_path / "index"), evidence_search=fake_evidence,
                                 translate=lambda text: text)
    frameworks = frameworks_dir or tmp_path / "frameworks"
    frameworks.mkdir(parents=True, exist_ok=True)
    return CurriculumToolbox(store=CurriculumStore(tmp_path / "curricula.sqlite3"), catalogue=catalogue,
                             frameworks_dir=frameworks, documents_dir=tmp_path / "docs",
                             research_domains=research_domains if research_domains is not None else [],
                             search=search)


__all__ = ["CapstoneItem", "CurriculumExtras", "CurriculumOutline", "FakeLLM", "ModuleDetailList", "NewModule",
           "NoModel", "RepairPatch", "RequestSlots", "SuggestionList", "canned_answers", "details_answer",
           "extras_answer", "in_process_tools", "outline_answer"]
