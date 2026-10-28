"""
Shared test data for Agent 3: a small temporary knowledge-base index in
the same shape the ingestion pipeline writes (chunk text + field_*
metadata), built with fake embeddings, plus helpers for curricula.

The courses are invented for tests only; the real catalogue is never
read or written by the normal test suite.
"""

from __future__ import annotations

import copy

from knowledge_base.vector_store import VectorStore

COURSES = [
    {"course_id": "DA-101", "name": "Spreadsheet Analysis Basics", "department": "Data Analytics",
     "level": "Beginner", "duration": "4 weeks (24 hours)", "audience": "Office staff new to data.",
     "outcomes": ["Clean data in spreadsheets", "Build pivot-table summaries"],
     "topics": ["Spreadsheet formulas", "Pivot tables", "Charts"], "skills": ["Excel"], "prereq": ""},
    {"course_id": "DA-201", "name": "SQL for Analysts", "department": "Data Analytics",
     "level": "Intermediate", "duration": "6 weeks (40 hours)", "audience": "Analysts who know spreadsheets.",
     "outcomes": ["Write SQL queries with joins", "Build dashboards from queries"],
     "topics": ["SELECT and joins", "Aggregations", "Dashboards"], "skills": ["SQL", "Power BI"],
     "prereq": "DA-101 Spreadsheet Analysis Basics"},
    {"course_id": "CL-101", "name": "Cloud Computing Foundations", "department": "Cloud",
     "level": "Beginner", "duration": "5 weeks (30 hours)", "audience": "IT staff moving to the cloud.",
     "outcomes": ["Deploy a virtual machine", "Compare cloud service models"],
     "topics": ["IaaS PaaS SaaS", "Virtual machines", "Storage"], "skills": ["AWS console"], "prereq": ""},
]
DATA_STATUS = "Synthetic demonstration data"


def _fake_embedding(text: str) -> list[float]:
    vector = [0.0] * 8
    for index, char in enumerate(text.casefold()):
        vector[index % 8] += (ord(char) % 31) / 31.0
    total = sum(vector) or 1.0
    return [v / total for v in vector]


def build_test_index(directory) -> VectorStore:
    """A temporary index with the test courses and one FAQ chunk (no course id)."""
    store = VectorStore(persist_directory=directory)
    chunks = []
    for number, course in enumerate(COURSES):
        fields = {"course_id": course["course_id"], "department": course["department"], "level": course["level"],
                  "duration": course["duration"], "target_audience": course["audience"],
                  "data_status": DATA_STATUS, "updated": "2026-10-27"}
        overview = (f"{course['name']}\nCourse ID: {course['course_id']}\nDepartment: {course['department']}\n"
                    f"Level: {course['level']}\nDuration: {course['duration']}\n")
        details = (f"{course['name']}\nEntry Requirements and Prerequisites\n"
                   f"Prerequisite courses: {course['prereq'] or 'None'}\n\nLearning Outcomes\n"
                   + "\n".join(f"- {o}" for o in course["outcomes"]) + "\n\nTopics Covered\n"
                   + "\n".join(f"- {t}" for t in course["topics"]) + "\n\nSkills Taught\n"
                   + "\n".join(f"- {s}" for s in course["skills"]))
        for part, text in enumerate((overview, details)):
            chunks.append({"text": text, "chunk_id": number * 10 + part, "source": f"catalogue_{course['course_id']}.md",
                           "filename": f"catalogue_{course['course_id']}.md", "section_path": course["name"],
                           "extracted_metadata": {"explicit_fields": fields}})
    chunks.append({"text": "FAQ\nQuestion: Is there a refund?\nAnswer: Yes.", "chunk_id": 99, "source": "faqs.json",
                   "filename": "faqs.json", "section_path": "FAQ", "extracted_metadata": {"explicit_fields": {"id": "FAQ-01"}}})
    store.add_chunks(chunks, [_fake_embedding(c["text"]) for c in chunks])
    return store


def valid_curriculum(language: str = "en", hours: int = 40) -> dict:
    """A curriculum that passes every validation rule."""
    if language == "ar":
        curriculum = {
            "title": "أساسيات تعلم الآلة", "language": "ar",
            "overview": "دورة عملية تقدم أساسيات تعلم الآلة للمبتدئين باستخدام بايثون.",
            "rationale": "يحتاج سوق العمل إلى مهارات تحليل البيانات وبناء النماذج.",
            "objectives": ["تقديم مفاهيم تعلم الآلة الأساسية", "تدريب المتعلمين على بناء نماذج بسيطة"],
            "target_audience": "الخريجون الجدد والمهتمون بالبيانات", "prerequisites": ["معرفة أساسية بالحاسوب"],
            "level": "Beginner", "total_hours": hours,
            "learning_outcomes": [{"id": "LO1", "statement": "يطبق خطوات تجهيز البيانات"},
                                  {"id": "LO2", "statement": "يقيم أداء نموذج تصنيف بسيط"}],
            "modules": [{"id": "M1", "title": "تجهيز البيانات", "hours": hours // 2, "topics": ["تنظيف البيانات"],
                         "description": "تنظيف البيانات وتحضيرها.", "activities": [{"type": "تمرين", "description": "تنظيف ملف بيانات"}],
                         "outcome_ids": ["LO1"]},
                        {"id": "M2", "title": "بناء النماذج وتقييمها", "hours": hours - hours // 2, "topics": ["التصنيف"],
                         "description": "تدريب نموذج وتقييمه.", "activities": [{"type": "مختبر", "description": "تدريب نموذج تصنيف"}],
                         "outcome_ids": ["LO2"]}],
            "assessments": [{"id": "A1", "type": "اختبار عملي", "description": "تجهيز بيانات وتقييم نموذج",
                             "outcome_ids": ["LO1", "LO2"]}],
            "capstone": {"title": "مشروع ختامي", "description": "بناء نموذج كامل وعرض نتائجه",
                         "outcome_ids": ["LO1", "LO2"], "deliverables": ["دفتر ملاحظات", "عرض تقديمي"]},
            "tools_resources": ["Python", "scikit-learn"], "competencies": [{"name": "تحليل البيانات", "source_ids": []}],
        }
    else:
        curriculum = {
            "title": "Machine Learning Foundations", "language": "en",
            "overview": "A practical first course in machine learning with Python.",
            "rationale": "Employers increasingly expect data and modelling skills.",
            "objectives": ["Introduce core machine learning ideas", "Practise building simple models"],
            "target_audience": "New graduates and analysts", "prerequisites": ["Basic computer skills"],
            "level": "Beginner", "total_hours": hours,
            "learning_outcomes": [{"id": "LO1", "statement": "Prepare a dataset for modelling"},
                                  {"id": "LO2", "statement": "Evaluate a simple classification model"}],
            "modules": [{"id": "M1", "title": "Data preparation", "hours": hours // 2, "topics": ["Cleaning data"],
                         "description": "Cleaning and preparing data.", "activities": [{"type": "exercise", "description": "Clean a dataset"}],
                         "outcome_ids": ["LO1"]},
                        {"id": "M2", "title": "Models and evaluation", "hours": hours - hours // 2, "topics": ["Classification"],
                         "description": "Training and evaluating a model.", "activities": [{"type": "lab", "description": "Train a classifier"}],
                         "outcome_ids": ["LO2"]}],
            "assessments": [{"id": "A1", "type": "practical test", "description": "Prepare data and evaluate a model",
                             "outcome_ids": ["LO1", "LO2"]}],
            "capstone": {"title": "Capstone project", "description": "Build and present an end-to-end model",
                         "outcome_ids": ["LO1", "LO2"], "deliverables": ["Notebook", "Presentation"]},
            "tools_resources": ["Python", "scikit-learn"], "competencies": [{"name": "Data analysis", "source_ids": []}],
        }
    curriculum.update({"capstone_omitted_reason": None, "references": [], "academy_evidence": [],
                       "assumptions": [], "open_questions": [], "alignment": None, "data_notice": None})
    return copy.deepcopy(curriculum)


FRAMEWORK = {
    "framework_id": "test-framework", "name": "Test Competency Framework", "publisher": "Test Body",
    "version": "1.0", "source_url": "https://example.org/framework", "retrieved_on": "2026-10-27", "language": "en",
    "requirements": [{"id": "R1", "text": "Prepare and clean a dataset for analysis"},
                     {"id": "R2", "text": "Evaluate classification model performance"},
                     {"id": "R3", "text": "Deploy a model to production with monitoring"}],
}
