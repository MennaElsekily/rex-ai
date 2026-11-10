"""
The approved document templates.

A template is data: the document type, its label, the ordered section
keys, the curriculum fields it cannot be built without and the schema of
the narrative parts the language model writes (None when the document is
built from the curriculum alone). composer.py turns each section key into
content blocks. To add a document type: add a template here, the section
builders it needs in composer.py and, if it has narrative parts, a schema
and a prompt. Nothing else changes.

These templates hold no prompts (see prompts.py) and no layout (pdf.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel

from agents.document_generation import schemas

# Curriculum fields every document needs; a document is never built
# without them (nothing is invented to fill a gap).
BASE_REQUIRED = ("title", "overview", "objectives", "learning_outcomes", "modules", "total_hours")


@dataclass(frozen=True)
class DocumentTemplate:
    doc_type: str
    label: dict[str, str]
    sections: tuple[str, ...]
    parts_schema: Optional[Type[BaseModel]] = None
    required: tuple[str, ...] = BASE_REQUIRED
    # Documents written one unit at a time (one model call per module, or per assessment for rubrics): the schema of
    # one unit, the field of parts_schema that holds the units, and the schema of the text written once for the whole
    # document (None when there is none). See generator.py and units.py.
    unit_schema: Optional[Type[BaseModel]] = None
    unit_field: str = "modules"
    frame_schema: Optional[Type[BaseModel]] = None
    # Model text may not state minute-level timings (the curriculum records hours only).
    forbid_minutes: bool = False
    # After the first repair fails, rewrite only the items still flagged, once more (generator._targeted_repair).
    targeted_repair: bool = False


TEMPLATES: dict[str, DocumentTemplate] = {t.doc_type: t for t in (
    DocumentTemplate(
        "course_description",
        {"en": "Course Description", "ar": "وصف الدورة"},
        ("overview", "summary", "objectives", "audience", "prerequisites", "outcomes", "modules_table",
         "hours", "format", "assessment", "traceability"),
        schemas.DescriptionParts),
    DocumentTemplate(
        "course_proposal",
        {"en": "Course Proposal", "ar": "مقترح دورة"},
        ("proposal_summary", "background", "objectives", "audience", "prerequisites", "outcomes", "methodology",
         "schedule", "activities", "assessment", "deliverables", "traceability"),
        schemas.ProposalParts),
    DocumentTemplate(
        "participant_guide",
        {"en": "Participant Guide", "ar": "دليل المتدرب"},
        ("guide_introduction", "objectives", "outcomes", "modules_table", "module_guides", "assessment_guide",
         "capstone", "references", "traceability"),
        schemas.GuideParts),
    DocumentTemplate(
        "capstone_brief",
        {"en": "Capstone Project Brief", "ar": "ملخص المشروع الختامي"},
        ("capstone_intro", "capstone", "capstone_outcomes", "capstone_guidance", "traceability"),
        schemas.CapstoneParts,
        BASE_REQUIRED + ("capstone",)),
    DocumentTemplate(
        "outcomes_assessment_plan",
        {"en": "Learning Outcomes and Assessment Plan", "ar": "مخرجات التعلم وخطة التقييم"},
        ("outcomes", "outcome_matrix", "assessment", "capstone", "traceability"),
        None,
        BASE_REQUIRED + ("assessments",)),
    DocumentTemplate(
        "instructor_guide",
        {"en": "Facilitator / Instructor Guide", "ar": "دليل المدرب"},
        ("instructor_note", "overview", "instructor_preparation", "teaching_objectives", "modules_table",
         "module_teaching", "materials", "assessment_guidance", "outcomes", "traceability"),
        schemas.InstructorParts, unit_schema=schemas.ModuleTeaching, frame_schema=schemas.InstructorFrame,
        forbid_minutes=True),
    DocumentTemplate(
        "presentation_outline",
        {"en": "Presentation Outline", "ar": "مخطط العرض التقديمي"},
        ("slides_title", "slides_overview", "slides_objectives", "slides_outcomes", "module_slides",
         "slides_assessment", "slides_closing", "traceability"),
        schemas.SlidesParts, unit_schema=schemas.ModuleSlides, frame_schema=schemas.SlidesFrame, forbid_minutes=True),
    DocumentTemplate(
        "question_bank",
        {"en": "Question Bank", "ar": "بنك الأسئلة"},
        ("qb_overview", "questions", "traceability"),
        schemas.QuestionBankParts, unit_schema=schemas.ModuleQuestions, forbid_minutes=True, targeted_repair=True),
    DocumentTemplate(
        "assessment_rubric",
        {"en": "Assessment Rubrics", "ar": "معايير التقييم"},
        ("rubric_overview", "rubrics", "scoring_note", "traceability"),
        schemas.RubricParts, BASE_REQUIRED + ("assessments",),
        unit_schema=schemas.AssessmentRubric, unit_field="rubrics", forbid_minutes=True),
)}

CURRICULUM_PDF = "curriculum_pdf"

# The curriculum PDF is not a generated document: it is Agent 3's own PDF of an approved curriculum, offered for
# download in the Documents tab (AdminService.approved_curriculum_pdf). A request to *generate* it is explained.
UNSUPPORTED: dict[str, dict[str, str]] = {
    CURRICULUM_PDF: {"en": "Curriculum PDF", "ar": "ملف PDF للمنهج",
                     "reason": "The curriculum PDF is not generated here: it is the Curriculum Agent's own PDF of an "
                               "approved curriculum. In the Documents tab choose \"Curriculum PDF\" to download it."},
}

SECTION_HEADINGS: dict[str, dict[str, str]] = {
    "overview": {"en": "Course overview", "ar": "نظرة عامة على الدورة"},
    "summary": {"en": "Summary", "ar": "ملخص"},
    "objectives": {"en": "Course objectives", "ar": "أهداف الدورة"},
    "audience": {"en": "Target audience", "ar": "الجمهور المستهدف"},
    "prerequisites": {"en": "Prerequisites", "ar": "المتطلبات السابقة"},
    "outcomes": {"en": "Learning outcomes", "ar": "مخرجات التعلم"},
    "modules_table": {"en": "Modules and topics", "ar": "الوحدات والموضوعات"},
    "hours": {"en": "Training hours", "ar": "ساعات التدريب"},
    "format": {"en": "Training format", "ar": "نمط التدريب"},
    "assessment": {"en": "Assessment approach", "ar": "أسلوب التقييم"},
    "proposal_summary": {"en": "Executive summary", "ar": "الملخص التنفيذي"},
    "background": {"en": "Background and training need", "ar": "الخلفية والحاجة التدريبية"},
    "methodology": {"en": "Training methodology", "ar": "منهجية التدريب"},
    "schedule": {"en": "Modules and schedule", "ar": "الوحدات والجدول"},
    "activities": {"en": "Practical activities", "ar": "الأنشطة العملية"},
    "deliverables": {"en": "Expected deliverables", "ar": "المخرجات المتوقعة"},
    "guide_introduction": {"en": "Course introduction", "ar": "مقدمة الدورة"},
    "module_guides": {"en": "Module-by-module content", "ar": "محتوى الوحدات"},
    "assessment_guide": {"en": "Assessment instructions", "ar": "تعليمات التقييم"},
    "capstone": {"en": "Capstone project", "ar": "المشروع الختامي"},
    "references": {"en": "References", "ar": "المراجع"},
    "capstone_intro": {"en": "Introduction", "ar": "مقدمة"},
    "capstone_outcomes": {"en": "Outcomes the project develops", "ar": "المخرجات التي يطورها المشروع"},
    "capstone_guidance": {"en": "Guidance for participants", "ar": "إرشادات للمتدربين"},
    "outcome_matrix": {"en": "Outcomes, modules and assessments", "ar": "المخرجات والوحدات والتقييمات"},
    "traceability": {"en": "Source and traceability", "ar": "المصدر وإمكانية التتبع"},
}

# Static wording used inside blocks (not part of the curriculum).
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "none_listed": "None listed in the approved curriculum.",
        "not_specified": "Not specified in the approved curriculum.",
        "no_capstone": "The approved curriculum has no capstone project.",
        "no_references": "The approved curriculum lists no references.",
        "hours_total": "Total training hours: {hours}",
        "module": "Module", "topics": "Topics", "hours": "Hours", "outcomes": "Outcomes",
        "outcome": "Learning outcome", "modules": "Modules", "assessments": "Assessments",
        "type": "Type", "description": "Description", "id": "ID",
        "key_concepts": "Key concepts", "exercise": "Practical exercise", "activities": "Activities",
        "explanation": "Explanation", "guide_example": "Example",
        "deliverables": "Deliverables", "organization": "Prepared for: {value}",
        "schedule_notes": "Schedule notes: {value}",
        "trace_curriculum": "Curriculum: {title} ({curriculum_id}), version {version}, approved by the academy.",
        "trace_level": "Level: {level}. Total hours: {hours}.",
        "trace_sources": "Sources recorded in the curriculum: {sources}",
        "trace_none": "The curriculum records no external research sources.",
        "disclaimer": ("This document was prepared with AI assistance from the approved curriculum above and must be "
                       "reviewed and approved by the academy before use. It makes no claim of accreditation, "
                       "certification or official approval."),
        "total": "Total",
    },
    "ar": {
        "none_listed": "لا يوجد في المنهج المعتمد.",
        "not_specified": "غير محدد في المنهج المعتمد.",
        "no_capstone": "لا يتضمن المنهج المعتمد مشروعًا ختاميًا.",
        "no_references": "لا يذكر المنهج المعتمد أي مراجع.",
        "hours_total": "إجمالي ساعات التدريب: {hours}",
        "module": "الوحدة", "topics": "الموضوعات", "hours": "الساعات", "outcomes": "المخرجات",
        "outcome": "مخرج التعلم", "modules": "الوحدات", "assessments": "التقييمات",
        "type": "النوع", "description": "الوصف", "id": "المعرّف",
        "key_concepts": "المفاهيم الأساسية", "exercise": "التمرين العملي", "activities": "الأنشطة",
        "explanation": "الشرح", "guide_example": "مثال",
        "deliverables": "المخرجات المطلوبة", "organization": "أُعدّ لـ: {value}",
        "schedule_notes": "ملاحظات الجدول: {value}",
        "trace_curriculum": "المنهج: {title} ({curriculum_id})، النسخة {version}، معتمد من الأكاديمية.",
        "trace_level": "المستوى: {level}. إجمالي الساعات: {hours}.",
        "trace_sources": "المصادر المسجلة في المنهج: {sources}",
        "trace_none": "لا يسجل المنهج أي مصادر بحث خارجية.",
        "disclaimer": ("أُعدّت هذه الوثيقة بمساعدة الذكاء الاصطناعي اعتمادًا على المنهج المعتمد المذكور أعلاه، ويجب أن "
                       "تراجعها الأكاديمية وتعتمدها قبل استخدامها. ولا تدّعي أي اعتماد أو شهادة أو موافقة رسمية."),
        "total": "الإجمالي",
    },
}

# Extra information the administrator may supply for any document; each
# fills something the curriculum does not record. Keys and limits.
EXTRA_FIELDS: dict[str, int] = {"training_format": 200, "organization": 120, "schedule_notes": 300}


# Names for the fields a document can report as "not specified in the curriculum" (content["missing"] keeps
# the raw field key; these are for display), and the curriculum level words in Arabic.
FIELD_LABELS: dict[str, dict[str, str]] = {
    "training_format": {"en": "Training format", "ar": "نمط التدريب"},
    "target_audience": {"en": "Target audience", "ar": "الجمهور المستهدف"},
    "assessment_outcomes": {"en": "Learning outcomes linked to an assessment", "ar": "مخرجات التعلم المرتبطة بالتقييم"},
}
LEVEL_LABELS: dict[str, dict[str, str]] = {
    "ar": {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"},
}


def field_label(field: str, language: str) -> str:
    """A readable name for a missing-information field key (the key itself if it has none)."""
    return FIELD_LABELS.get(field, {}).get(language, field)


def hours_text(hours: Any, language: str) -> str:
    """
    A number of hours as a phrase: "1 hour", "20 hours"; Arabic with its own number rules ("ساعة واحدة", "ساعتان",
    "5 ساعات", "20 ساعة"). Not the column label "Hours"/"الساعات", which is not a unit of a number.
    """
    try:
        value = float(hours)
    except (TypeError, ValueError):
        return str(hours)
    number = int(value) if value == int(value) else value
    if language == "ar":
        if number == 1:
            return "ساعة واحدة"
        if number == 2:
            return "ساعتان"
        return f"{number} ساعات" if isinstance(number, int) and 3 <= number <= 10 else f"{number} ساعة"
    return f"{number} {'hour' if number == 1 else 'hours'}"


def level_label(level: Optional[str], language: str) -> str:
    """The curriculum level in the document's language (Arabic words in Arabic documents)."""
    return LEVEL_LABELS.get(language, {}).get(level, level or "")


SECTION_HEADINGS.update({
    "instructor_note": {"en": "Instructor-only document", "ar": "وثيقة للمدرب فقط"},
    "instructor_preparation": {"en": "Instructor preparation", "ar": "تحضير المدرب"},
    "teaching_objectives": {"en": "Teaching objectives", "ar": "أهداف التدريس"},
    "module_teaching": {"en": "Module-by-module teaching guide", "ar": "دليل التدريس لكل وحدة"},
    "materials": {"en": "Required materials and resources", "ar": "المواد والموارد المطلوبة"},
    "assessment_guidance": {"en": "Assessment guidance", "ar": "إرشادات التقييم"},
    "slides_title": {"en": "Title slide", "ar": "شريحة العنوان"},
    "slides_overview": {"en": "Course overview slide", "ar": "شريحة نظرة عامة"},
    "slides_objectives": {"en": "Objectives slide", "ar": "شريحة الأهداف"},
    "slides_outcomes": {"en": "Learning outcomes slide", "ar": "شريحة مخرجات التعلم"},
    "module_slides": {"en": "Module slides", "ar": "شرائح الوحدات"},
    "slides_assessment": {"en": "Assessment slides", "ar": "شرائح التقييم"},
    "slides_closing": {"en": "Closing slide", "ar": "شريحة الختام"},
    "qb_overview": {"en": "About this question bank", "ar": "حول بنك الأسئلة"},
    "questions": {"en": "Questions by module", "ar": "الأسئلة حسب الوحدة"},
    "rubric_overview": {"en": "About these rubrics", "ar": "حول معايير التقييم"},
    "rubrics": {"en": "Rubrics", "ar": "معايير التقييم لكل تقييم"},
    "scoring_note": {"en": "How the scores are set", "ar": "كيفية تحديد الدرجات"},
})

_NEW_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "instructor_only_notice": ("This guide contains teaching guidance and notes for the instructor and must not be handed "
                                   "to participants. Participant-facing material is in the Participant Guide."),
        # the wording of documents made before the notice was reworded (see composer.compose(legacy=True))
        "instructor_only_notice_legacy": ("Instructor-only document. It contains teaching guidance and answers' rationale that "
                                          "must not be handed to participants. Participant-facing material is in the Participant Guide."),
        "session_structure": "Session structure and timing",
        "session_text": "Allow the module's {hours} hours. Cover the topics in this order: {topics}.",
        "participant_activities": "Participant activities (from the curriculum)",
        "teaching_methods": "Suggested teaching methods (instructor guidance)",
        "instructor_activities": "Instructor activities",
        "instructor_notes": "Instructor notes (instructor only)",
        "discussion_questions": "Discussion questions",
        "slide": "Slide",
        "purpose": "Purpose",
        "slide_purpose_title": "Open the session and introduce the course.",
        "slide_purpose_overview": "Orient participants to what the course is about.",
        "slide_purpose_objectives": "State what the course aims to achieve.",
        "slide_purpose_outcomes": "Show what participants will be able to do.",
        "slide_purpose_module": "Teach module {id}: {title}.",
        "slide_purpose_assessment": "Explain how learning is assessed.",
        "slide_purpose_capstone": "Introduce the capstone project.",
        "slide_purpose_closing": "Summarise the course and close the session.",
        "topics": "Topics covered",
        "key_points": "Key points",
        "example": "Suggested example",
        "discussion_prompt": "Discussion or activity prompt",
        "visual": "Suggested visual element",
        "curriculum_activities": "Activities from the curriculum",
        "outcomes_recap": "Learning outcomes recap",
        "slide_hours_level": "Level: {level}. Total hours: {hours}.",
        "qb_intro": ("Questions are grouped by module. Each module has two multiple-choice questions, one true/false "
                     "question and one short-answer question. The difficulty of each question follows its position, and "
                     "each question is mapped to the learning outcomes of its module. Questions, options, answers and "
                     "explanations can be edited one by one."),
        "question": "Question", "correct_answer": "Correct answer", "rationale": "Explanation", "model_answer": "Model answer",
        "q_mc": "Multiple choice", "q_tf": "True or false", "q_sa": "Short answer",
        "a_mc": "Quiz (multiple choice)", "a_tf": "Quiz (true or false)", "a_sa": "Short-answer question",
        "difficulty": {"easy": "Easy", "medium": "Medium", "hard": "Hard"},
        "true": "True", "false": "False",
        "outcomes_label": "Outcomes",
        "rubric": "Rubric",
        "rubric_intro": ("Each rubric scores work against the learning outcomes the assessment is linked to. Every criterion "
                         "has four performance levels. A level earns a quarter, half, three quarters or all of the "
                         "criterion's points."),
        "level": {"beginning": "Beginning", "developing": "Developing", "proficient": "Proficient", "advanced": "Advanced"},
        "level_share": "{level}: {share} of a criterion's points",
        "related_outcomes": "Related learning outcomes",
        "scoring_summary": "Scoring summary",
        "criterion": "Criterion", "weight": "Weight", "points": "points", "total_possible": "Total possible score",
        "assessment_instructions": "Assessment instructions",
        "scoring_note_text": ("Each rubric is worth exactly 100 points. The points are shared equally between its criteria; "
                              "when 100 does not divide evenly, the remaining points go one at a time to the first "
                              "criteria. A level earns 1/4, 2/4, 3/4 or 4/4 of its criterion's weight."),
    },
    "ar": {
        "instructor_only_notice": ("يحتوي هذا الدليل على إرشادات التدريس وملاحظات المدرب ولا يجوز تسليمه للمتدربين. "
                                   "المادة الموجهة للمتدربين موجودة في دليل المتدرب."),
        "instructor_only_notice_legacy": ("وثيقة للمدرب فقط. تحتوي على إرشادات التدريس وتفسير الإجابات ولا يجوز تسليمها "
                                          "للمتدربين. المادة الموجهة للمتدربين موجودة في دليل المتدرب."),
        "session_structure": "هيكل الجلسة والتوقيت",
        "session_text": "خصّص ساعات الوحدة ({hours}). غطِّ الموضوعات بهذا الترتيب: {topics}.",
        "participant_activities": "أنشطة المتدربين (من المنهج)",
        "teaching_methods": "أساليب التدريس المقترحة (إرشاد للمدرب)",
        "instructor_activities": "أنشطة المدرب",
        "instructor_notes": "ملاحظات المدرب (للمدرب فقط)",
        "discussion_questions": "أسئلة للنقاش",
        "slide": "الشريحة",
        "purpose": "الغرض",
        "slide_purpose_title": "افتتاح الجلسة وتقديم الدورة.",
        "slide_purpose_overview": "تعريف المتدربين بموضوع الدورة.",
        "slide_purpose_objectives": "توضيح ما تهدف الدورة إلى تحقيقه.",
        "slide_purpose_outcomes": "عرض ما سيتمكن المتدربون من فعله.",
        "slide_purpose_module": "تدريس الوحدة {id}: {title}.",
        "slide_purpose_assessment": "شرح كيفية تقييم التعلم.",
        "slide_purpose_capstone": "تقديم المشروع الختامي.",
        "slide_purpose_closing": "تلخيص الدورة وإنهاء الجلسة.",
        "topics": "الموضوعات المغطاة",
        "key_points": "النقاط الرئيسية",
        "example": "مثال مقترح",
        "discussion_prompt": "سؤال نقاش أو نشاط",
        "visual": "عنصر مرئي مقترح",
        "curriculum_activities": "أنشطة من المنهج",
        "outcomes_recap": "مراجعة مخرجات التعلم",
        "slide_hours_level": "المستوى: {level}. إجمالي الساعات: {hours}.",
        "qb_intro": ("الأسئلة مجمعة حسب الوحدة. لكل وحدة سؤالان من نوع الاختيار من متعدد وسؤال صح أو خطأ وسؤال "
                     "بإجابة قصيرة. تتبع صعوبة كل سؤال موضعه، ويرتبط كل سؤال بمخرجات التعلم الخاصة بوحدته. يمكن "
                     "تعديل الأسئلة والخيارات والإجابات والتفسيرات واحدًا تلو الآخر."),
        "question": "السؤال", "correct_answer": "الإجابة الصحيحة", "rationale": "التفسير", "model_answer": "الإجابة النموذجية",
        "q_mc": "اختيار من متعدد", "q_tf": "صح أو خطأ", "q_sa": "إجابة قصيرة",
        "a_mc": "اختبار قصير (اختيار من متعدد)", "a_tf": "اختبار قصير (صح أو خطأ)", "a_sa": "سؤال بإجابة قصيرة",
        "difficulty": {"easy": "سهل", "medium": "متوسط", "hard": "صعب"},
        "true": "صحيح", "false": "خطأ",
        "outcomes_label": "المخرجات",
        "rubric": "معيار التقييم",
        "rubric_intro": ("يقيّم كل معيار العمل وفق مخرجات التعلم المرتبطة بالتقييم. لكل بند أربعة مستويات أداء، ويحصل "
                         "المستوى على ربع درجات البند أو نصفها أو ثلاثة أرباعها أو كاملها."),
        "level": {"beginning": "بداية", "developing": "نامٍ", "proficient": "متمكن", "advanced": "متقدم"},
        "level_share": "{level}: {share} من درجات البند",
        "related_outcomes": "مخرجات التعلم المرتبطة",
        "scoring_summary": "ملخص الدرجات",
        "criterion": "البند", "weight": "الوزن", "points": "درجة", "total_possible": "إجمالي الدرجة الممكنة",
        "assessment_instructions": "تعليمات التقييم",
        "scoring_note_text": ("قيمة كل معيار تقييم 100 درجة بالضبط. توزَّع الدرجات بالتساوي بين بنوده؛ وإذا لم تنقسم 100 "
                              "بالتساوي تُضاف الدرجات المتبقية درجة درجة إلى البنود الأولى. ويحصل المستوى على 1/4 أو "
                              "2/4 أو 3/4 أو 4/4 من وزن بنده."),
    },
}
for _language, _entries in _NEW_STRINGS.items():
    STRINGS[_language].update(_entries)


def supported_types() -> list[str]:
    return list(TEMPLATES)


def get_template(doc_type: str) -> Optional[DocumentTemplate]:
    return TEMPLATES.get(doc_type)


def type_label(doc_type: str, language: str) -> str:
    template = TEMPLATES.get(doc_type)
    if template:
        return template.label[language]
    return UNSUPPORTED.get(doc_type, {}).get(language, doc_type)
