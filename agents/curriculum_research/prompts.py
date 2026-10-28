"""
LangChain prompt templates for Agent 3.

Every prompt is short and asks for one small structured object (see
schemas.py): a 3-4B local model is far more reliable that way. The rules
the validator enforces are stated in the prompts too, so the first answer
usually passes; the validator still checks everything.

Evidence is passed as plain text blocks with ids. The model may cite
those ids but never invent sources, academy facts or accreditation.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic (Modern Standard Arabic)"}

_COMMON_RULES = """Rules:
- Write every text value in {language_name}. Keep widely used technical terms (e.g. Python, SQL, Power BI) as they are.
- Use only the evidence given. Never invent facts about the academy, statistics, laws or sources.
- Never claim the course is accredited, certified, officially approved or compliant with any standard.
- Cite evidence only by the ids given (e.g. SRC-..., a course id). If there is no evidence, write a general rationale without citations.
- Keep text short: descriptions one or two sentences, titles under ten words, activity and assessment types one or two words (never ids)."""

OUTLINE = ChatPromptTemplate.from_messages([
    ("system", "You design practical training curricula for a professional training academy.\n" + _COMMON_RULES + """
- Learning outcomes: 4 to 8, ids "LO1", "LO2", ...; each starts with ONE measurable action verb ({verb_examples}). Never start with "understand", "know" or "learn".
- Modules: ids "M1", "M2", ...; choose the number of modules that fits the topic and the hours; each module has whole hours above zero and the module hours add up to EXACTLY {total_hours}.
- level must be exactly one of: Beginner, Intermediate, Advanced (in English, even for Arabic content)."""),
    ("human", """Design the outline of a {level} curriculum.
Topic: {topic}
Target audience: {audience}
Total hours: {total_hours}
{revision_block}
Academy evidence:
{academy_block}

Research sources:
{sources_block}"""),
])

MODULE_DETAILS = ChatPromptTemplate.from_messages([
    ("system", "You write the module details of a training curriculum.\n" + _COMMON_RULES + """
- Return one entry per module id given, in the same order.
- description: two sentences. activities: one to three practical activities (lab, exercise, case study, project work).
- outcome_ids: the learning outcome ids (from the list given) that the module develops. Every outcome must be developed by at least one module."""),
    ("human", """Curriculum: {title} ({level}, {total_hours} hours) for {audience}.

Learning outcomes:
{outcomes_block}

Modules:
{modules_block}"""),
])

EXTRAS = ChatPromptTemplate.from_messages([
    ("system", "You complete a training curriculum: assessments, capstone, resources and competencies.\n" + _COMMON_RULES + """
- assessments: ids "A1", "A2", ...; each lists the outcome ids it assesses. Together with the capstone, every outcome is assessed at least once.
- capstone: a practical final project citing outcome ids, with deliverables{capstone_rule}.
- tools_resources: software, datasets or materials learners need.
- competencies: the skills the course builds; source_ids only from the research source ids given, otherwise empty.
- assumptions: what you assumed that the request did not state. open_questions: what the academy should still decide."""),
    ("human", """Curriculum: {title} ({level}, {total_hours} hours) for {audience}.

Learning outcomes:
{outcomes_block}

Modules:
{modules_block}

Research sources:
{sources_block}"""),
])

REPAIR = ChatPromptTemplate.from_messages([
    ("system", "You fix a training curriculum so that it passes validation.\n" + _COMMON_RULES + """
- Fix ONLY the problems listed. Keep ids unchanged. Return full lists for the parts you change; leave the others empty.
- Outcomes must start with one measurable action verb ({verb_examples}).
- Every outcome must be developed by a module and assessed by an assessment or the capstone."""),
    ("human", """Problems:
{errors_block}

Curriculum (JSON):
{curriculum_json}"""),
])

NEW_MODULE = ChatPromptTemplate.from_messages([
    ("system", "You add one module to an existing training curriculum.\n" + _COMMON_RULES + """
- outcome_ids: existing outcome ids (from the list) that the new module develops.
- one to three practical activities."""),
    ("human", """Curriculum: {title} ({level}) for {audience}.
Existing modules: {module_titles}
Learning outcomes:
{outcomes_block}

Add a module about: {subject}"""),
])

CAPSTONE = ChatPromptTemplate.from_messages([
    ("system", "You write the capstone project of an existing training curriculum.\n" + _COMMON_RULES + """
- The capstone is a practical final project that demonstrates several outcomes (cite their ids) and lists deliverables."""),
    ("human", """Curriculum: {title} ({level}) for {audience}.
Modules: {module_titles}
Learning outcomes:
{outcomes_block}
{instruction}"""),
])

SLOTS = ChatPromptTemplate.from_messages([
    ("system", """You read an administrator's request for a new training course and extract what it asks for.
Copy values from the message only; leave a field empty when the message does not state it. Do not guess."""),
    ("human", "{text}"),
])

SUGGEST = ChatPromptTemplate.from_messages([
    ("system", "You advise a training academy on new courses to add to its catalogue.\n" + _COMMON_RULES + """
- Suggest at most {max_suggestions} new courses that answer the catalogue gaps listed (cite their ids in gap_ids) or the research sources (source_ids).
- Do not suggest a course the catalogue already has. prerequisite_course_ids only from the existing course ids listed.
- level: Beginner, Intermediate or Advanced. proposed_hours: a whole number. confidence: low unless the evidence is strong.
- The catalogue data is {data_status}; do not present gaps as verified market needs."""),
    ("human", """Request: {request}

Existing courses:
{courses_block}

Catalogue gaps (from catalogue data):
{gaps_block}

Research sources:
{sources_block}"""),
])

ALIGN = ChatPromptTemplate.from_messages([
    ("system", """You compare a training curriculum with the requirements of a competency framework.
For EACH requirement listed, give one judgement:
- covered: the curriculum clearly teaches it and assesses it;
- partially_covered: it is taught or assessed only in part;
- not_covered: nothing in the curriculum addresses it;
- unable_to_determine: the requirement or the curriculum text is not clear enough to judge.
element_ids: the ids (LO*, M*, A*) of the curriculum elements that show it -- required for covered and
partially_covered, only ids from the list given. "Possible evidence" is a word-overlap hint, not proof.
explanation: one sentence. recommendation: for partially_covered or not_covered, one sentence on what to add.
Write explanation and recommendation in {language_name}. Never claim accreditation, certification or compliance."""),
    ("human", """Framework: {framework_name}

Curriculum elements:
{elements_block}

Requirements:
{requirements_block}"""),
])

VERB_EXAMPLES = {
    "en": "apply, analyse, build, design, evaluate, implement, create, compare",
    "ar": "يطبق، يحلل، يبني، يصمم، يقيّم، ينفذ، ينشئ، يقارن",
}
