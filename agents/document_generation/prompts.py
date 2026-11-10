"""
LangChain prompts for Agent 4, kept apart from the schemas (schemas.py)
and the templates (templates.py).

Each prompt asks for one small structured object. The model is shown the
approved curriculum and told to add nothing to it; validation.py still
checks the answer (accreditation claims, hour figures, module ids).
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic (Modern Standard Arabic)"}

_RULES = """Rules:
- Write every text value in {language_name}. Keep widely used technical terms (e.g. Python, SQL) as they are.
- Use ONLY the approved curriculum given below. Never add facts, statistics, prices, dates, prerequisites, tools or
  hours that it does not state. If something is not in the curriculum, leave it out.
- Mention training hours only with the figures given in the curriculum.
- Never claim the course is accredited, certified, officially approved or compliant with any standard.
- Keep the text short, plain and professional."""


# The participant guide teaches: unlike the other documents it may explain the listed topics with well-established general
# knowledge. What the curriculum states (topics, hours, outcomes) stays as given, and nothing new is added to it.
_TEACHING_RULES = _RULES.replace(
    """- Use ONLY the approved curriculum given below. Never add facts, statistics, prices, dates, prerequisites, tools or
  hours that it does not state. If something is not in the curriculum, leave it out.""",
    """- The approved curriculum below decides WHAT is taught: teach only its listed topics, in its module order. You may
  explain those topics with accurate, well-established general knowledge, but never add topics, tools, prerequisites,
  statistics, prices, dates or hours that the curriculum does not state.""").replace(
    "- Keep the text short, plain and professional.", "- Write plain, professional text a learner can follow.")


def _prompt(task: str, rules: str = _RULES) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You write professional training documents for an academy from an approved curriculum.\n" + rules
         + "\n" + task),
        ("human", "Approved curriculum:\n{curriculum_block}\n{problems_block}"),
    ])


DESCRIPTION = _prompt("Write the summary (2-3 sentences) and the assessment approach (1-2 sentences, using only the listed assessments).")
PROPOSAL = _prompt("Write a proposal: executive summary, background and need (use only the rationale and audience given, no statistics), "
                   "methodology (use only the activity types listed) and the expected deliverables.")
GUIDE = _prompt("Write a participant guide that TEACHES the course. Write a welcome introduction, instructions for the listed "
                "assessments, and for EVERY module id given:\n"
                "- explanation: 4 to 7 sentences in your own words that explain each listed topic of that module (what it is and why "
                "it matters). Teach the content itself, for example what each topic is, how it works and how it is used; do not "
                "announce what the module will cover and do not start with \"In this module\". Do not copy the topic list or the "
                "module title as the explanation.\n"
                "- example: one short worked example that shows a topic of the module in use.\n"
                "- key_concepts: the main ideas of the listed topics.\n"
                "- exercise: a practice exercise for the learner, based on the listed activities.\n"
                "Use each module id exactly as given.", _TEACHING_RULES)
CAPSTONE = _prompt("Write a short introduction to the capstone project and guidance on preparing its deliverables. Use only the project given.")

def _unit_prompt(task: str) -> ChatPromptTemplate:
    """A prompt for ONE unit of a document (one module, or one assessment): the whole curriculum for context, then the unit."""
    return ChatPromptTemplate.from_messages([
        ("system", "You write professional training documents for an academy from an approved curriculum.\n" + _RULES
         + "\n- Never state timings in minutes; the curriculum records hours only.\n" + task),
        ("human", "Approved curriculum:\n{curriculum_block}\n\nWrite ONLY for this part of it:\n{unit_block}\n{problems_block}"),
    ])


INSTRUCTOR_FRAME = _prompt("Write what the instructor should prepare before teaching this course (2-4 sentences), and guidance "
                           "for running and marking the listed assessments (2-4 sentences).")
MODULE_TEACHING = _unit_prompt("For the ONE module given, write: 2 to 4 suggested teaching methods, 1 to 3 things the instructor "
                               "does, notes for the instructor only (what to emphasise, common difficulties) and 2 to 4 "
                               "discussion questions about the module's own topics. Use the module id exactly as given.")
SLIDES_FRAME = _prompt("Write a short closing summary of the whole course (2-3 sentences) for the last slide.")
MODULE_SLIDES = _unit_prompt("For the ONE module given, write the content of its slide: 3 to 5 key points taken from its topics, "
                             "one suggested example, one discussion or activity prompt and one suggested visual element. "
                             "Use the module id exactly as given.")
MODULE_QUESTIONS = _unit_prompt(
    "For the ONE module given, write exactly: 2 multiple-choice questions, 1 true/false question and 1 short-answer question.\n"
    "- Every question must be about the module's own topics and outcomes; do not use anything outside them.\n"
    "- A multiple-choice question has exactly 4 different options and correct_index (0 to 3) is the position of the ONE "
    "option that is correct. Never use \"all of the above\" or \"none of the above\".\n"
    "- Write each option as a complete, meaningful statement of its own. All four options must be different from one "
    "another: never repeat an option, and never refer to another option.\n"
    "- Do not number or label the options and do not put a position or index (0, 1, 2, 3, A, B, C, D) in an option's "
    "text. Only correct_index says which option is correct.\n"
    "- Write only in the requested language (technical terms such as Python or SQL may stay in English) and do not "
    "repeat a sentence within an answer or a rationale. Never put English letters inside an Arabic word.\n"
    "- Every multiple-choice question has its own four options: never reuse the options of another question, even in a "
    "different order. Never use \"all the options\" or similar wording.\n"
    "- End every question and every true/false statement with a question mark or a full stop.\n"
    "- Keep a model answer to one or two sentences (at most 80 words) and every rationale to one or two sentences "
    "(at most 60 words).\n"
    "- The true/false statement must be clearly true or clearly false, and answer says which.\n"
    "- Give a short rationale for every question, and a model answer for the short-answer question.\n"
    "- The four questions must all be different. Use the module id exactly as given.")

# The second, targeted repair of a question bank: ONE question that is still invalid is rewritten; the rest is not touched.
QUESTION_FIX = ChatPromptTemplate.from_messages([
    ("system", "You write professional training documents for an academy from an approved curriculum.\n" + _RULES
     + "\n- Never state timings in minutes; the curriculum records hours only.\n"
     "You are correcting ONE question of a question bank. Write only that question, in the same structure as before.\n"
     "{kind_rules}\n"
     "- Do not number or label options, never repeat an option, never use \"all of the above\" or \"none of the above\", "
     "and do not repeat a sentence within a text."),
    ("human", "Approved curriculum:\n{curriculum_block}\n\nThe question belongs to this part of it:\n{unit_block}\n\n"
              "The module's other questions (yours must be about a different point and must not copy them):\n{others}\n\n"
              "Your first version of this question (JSON):\n{original}\n"
              "Problems found in it:\n{original_problems}\n\n"
              "Your corrected version, which still fails (JSON):\n{current}\n"
              "Problems that remain:\n{current_problems}\n\n"
              "What must change:\n{changes}\n"
              "Write a new version of this one question that fixes every remaining problem."),
])
QUESTION_KIND_RULES = {
    "multiple_choice": "- A multiple-choice question has exactly 4 different options; correct_index (0 to 3) is the position "
                       "of the ONE correct option. Give a short rationale.",
    "true_false": "- A true/false statement is clearly true or clearly false; answer says which. Give a short rationale.",
    "short_answer": "- A short-answer question has a model answer of one or two sentences and a short rationale.",
}
QUESTION_CHANGES = {
    "question_options": "Rewrite the four options so they are different, complete and meaningful statements about the "
                        "module, with no numbering or position index, and no \"all/none of the above\".",
    "question_duplicate": "Write a genuinely different question about another point of the module's topics.",
    "question_ungrounded": "Write the question about the module's own topics and outcomes.",
    "repeated_text": "Say each thing once; keep the text short.",
    "foreign_script": "Write only in the document's language; remove every character from another writing system.",
    "empty_text": "Fill in every field.",
}
ASSESSMENT_RUBRIC = _unit_prompt(
    "For the ONE assessment given, write instructions for the assessor and, for EACH criterion listed (use its outcome id "
    "exactly), four descriptions of the work at the levels Beginning, Developing, Proficient and Advanced.\n"
    "- Describe observable work for the criterion's own learning outcome only; do not add other outcomes.\n"
    "- The four descriptions of a criterion must clearly differ from one another, from weakest to strongest. Change what "
    "the learner does or achieves at each level; never write a level as the previous level with an extra clause added.\n"
    "- The assessor instructions must explain HOW to assess (what evidence to look at and how to apply the four levels). "
    "Do not repeat the assessment's description or purpose.\n"
    "- Write only in the requested language; never put English letters inside an Arabic word.")

PROMPTS = {
    "DescriptionParts": DESCRIPTION, "ProposalParts": PROPOSAL, "GuideParts": GUIDE, "CapstoneParts": CAPSTONE,
    "InstructorFrame": INSTRUCTOR_FRAME, "ModuleTeaching": MODULE_TEACHING, "SlidesFrame": SLIDES_FRAME,
    "ModuleSlides": MODULE_SLIDES, "ModuleQuestions": MODULE_QUESTIONS, "AssessmentRubric": ASSESSMENT_RUBRIC,
}


def unit_block(unit_field: str, content: dict[str, Any], unit_id: str) -> str:
    """The one module (or assessment with its criteria) a unit prompt is about, as plain text."""
    from agents.document_generation import units

    if unit_field == "rubrics":
        spec = units.rubric_spec(content, unit_id) or {}
        lines = [f"Assessment id: {unit_id}", f"Assessment: {spec.get('title')}", f"Description: {spec.get('description')}",
                 "Criteria (one per learning outcome; write four level descriptions for each):"]
        lines += [f"- {c['outcome_id']}: {c['statement']}" for c in spec.get("criteria") or []]
        return "\n".join(lines)
    module = units.find_module(content, unit_id) or {}
    lines = [f"Module id: {unit_id}", f"Module: {module.get('title')} ({module.get('hours')} hours)",
             f"Description: {module.get('description') or '-'}", f"Topics: {'; '.join(module.get('topics') or [])}"]
    lines += [f"Activity ({a.get('type')}): {a.get('description')}" for a in module.get("activities") or []]
    for outcome_id in module.get("outcome_ids") or []:
        lines.append(f"Learning outcome {outcome_id}: {units.outcome_statement(content, outcome_id)}")
    return "\n".join(lines)


def curriculum_block(content: dict[str, Any]) -> str:
    """The approved curriculum as plain text for a prompt."""
    lines = [f"Title: {content.get('title')}", f"Level: {content.get('level')}",
             f"Total hours: {content.get('total_hours')}", f"Target audience: {content.get('target_audience')}",
             f"Overview: {content.get('overview')}", f"Rationale: {content.get('rationale') or '-'}",
             "Objectives:", *[f"- {o}" for o in content.get("objectives") or []], "Learning outcomes:",
             *[f"- {o.get('id')}: {o.get('statement')}" for o in content.get("learning_outcomes") or []],
             "Modules:"]
    for module in content.get("modules") or []:
        lines.append(f"- {module.get('id')} {module.get('title')} ({module.get('hours')} hours). "
                     f"Topics: {'; '.join(module.get('topics') or [])}")
        lines += [f"    activity ({a.get('type')}): {a.get('description')}" for a in module.get("activities") or []]
    lines.append("Assessments:")
    lines += [f"- {a.get('id')} ({a.get('type')}): {a.get('description')}" for a in content.get("assessments") or []]
    capstone = content.get("capstone")
    if isinstance(capstone, dict):
        lines += [f"Capstone: {capstone.get('title')}: {capstone.get('description')}",
                  f"Capstone deliverables: {'; '.join(capstone.get('deliverables') or [])}"]
    return "\n".join(lines)
