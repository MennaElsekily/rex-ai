"""
Data shapes for Agent 4 (Educational Content & Document Generation).

- The *LLM part* models are what the language model returns through
  LangChain structured output. They are small on purpose (a 3-4B local
  model is reliable with short, flat answers) and cover only the
  narrative text. Everything the approved curriculum already states
  (title, objectives, outcomes, modules, hours, assessments, capstone) is
  copied from it by composer.py, never re-written by the model.
- The stored document *content* is a plain JSON-ready dict:

      {"document_type", "title", "language", "curriculum_id",
       "curriculum_version", "sections": [{"key", "heading", "blocks": [...]}],
       "missing": [{"section", "field"}], "extra": {...}}

  A block is {"type": "paragraph" | "bullets" | "heading" | "table",
  "source": "curriculum" | "generated" | "admin" | "missing", ...}. The
  source says where the words came from: only "generated" and "admin"
  blocks can be edited; "curriculum" blocks change only by revising and
  re-approving the curriculum.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

BLOCK_TYPES = ("paragraph", "bullets", "heading", "table")
BLOCK_SOURCES = ("curriculum", "generated", "admin", "missing")
EDITABLE_SOURCES = ("generated", "admin")
EDITABLE_TYPES = ("paragraph", "bullets")

Id = Annotated[str, Field(max_length=12)]
Short = Annotated[str, Field(max_length=200)]
Text = Annotated[str, Field(max_length=700)]
Teaching = Annotated[str, Field(max_length=1500)]   # explanatory text a participant reads (participant guide)


class DescriptionParts(BaseModel):
    summary: Text = Field(description="Two or three sentences summarising the course, using only the curriculum given.")
    assessment_approach: Text = Field(description="One or two sentences describing how learners are assessed, using only the listed assessments.")


class ProposalParts(BaseModel):
    executive_summary: Text = Field(description="Three or four sentences presenting the course to a client.")
    background_need: Text = Field(description="Why the course is needed, using only the curriculum's rationale and audience. No statistics.")
    methodology: Text = Field(description="How the course is delivered, based only on the listed activity types.")
    expected_deliverables: list[Short] = Field(min_length=1, max_length=6, description="What participants and the client receive; base it on the assessments and capstone listed.")


class ModuleGuide(BaseModel):
    module_id: Id = Field(description='The module id exactly as given, e.g. "M1".')
    explanation: Teaching = Field(description="A teaching explanation of the module for a learner, in your own words: 4 to 7 sentences that explain what each listed topic is and why it matters. Do not just list the topics.")
    example: Teaching = Field(description="One short worked example or illustration (2 to 4 sentences) that shows one of the module's topics in use.")
    key_concepts: list[Short] = Field(min_length=2, max_length=6, description="Key concepts the module covers; use the listed topics.")
    exercise: Text = Field(description="Practice instructions for the participant, based on the module's listed activities.")


class GuideParts(BaseModel):
    introduction: Text = Field(description="A short welcome and introduction to the course for participants.")
    assessment_instructions: Text = Field(description="Instructions for participants about the listed assessments.")
    modules: list[ModuleGuide] = Field(min_length=1, max_length=12, description="Exactly one entry per module id given.")


class CapstoneParts(BaseModel):
    introduction: Text = Field(description="A short introduction to the capstone project for participants, using only the project given.")
    evaluation_guidance: Text = Field(description="How participants should prepare their deliverables, using only the project given.")


# --------------------------------------------------------------------
# Documents written per unit: one model call per module (or per assessment for rubrics), then assembled.
# The container classes (InstructorParts, SlidesParts, QuestionBankParts, RubricParts) are what the composer reads;
# the *Frame classes are the document-wide text written once. Counts are fixed with min_length = max_length so the
# model's structured output (and the validator) can rely on them.
# --------------------------------------------------------------------

class ModuleTeaching(BaseModel):
    module_id: Id = Field(description='The module id exactly as given, e.g. "M1".')
    teaching_methods: list[Short] = Field(min_length=2, max_length=4, description="Suggested teaching methods for this module.")
    instructor_activities: list[Short] = Field(min_length=1, max_length=3, description="What the instructor does in this module.")
    instructor_notes: Text = Field(description="Notes for the instructor only: what to emphasise and common difficulties.")
    discussion_questions: list[Short] = Field(min_length=2, max_length=4, description="Discussion questions about the module's topics.")


class InstructorFrame(BaseModel):
    preparation: Text = Field(description="What the instructor should prepare before teaching the course.")
    assessment_guidance: Text = Field(description="How the instructor should run and mark the listed assessments.")


class InstructorParts(InstructorFrame):
    modules: list[ModuleTeaching] = Field(min_length=1, max_length=12)


class ModuleSlides(BaseModel):
    module_id: Id = Field(description='The module id exactly as given, e.g. "M1".')
    key_points: list[Short] = Field(min_length=3, max_length=5, description="Key points for the module's slide, from its topics.")
    example: Text = Field(description="One suggested example to show, based on the module's topics and activities.")
    discussion_prompt: Short = Field(description="A discussion prompt or activity for the slide.")
    visual: Short = Field(description="A suggested visual element (diagram, chart, screenshot...).")


class SlidesFrame(BaseModel):
    closing_summary: Text = Field(description="A short closing summary of the course for the last slide.")


class SlidesParts(SlidesFrame):
    modules: list[ModuleSlides] = Field(min_length=1, max_length=12)


class McQuestion(BaseModel):
    question: Text = Field(description="A multiple-choice question about this module's topics.")
    options: list[Short] = Field(min_length=4, max_length=4, description="Exactly four different answer options; exactly one is correct.")
    correct_index: int = Field(ge=0, le=3, description="Position (0 to 3) of the one correct option.")
    rationale: Text = Field(description="Why the correct option is correct.")


class TfQuestion(BaseModel):
    statement: Text = Field(description="A statement about this module's topics that is clearly true or clearly false.")
    answer: bool = Field(description="true if the statement is true, false if it is false.")
    rationale: Text = Field(description="Why the statement is true or false.")


class SaQuestion(BaseModel):
    question: Text = Field(description="A short-answer question about this module's topics.")
    model_answer: Text = Field(description="A model answer of one or two sentences.")
    rationale: Text = Field(description="What a good answer must show.")


class ModuleQuestions(BaseModel):
    module_id: Id = Field(description='The module id exactly as given, e.g. "M1".')
    multiple_choice: list[McQuestion] = Field(min_length=2, max_length=2)
    true_false: list[TfQuestion] = Field(min_length=1, max_length=1)
    short_answer: list[SaQuestion] = Field(min_length=1, max_length=1)


class QuestionBankParts(BaseModel):
    modules: list[ModuleQuestions] = Field(min_length=1, max_length=12)


class CriterionLevels(BaseModel):
    outcome_id: Id = Field(description='The learning outcome id of the criterion exactly as given, e.g. "LO1".')
    beginning: Text = Field(description="What work at the Beginning level looks like for this criterion.")
    developing: Text = Field(description="What work at the Developing level looks like.")
    proficient: Text = Field(description="What work at the Proficient level looks like.")
    advanced: Text = Field(description="What work at the Advanced level looks like.")


class AssessmentRubric(BaseModel):
    assessment_id: Id = Field(description="The assessment id exactly as given, e.g. \"A1\" or \"CAP\".")
    instructions: Text = Field(description="Instructions for the assessor on using this rubric.")
    criteria: list[CriterionLevels] = Field(min_length=1, max_length=12, description="One entry per criterion given.")


class RubricParts(BaseModel):
    rubrics: list[AssessmentRubric] = Field(min_length=1, max_length=20)


# --------------------------------------------------------------------
# Block constructors (used by composer.py and the admin edit path)
# --------------------------------------------------------------------

def _with_role(block: dict, role: str | None) -> dict:
    # A role names what a block is for (a question stem, the answer options...) so the validator can check stored
    # content after an edit. It is not part of the block's layout.
    if role:
        block["role"] = role
    return block


def paragraph(text: str, source: str, role: str | None = None) -> dict:
    return _with_role({"type": "paragraph", "text": text, "source": source}, role)


def bullets(items: list[str], source: str, role: str | None = None) -> dict:
    return _with_role({"type": "bullets", "items": [str(i) for i in items if str(i).strip()], "source": source}, role)


def heading(text: str, source: str = "curriculum") -> dict:
    return {"type": "heading", "text": text, "source": source}


def table(header: list[str], rows: list[list], source: str) -> dict:
    return {"type": "table", "header": [str(h) for h in header],
            "rows": [[str(c if c is not None else "") for c in row] for row in rows], "source": source}
