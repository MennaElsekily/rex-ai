"""
Data shapes for Agent 3.

- The *LLM step* models (CurriculumOutline, CurriculumDetails,
  SuggestionList, AlignmentJudgement) are what the language model returns
  through LangChain structured output. Each is small on purpose: a 3-4B
  local model is far more reliable with short, flat answers.
- The assembled curriculum is a plain dict (JSON-ready), built by
  ``assemble_curriculum`` from the two generation steps plus what the
  tools supplied (academy evidence, research sources). It is what the
  validator checks, the store saves and the PDF renders.

Every claim has an origin: academy data (course id + file), research
(source id) or "proposed" (written by the model). The model never writes
references itself; it can only cite source ids the tools returned.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

LEVELS = ("Beginner", "Intermediate", "Advanced")
ALIGNMENT_STATUSES = ("covered", "partially_covered", "not_covered", "unable_to_determine")

# Upper bounds for the model's answers. Ollama enforces maxLength/maxItems
# while decoding, so a run-on description or list (seen with qwen2.5:3b in
# Arabic: one assessment description filled the whole token budget) is cut
# instead of breaking the JSON. The prompts ask for much shorter text.
Id = Annotated[str, Field(max_length=12)]
Short = Annotated[str, Field(max_length=200)]
Text = Annotated[str, Field(max_length=700)]


# ---------------------------------------------------------------------
# LLM step 1 -- outline
#
# Fields the curriculum needs are REQUIRED (no defaults) and lists carry
# minimum sizes: the JSON schema sent to Ollama then forces the model to
# write them. With defaults, a 3B model silently left out whole sections
# (learning outcomes, objectives, capstone) -- measured in the §17 runs.
# ---------------------------------------------------------------------

class OutcomeItem(BaseModel):
    id: Id = Field(description='Outcome id: "LO1", "LO2", ...')
    statement: Short = Field(description="A measurable learning outcome starting with an action verb.")


class ModulePlan(BaseModel):
    id: Id = Field(description='Module id: "M1", "M2", ...')
    title: Short
    hours: int = Field(description="Whole training hours for this module (more than zero).")
    topics: list[Short] = Field(min_length=1, max_length=8, description="The topics this module covers.")


class CurriculumOutline(BaseModel):
    title: Short
    overview: Text = Field(description="Two to four sentences describing the course.")
    rationale: Text = Field(description="Why this course matters for the audience; cite source ids if used.")
    objectives: list[Short] = Field(min_length=2, max_length=8, description="Three to six course objectives.")
    target_audience: Short
    prerequisites: list[Short] = Field(max_length=8, description="Prior knowledge needed; an empty list if none.")
    level: Id = Field(description="Beginner, Intermediate or Advanced.")
    learning_outcomes: list[OutcomeItem] = Field(min_length=3, max_length=8, description="Four to eight measurable outcomes.")
    modules: list[ModulePlan] = Field(min_length=1, max_length=12, description="Modules sized for the total hours.")


# ---------------------------------------------------------------------
# LLM step 2 -- details
# ---------------------------------------------------------------------

class Activity(BaseModel):
    type: Annotated[str, Field(max_length=40)] = Field(description='One or two words: "lab", "exercise", "case study".')
    description: Text


class ModuleDetail(BaseModel):
    module_id: Id
    description: Text
    activities: list[Activity] = Field(min_length=1, max_length=3)
    outcome_ids: list[Id] = Field(min_length=1, max_length=8, description="Outcome ids this module develops.")


class AssessmentItem(BaseModel):
    id: Id = Field(description='Assessment id: "A1", "A2", ...')
    type: Annotated[str, Field(max_length=40)] = Field(description='One or two words: "quiz", "project".')
    description: Text
    outcome_ids: list[Id] = Field(min_length=1, max_length=10)


class CapstoneItem(BaseModel):
    title: Short
    description: Text
    outcome_ids: list[Id] = Field(min_length=1, max_length=10)
    deliverables: list[Short] = Field(min_length=1, max_length=6)


class CompetencyItem(BaseModel):
    name: Short
    source_ids: list[Annotated[str, Field(max_length=20)]] = Field(max_length=5, description="Ids of the research sources supporting it; empty if none.")


class CurriculumDetails(BaseModel):
    """Steps 2 and 3 joined (built by the graph, not asked of the model)."""
    modules: list[ModuleDetail] = Field(default_factory=list)
    assessments: list[AssessmentItem] = Field(default_factory=list)
    capstone: Optional[CapstoneItem] = None
    capstone_omitted_reason: Optional[str] = None
    tools_resources: list[str] = Field(default_factory=list)
    competencies: list[CompetencyItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# The graph asks for CurriculumDetails in two smaller calls (step 2:
# module details; step 3: everything else) and joins them.

class ModuleDetailList(BaseModel):
    modules: list[ModuleDetail] = Field(min_length=1, max_length=12, description="One entry per module id.")


class CurriculumExtras(BaseModel):
    assessments: list[AssessmentItem] = Field(min_length=1, max_length=6)
    capstone: Optional[CapstoneItem] = Field(description="The capstone project; null only if none was requested.")
    capstone_omitted_reason: Optional[Text] = Field(None, description="Only when there is no capstone: why.")
    tools_resources: list[Short] = Field(min_length=1, max_length=12)
    competencies: list[CompetencyItem] = Field(min_length=1, max_length=10)
    assumptions: list[Short] = Field(max_length=6)
    open_questions: list[Short] = Field(max_length=5)


# ---------------------------------------------------------------------
# Request understanding fallback, repair and revisions
# ---------------------------------------------------------------------

class RequestSlots(BaseModel):
    """What a curriculum request asks for, when the rules found no topic."""
    topic: Optional[str] = Field(None, description="The subject of the course, in the user's words.")
    total_hours: Optional[int] = Field(None, description="Total training hours, only if the user gave them.")
    level: Optional[str] = Field(None, description="Beginner, Intermediate or Advanced, only if stated.")
    target_audience: Optional[str] = Field(None, description="Who the course is for, only if stated.")


class ModuleLink(BaseModel):
    module_id: str
    outcome_ids: list[str] = Field(default_factory=list)


class RepairPatch(BaseModel):
    """Corrections for the listed validation errors; an empty field means no change."""
    learning_outcomes: list[OutcomeItem] = Field(default_factory=list, description="All outcomes, rewritten if needed.")
    module_links: list[ModuleLink] = Field(default_factory=list, description="Outcome ids per module.")
    assessments: list[AssessmentItem] = Field(default_factory=list, description="All assessments, if they change.")
    capstone: Optional[CapstoneItem] = None
    tools_resources: list[str] = Field(default_factory=list)
    competencies: list[CompetencyItem] = Field(default_factory=list)


class NewModule(BaseModel):
    """One module to add to an existing curriculum."""
    title: Short
    topics: list[Short] = Field(min_length=1, max_length=8)
    description: Text
    activities: list[Activity] = Field(min_length=1, max_length=3)
    outcome_ids: list[Id] = Field(min_length=1, max_length=8, description="Existing outcome ids this module develops.")


# ---------------------------------------------------------------------
# Workflow B -- suggestions
# ---------------------------------------------------------------------

class Suggestion(BaseModel):
    title: Short
    department: str = Field(description="An existing department name, or a proposed new one.")
    level: str
    target_audience: str
    prerequisite_course_ids: list[str] = Field(description="Existing course ids only; empty if none.")
    outcomes: list[Short] = Field(min_length=1, max_length=6)
    proposed_hours: int
    modules_outline: list[Short] = Field(min_length=1, max_length=10)
    rationale: Text
    gap_ids: list[str] = Field(description="Ids of the catalogue gaps this answers; empty if none.")
    source_ids: list[str] = Field(description="Ids of research sources; empty if none.")
    confidence: Literal["low", "medium", "high"]
    assumptions: list[str] = Field(default_factory=list)


class SuggestionList(BaseModel):
    suggestions: list[Suggestion] = Field(max_length=5, description="At most five suggestions.")


# ---------------------------------------------------------------------
# Framework alignment
# ---------------------------------------------------------------------

class AlignmentItem(BaseModel):
    requirement_id: str
    status: Literal["covered", "partially_covered", "not_covered", "unable_to_determine"]
    element_ids: list[Id] = Field(max_length=8, description="Curriculum outcome/module/assessment ids used as evidence.")
    explanation: Text = Field(description="One sentence: why this status.")
    recommendation: Optional[Text] = Field(None, description="For partially or not covered: what to add.")


class AlignmentJudgement(BaseModel):
    items: list[AlignmentItem] = Field(max_length=20, description="One judgement per requirement listed.")


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------

def assemble_curriculum(
    outline: CurriculumOutline,
    details: CurriculumDetails,
    *,
    language: str,
    total_hours: int,
    sources: list[dict[str, Any]],
    academy_evidence: list[dict[str, Any]],
    data_notice: Optional[str],
    extra_assumptions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """The full curriculum dict from the two generation steps and the tool evidence."""
    detail_by_id = {d.module_id: d for d in details.modules}
    modules = []
    for plan in outline.modules:
        detail = detail_by_id.get(plan.id)
        modules.append({
            "id": plan.id,
            "title": plan.title,
            "hours": plan.hours,
            "topics": list(plan.topics),
            "description": detail.description if detail else "",
            "activities": [a.model_dump() for a in detail.activities] if detail else [],
            "outcome_ids": list(detail.outcome_ids) if detail else [],
        })
    cited = {sid for c in details.competencies for sid in c.source_ids}
    references = [s for s in sources if s.get("source_id") in cited or s.get("source_id") in outline.rationale]
    return {
        "title": outline.title,
        "language": language,
        "overview": outline.overview,
        "rationale": outline.rationale,
        "objectives": list(outline.objectives),
        "target_audience": outline.target_audience,
        "prerequisites": list(outline.prerequisites),
        "level": outline.level,
        "total_hours": total_hours,
        "learning_outcomes": [o.model_dump() for o in outline.learning_outcomes],
        "modules": modules,
        "assessments": [a.model_dump() for a in details.assessments],
        "capstone": details.capstone.model_dump() if details.capstone else None,
        "capstone_omitted_reason": details.capstone_omitted_reason,
        "tools_resources": list(details.tools_resources),
        "competencies": [c.model_dump() for c in details.competencies],
        "references": references,
        "academy_evidence": list(academy_evidence),
        "assumptions": [*(extra_assumptions or []), *details.assumptions],
        "open_questions": list(details.open_questions),
        "alignment": None,
        "data_notice": data_notice,
    }
