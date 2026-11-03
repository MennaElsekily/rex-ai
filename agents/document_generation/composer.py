"""
Builds the structured content of a document from the approved curriculum
(copied verbatim, source "curriculum"), the narrative parts the model
wrote (source "generated") and the extra information the administrator
supplied (source "admin"). A field no source provides is reported in
content["missing"] and shown in the document as "not specified": it is
never invented.

One small function per section key (templates.py lists which sections a
document type uses).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agents.document_generation import schemas
from agents.document_generation.schemas import bullets, heading, paragraph, table
from agents.document_generation.templates import (
    SECTION_HEADINGS,
    STRINGS,
    DocumentTemplate,
    hours_text,
    level_label,
    type_label,
)
from agents.document_generation.units import (
    PERFORMANCE_LEVELS,
    QUESTION_PLAN,
    TOTAL_POINTS,
    format_points,
    level_points,
    rubric_specs,
    rubric_weights,
)

CURRICULUM, GENERATED, ADMIN, MISSING = "curriculum", "generated", "admin", "missing"


class _Context:
    def __init__(self, record: dict[str, Any], language: str, parts: Any, extra: dict[str, str],
                 localize_level: bool = True, legacy: bool = False) -> None:
        self.record = record
        self.localize_level = localize_level
        self.legacy = legacy   # the wording of the new document types before the layout fixes (see compose)
        self.c: dict[str, Any] = record["content"]
        self.language = language
        self.s = STRINGS[language]
        self.parts = parts
        self.extra = extra
        self.missing: list[dict[str, str]] = []
        self.slide = 0   # the running slide number of a presentation outline

    def outcome_text(self, outcome_ids: list[str]) -> list[str]:
        by_id = {o.get("id"): o.get("statement") for o in self.c.get("learning_outcomes") or []}
        return [f"{i}: {by_id[i]}" for i in outcome_ids if i in by_id]


def _modules(ctx: _Context) -> list[dict[str, Any]]:
    return ctx.c.get("modules") or []


def _join(ctx: _Context, items: list[str]) -> str:
    return ("، " if ctx.language == "ar" else ", ").join(items)


def _assessment_lines(ctx: _Context) -> list[str]:
    return [f"{a.get('id')} ({a.get('type')}): {a.get('description')} [{', '.join(a.get('outcome_ids') or [])}]"
            for a in ctx.c.get("assessments") or []]


# -- sections ------------------------------------------------------------

def _overview(ctx): return [paragraph(ctx.c["overview"], CURRICULUM)]
def _summary(ctx): return [paragraph(ctx.parts.summary, GENERATED)]
def _objectives(ctx): return [bullets(ctx.c["objectives"], CURRICULUM)]
def _audience(ctx):
    audience = ctx.c.get("target_audience")
    return [paragraph(audience, CURRICULUM)] if audience else _missing(ctx, "audience", "target_audience")


def _prerequisites(ctx):
    items = ctx.c.get("prerequisites") or []
    return [bullets(items, CURRICULUM)] if items else [paragraph(ctx.s["none_listed"], CURRICULUM)]


def _outcomes(ctx):
    return [bullets([f"{o.get('id')}: {o.get('statement')}" for o in ctx.c["learning_outcomes"]], CURRICULUM)]


def _modules_table(ctx):
    s = ctx.s
    rows = [[f"{m.get('id')} {m.get('title')}", _join(ctx, m.get("topics") or []), m.get("hours")] for m in _modules(ctx)]
    return [table([s["module"], s["topics"], s["hours"]], rows, CURRICULUM)]


def _hours(ctx): return [paragraph(ctx.s["hours_total"].format(hours=ctx.c["total_hours"]), CURRICULUM)]


def _missing(ctx, section: str, field: str) -> list[dict]:
    ctx.missing.append({"section": section, "field": field})
    return [paragraph(ctx.s["not_specified"], MISSING)]


def _format(ctx):
    value = ctx.extra.get("training_format")
    return [paragraph(value, ADMIN)] if value else _missing(ctx, "format", "training_format")


def _assessment(ctx):
    blocks = [bullets(_assessment_lines(ctx), CURRICULUM)] if ctx.c.get("assessments") else \
        [paragraph(ctx.s["none_listed"], CURRICULUM)]
    approach = getattr(ctx.parts, "assessment_approach", None)
    return ([paragraph(approach, GENERATED)] if approach else []) + blocks


def _proposal_summary(ctx):
    blocks = []
    if ctx.extra.get("organization"):
        blocks.append(paragraph(ctx.s["organization"].format(value=ctx.extra["organization"]), ADMIN))
    return blocks + [paragraph(ctx.parts.executive_summary, GENERATED)]


def _background(ctx):
    blocks = [paragraph(ctx.parts.background_need, GENERATED)]
    if ctx.c.get("rationale"):
        blocks.append(paragraph(ctx.c["rationale"], CURRICULUM))
    return blocks


def _methodology(ctx): return [paragraph(ctx.parts.methodology, GENERATED)]


def _schedule(ctx):
    s = ctx.s
    rows = [[f"{m.get('id')} {m.get('title')}", m.get("hours")] for m in _modules(ctx)]
    rows.append([s["total"], ctx.c["total_hours"]])
    blocks = [table([s["module"], s["hours"]], rows, CURRICULUM)]
    if ctx.extra.get("schedule_notes"):
        blocks.append(paragraph(s["schedule_notes"].format(value=ctx.extra["schedule_notes"]), ADMIN))
    return blocks


def _activities(ctx):
    lines = [f"{m.get('id')} – {a.get('type')}: {a.get('description')}"
             for m in _modules(ctx) for a in m.get("activities") or []]
    return [bullets(lines, CURRICULUM)] if lines else [paragraph(ctx.s["none_listed"], CURRICULUM)]


def _deliverables(ctx):
    blocks = [bullets(ctx.parts.expected_deliverables, GENERATED)]
    capstone = ctx.c.get("capstone")
    if isinstance(capstone, dict) and capstone.get("deliverables"):
        blocks.append(heading(ctx.s["deliverables"]))
        blocks.append(bullets(capstone["deliverables"], CURRICULUM))
    return blocks


def _guide_introduction(ctx): return [paragraph(ctx.parts.introduction, GENERATED), paragraph(ctx.c["overview"], CURRICULUM)]


def _module_guides(ctx):
    s = ctx.s
    guides = {g.module_id: g for g in ctx.parts.modules}
    blocks: list[dict] = []
    for module in _modules(ctx):
        hours = f"{module.get('hours')} {s['hours']}" if ctx.legacy else hours_text(module.get("hours"), ctx.language)
        blocks.append(heading(f"{module.get('id')} {module.get('title')} ({hours})"))
        if module.get("description"):
            blocks.append(paragraph(module["description"], CURRICULUM))
        if module.get("topics"):
            blocks.append(bullets(module["topics"], CURRICULUM))
        activities = [f"{a.get('type')}: {a.get('description')}" for a in module.get("activities") or []]
        if activities:
            blocks += [heading(s["activities"]), bullets(activities, CURRICULUM)]
        guide = guides.get(module.get("id"))
        if guide:
            blocks += [heading(s["key_concepts"]), bullets(guide.key_concepts, GENERATED),
                       heading(s["exercise"]), paragraph(guide.exercise, GENERATED)]
    return blocks


def _assessment_guide(ctx):
    blocks = [paragraph(ctx.parts.assessment_instructions, GENERATED)]
    blocks.append(bullets(_assessment_lines(ctx), CURRICULUM) if ctx.c.get("assessments")
                  else paragraph(ctx.s["none_listed"], CURRICULUM))
    return blocks


def _capstone(ctx):
    capstone = ctx.c.get("capstone")
    if not isinstance(capstone, dict):
        reason = ctx.c.get("capstone_omitted_reason")
        return [paragraph(ctx.s["no_capstone"] + (f" {reason}" if reason else ""), CURRICULUM)]
    blocks = [paragraph(f"{capstone.get('title')}: {capstone.get('description')}", CURRICULUM)]
    if capstone.get("deliverables"):
        blocks += [heading(ctx.s["deliverables"]), bullets(capstone["deliverables"], CURRICULUM)]
    return blocks


def _references(ctx):
    refs = [f"[{r.get('source_id')}] {r.get('title')} — {r.get('publisher')}. {r.get('url')}"
            for r in ctx.c.get("references") or []]
    return [bullets(refs, CURRICULUM)] if refs else [paragraph(ctx.s["no_references"], CURRICULUM)]


def _capstone_intro(ctx): return [paragraph(ctx.parts.introduction, GENERATED)]
def _capstone_guidance(ctx): return [paragraph(ctx.parts.evaluation_guidance, GENERATED)]


def _capstone_outcomes(ctx):
    capstone = ctx.c.get("capstone") or {}
    lines = ctx.outcome_text(capstone.get("outcome_ids") or [])
    return [bullets(lines, CURRICULUM)] if lines else [paragraph(ctx.s["none_listed"], CURRICULUM)]


def _outcome_matrix(ctx):
    s = ctx.s
    rows = []
    for outcome in ctx.c["learning_outcomes"]:
        oid = outcome.get("id")
        modules = [m.get("id") for m in _modules(ctx) if oid in (m.get("outcome_ids") or [])]
        assessments = [a.get("id") for a in ctx.c.get("assessments") or [] if oid in (a.get("outcome_ids") or [])]
        rows.append([f"{oid}: {outcome.get('statement')}", ", ".join(modules) or "—", ", ".join(assessments) or "—"])
    return [table([s["outcome"], s["modules"], s["assessments"]], rows, CURRICULUM)]


def _traceability(ctx):
    s, record = ctx.s, ctx.record
    sources = [f"{r.get('title')} ({r.get('url')})" for r in ctx.c.get("references") or []]
    level = level_label(ctx.c.get("level"), ctx.language) if ctx.localize_level else ctx.c.get("level")
    blocks = [paragraph(s["trace_curriculum"].format(title=ctx.c["title"], curriculum_id=record["curriculum_id"],
                                                     version=record["version"]), CURRICULUM),
              paragraph(s["trace_level"].format(level=level or "—", hours=ctx.c["total_hours"]), CURRICULUM),
              paragraph(s["trace_sources"].format(sources="; ".join(sources)) if sources else s["trace_none"], CURRICULUM),
              paragraph(s["disclaimer"], CURRICULUM)]
    return blocks


# -- units written by the model, one per module or assessment ---------------------

def _unit(ctx, field: str, key: str, unit_id: str):
    """The model's part for one module (or assessment); the composer never invents one."""
    return next((u for u in getattr(ctx.parts, field) if getattr(u, key) == unit_id), None)


def _bullets_or_none(ctx, items: list[str], source: str = CURRICULUM) -> dict:
    return bullets(items, source) if [i for i in items if str(i).strip()] else paragraph(ctx.s["none_listed"], source)


# Instructor guide

def _instructor_note(ctx):
    return [paragraph(ctx.s["instructor_only_notice_legacy" if ctx.legacy else "instructor_only_notice"], CURRICULUM)]
def _instructor_preparation(ctx): return [paragraph(ctx.parts.preparation, GENERATED)]


def _module_teaching(ctx):
    s = ctx.s
    blocks: list[dict] = []
    for module in _modules(ctx):
        unit = _unit(ctx, "modules", "module_id", module.get("id"))
        hours = f"{module.get('hours')} {s['hours']}" if ctx.legacy else hours_text(module.get("hours"), ctx.language)
        blocks.append(heading(f"{module.get('id')} {module.get('title')} ({hours})"))
        if module.get("description"):
            blocks.append(paragraph(module["description"], CURRICULUM))
        blocks += [heading(s["session_structure"]),
                   paragraph(s["session_text"].format(hours=module.get("hours"), topics=_join(ctx, module.get("topics") or [])),
                             CURRICULUM),
                   heading(s["participant_activities"]),
                   _bullets_or_none(ctx, [f"{a.get('type')}: {a.get('description')}" for a in module.get("activities") or []])]
        if unit is not None:
            blocks += [heading(s["teaching_methods"]), bullets(unit.teaching_methods, GENERATED),
                       heading(s["instructor_activities"]), bullets(unit.instructor_activities, GENERATED),
                       heading(s["instructor_notes"]), paragraph(unit.instructor_notes, GENERATED),
                       heading(s["discussion_questions"]), bullets(unit.discussion_questions, GENERATED)]
    return blocks


def _materials(ctx): return [_bullets_or_none(ctx, ctx.c.get("tools_resources") or [])]


def _assessment_guidance(ctx):
    return [paragraph(ctx.parts.assessment_guidance, GENERATED), _bullets_or_none(ctx, _assessment_lines(ctx))]


# Presentation outline

def _slide(ctx, title: str, purpose: str) -> list[dict]:
    """The start of one slide: its number and title, and the one-line purpose that every slide has."""
    ctx.slide += 1
    return [heading(f"{ctx.s['slide']} {ctx.slide} — {title}"), paragraph(f"{ctx.s['purpose']}: {purpose}", CURRICULUM)]


def _slides_title(ctx):
    return _slide(ctx, ctx.c["title"], ctx.s["slide_purpose_title"]) + [
        paragraph(ctx.c["title"], CURRICULUM),
        paragraph(ctx.s["slide_hours_level"].format(level=level_label(ctx.c.get("level"), ctx.language) or "—",
                                                    hours=ctx.c["total_hours"]), CURRICULUM)]


def _slides_overview(ctx):
    return _slide(ctx, SECTION_HEADINGS["overview"][ctx.language], ctx.s["slide_purpose_overview"]) + _overview(ctx)


def _slides_objectives(ctx):
    return _slide(ctx, SECTION_HEADINGS["objectives"][ctx.language], ctx.s["slide_purpose_objectives"]) + _objectives(ctx)


def _slides_outcomes(ctx):
    return _slide(ctx, SECTION_HEADINGS["outcomes"][ctx.language], ctx.s["slide_purpose_outcomes"]) + _outcomes(ctx)


def _module_slides(ctx):
    s = ctx.s
    blocks: list[dict] = []
    for module in _modules(ctx):
        unit = _unit(ctx, "modules", "module_id", module.get("id"))
        blocks += _slide(ctx, f"{module.get('id')} {module.get('title')}",
                         s["slide_purpose_module"].format(id=module.get("id"), title=module.get("title")))
        blocks += [heading(s["topics"]), _bullets_or_none(ctx, module.get("topics") or [])]
        if unit is not None:
            blocks += [heading(s["key_points"]), bullets(unit.key_points, GENERATED),
                       heading(s["example"]), paragraph(unit.example, GENERATED),
                       heading(s["discussion_prompt"]), paragraph(unit.discussion_prompt, GENERATED),
                       heading(s["visual"]), paragraph(unit.visual, GENERATED)]
        activities = [f"{a.get('type')}: {a.get('description')}" for a in module.get("activities") or []]
        if activities:
            blocks += [heading(s["curriculum_activities"]), bullets(activities, CURRICULUM)]
    return blocks


def _slides_assessment(ctx):
    blocks = _slide(ctx, SECTION_HEADINGS["assessment"][ctx.language], ctx.s["slide_purpose_assessment"])
    blocks.append(_bullets_or_none(ctx, _assessment_lines(ctx)))
    capstone = ctx.c.get("capstone")
    if isinstance(capstone, dict):
        blocks += _slide(ctx, str(capstone.get("title")), ctx.s["slide_purpose_capstone"])
        blocks.append(paragraph(str(capstone.get("description")), CURRICULUM))
        if capstone.get("deliverables"):
            blocks += [heading(ctx.s["deliverables"]), bullets(capstone["deliverables"], CURRICULUM)]
    return blocks


def _slides_closing(ctx):
    return _slide(ctx, SECTION_HEADINGS["slides_closing"][ctx.language], ctx.s["slide_purpose_closing"]) + [
        paragraph(ctx.parts.closing_summary, GENERATED), heading(ctx.s["outcomes_recap"])] + _outcomes(ctx)


# Question bank

def _qb_overview(ctx): return [paragraph(ctx.s["qb_intro"], CURRICULUM)]


def _questions(ctx):
    s = ctx.s
    blocks: list[dict] = []
    number = 0
    for module in _modules(ctx):
        unit = _unit(ctx, "modules", "module_id", module.get("id"))
        blocks.append(heading(f"{module.get('id')} {module.get('title')}"))
        if unit is None:
            continue
        pools = {"mc": list(unit.multiple_choice), "tf": list(unit.true_false), "sa": list(unit.short_answer)}
        outcomes = ", ".join(module.get("outcome_ids") or []) or "—"
        for kind, difficulty in QUESTION_PLAN:
            number += 1
            question = pools[kind].pop(0)
            title = f"{s['question']} {number} · {s['q_' + kind]} · {s['difficulty'][difficulty]} · {s['outcomes_label']}: {outcomes}"
            blocks.append(heading(f"{title} · {s['a_' + kind]}" if ctx.legacy else title))
            if kind == "mc":
                blocks += [paragraph(question.question, GENERATED, "q_stem"), bullets(question.options, GENERATED, "q_options"),
                           heading(s["correct_answer"]),
                           paragraph(question.options[question.correct_index], GENERATED, "q_answer")]
            elif kind == "tf":
                blocks += [paragraph(question.statement, GENERATED, "q_stem"), heading(s["correct_answer"]),
                           paragraph(s["true"] if question.answer else s["false"], GENERATED, "tf_answer")]
            else:
                blocks += [paragraph(question.question, GENERATED, "q_stem"), heading(s["model_answer"]),
                           paragraph(question.model_answer, GENERATED, "q_model_answer")]
            blocks += [heading(s["rationale"]), paragraph(question.rationale, GENERATED)]
    return blocks


# Assessment rubrics

def _rubric_overview(ctx):
    s = ctx.s
    shares = [s["level_share"].format(level=s["level"][name], share=f"{100 * (k + 1) // len(PERFORMANCE_LEVELS)}%")
              for k, name in enumerate(PERFORMANCE_LEVELS)]
    return [paragraph(s["rubric_intro"], CURRICULUM), bullets(shares, CURRICULUM)]


def _rubrics(ctx):
    s = ctx.s
    blocks: list[dict] = []
    for spec in rubric_specs(ctx.c):
        if not spec["criteria"]:
            ctx.missing.append({"section": "rubrics", "field": "assessment_outcomes"})
            continue
        unit = _unit(ctx, "rubrics", "assessment_id", spec["id"])
        by_outcome = {c.outcome_id: c for c in unit.criteria} if unit is not None else {}
        weights = rubric_weights(len(spec["criteria"]))
        names = [s["level"][name] for name in PERFORMANCE_LEVELS]
        rows, totals = [], [0] * len(PERFORMANCE_LEVELS)
        for criterion, weight in zip(spec["criteria"], weights):
            points = level_points(weight)
            totals = [t + p for t, p in zip(totals, points)]
            rows.append([f"{criterion['outcome_id']}: {criterion['statement']}", weight, *[format_points(p) for p in points]])
        rows.append([s["total_possible"], TOTAL_POINTS, *[format_points(t) for t in totals]])
        blocks += [heading(f"{s['rubric']}: {spec['title']}"), heading(s["purpose"]), paragraph(str(spec["description"]), CURRICULUM),
                   heading(s["related_outcomes"]),
                   bullets([f"{c['outcome_id']}: {c['statement']}" for c in spec["criteria"]], CURRICULUM),
                   heading(s["scoring_summary"]), table([s["criterion"], s["weight"], *names], rows, CURRICULUM)]
        for number, (criterion, weight) in enumerate(zip(spec["criteria"], weights), start=1):
            blocks.append(heading(f"{s['criterion']} {number}: {criterion['outcome_id']} — {criterion['statement']} "
                                  f"({weight} {s['points']})"))
            levels = by_outcome.get(criterion["outcome_id"])
            if levels is None:
                continue
            for name, points in zip(PERFORMANCE_LEVELS, level_points(weight)):
                blocks += [heading(f"{s['level'][name]} — {format_points(points)} {s['points']}"),
                           paragraph(getattr(levels, name), GENERATED, "descriptor")]
        if unit is not None:
            blocks += [heading(s["assessment_instructions"]), paragraph(unit.instructions, GENERATED)]
    return blocks


def _scoring_note(ctx): return [paragraph(ctx.s["scoring_note_text"], CURRICULUM)]


BUILDERS: dict[str, Callable[[_Context], list[dict]]] = {
    "overview": _overview, "summary": _summary, "objectives": _objectives, "audience": _audience,
    "prerequisites": _prerequisites, "outcomes": _outcomes, "modules_table": _modules_table, "hours": _hours,
    "format": _format, "assessment": _assessment, "proposal_summary": _proposal_summary,
    "background": _background, "methodology": _methodology, "schedule": _schedule, "activities": _activities,
    "deliverables": _deliverables, "guide_introduction": _guide_introduction, "module_guides": _module_guides,
    "assessment_guide": _assessment_guide, "capstone": _capstone, "references": _references,
    "capstone_intro": _capstone_intro, "capstone_outcomes": _capstone_outcomes,
    "capstone_guidance": _capstone_guidance, "outcome_matrix": _outcome_matrix, "traceability": _traceability,
    "instructor_note": _instructor_note, "instructor_preparation": _instructor_preparation,
    "teaching_objectives": _objectives, "module_teaching": _module_teaching, "materials": _materials,
    "assessment_guidance": _assessment_guidance, "slides_title": _slides_title, "slides_overview": _slides_overview,
    "slides_objectives": _slides_objectives, "slides_outcomes": _slides_outcomes, "module_slides": _module_slides,
    "slides_assessment": _slides_assessment, "slides_closing": _slides_closing, "qb_overview": _qb_overview,
    "questions": _questions, "rubric_overview": _rubric_overview, "rubrics": _rubrics, "scoring_note": _scoring_note,
}


def compose(template: DocumentTemplate, record: dict[str, Any], language: str, parts: Any = None,
            extra: Optional[dict[str, str]] = None, localize_level: bool = True, legacy: bool = False) -> dict[str, Any]:
    """
    The structured content of one document (see schemas.py for the shape).
    localize_level=False reproduces documents made before Arabic documents showed the level in Arabic, so the
    validator can still recognise their (unchanged) curriculum text. legacy=True does the same for the instructor
    notice, the module-hours heading and the question headings as they were worded before the layout fixes: documents
    made then keep validating (and stay editable) against their own wording; new documents use the new wording.
    """
    ctx = _Context(record, language, parts, extra or {}, localize_level, legacy)
    content = ctx.c
    sections = []
    for key in template.sections:
        blocks = BUILDERS[key](ctx)
        sections.append({"key": key, "heading": SECTION_HEADINGS[key][language], "blocks": blocks})
    return {
        "document_type": template.doc_type,
        "title": f"{type_label(template.doc_type, language)}: {content['title']}",
        "course_title": content["title"],
        "language": language,
        "curriculum_id": record["curriculum_id"],
        "curriculum_version": record["version"],
        "level": content.get("level"),
        "total_hours": content.get("total_hours"),
        "sections": sections,
        "missing": ctx.missing,
        "extra": dict(extra or {}),
    }


__all__ = ["BUILDERS", "compose", "schemas"]
