"""
Writes the narrative parts of a document with the language model and
checks them against the approved curriculum (validation.check_parts).

One controlled repair attempt: if the first answer has problems (wrong
module ids, an unsupported hour figure, an accreditation claim) the model
is asked once more with the problems listed. Parts that still fail are
returned with their problems and nothing is saved by the caller.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agents.curriculum_research.llm import structured
from agents.document_generation import prompts, schemas, units, validation
from agents.document_generation.templates import DocumentTemplate
from agents.document_generation.validation import check_parts, check_unit, check_unit_parts_frame, issue


def generate_parts(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str,
                   problems: list[dict[str, str]] | None = None) -> Any:
    schema = template.parts_schema
    prompt = prompts.PROMPTS[schema.__name__]
    problems_block = ""
    if problems:
        problems_block = ("\nYour previous answer had these problems. Fix them and answer again:\n"
                          + "\n".join(f"- {p['message']}" for p in problems))
    messages = prompt.format_messages(
        language_name=prompts.LANGUAGE_NAMES[language], curriculum_block=prompts.curriculum_block(curriculum),
        problems_block=problems_block)
    return structured(llm, schema, messages, f"{template.doc_type} writing")


def _tell(progress: Optional[Callable[[int, int, str], None]], done: int, total: int, label: str) -> None:
    """Tell the caller how far the writing has got (the admin page shows it). A failing callback never stops the run."""
    if progress is not None:
        try:
            progress(done, total, label)
        except Exception:
            pass


def generate_with_repair(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str,
                         progress: Optional[Callable[[int, int, str], None]] = None) -> tuple[Any, list[dict[str, str]]]:
    """
    (parts, problems): problems is empty when the parts passed (possibly after the repair attempt). Documents written
    per module or assessment make one model call per unit (see _generate_in_units); the others make one call.
    progress(done, total, label) is told after each step.
    """
    if template.unit_schema is not None:
        return _generate_in_units(llm, template, curriculum, language, progress)
    _tell(progress, 0, 1, template.label[language])
    parts = generate_parts(llm, template, curriculum, language)
    problems = check_parts(parts, template, curriculum)
    if problems:
        parts = generate_parts(llm, template, curriculum, language, problems)
        problems = check_parts(parts, template, curriculum)
    _tell(progress, 1, 1, template.label[language])
    return parts, problems


def _problem_block(problems: Optional[list[dict[str, str]]]) -> str:
    if not problems:
        return ""
    return ("\nYour previous answer had these problems. Fix them and answer again:\n"
            + "\n".join(f"- {p['message']}" for p in problems))


def generate_unit(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str, unit_id: str,
                  problems: Optional[list[dict[str, str]]] = None) -> Any:
    """One model call for one module (or one assessment's rubric)."""
    schema = template.unit_schema
    messages = prompts.PROMPTS[schema.__name__].format_messages(
        language_name=prompts.LANGUAGE_NAMES[language], curriculum_block=prompts.curriculum_block(curriculum),
        unit_block=prompts.unit_block(template.unit_field, curriculum, unit_id), problems_block=_problem_block(problems))
    return structured(llm, schema, messages, f"{template.doc_type} {unit_id}")


_ITEM_SCHEMAS = {"multiple_choice": schemas.McQuestion, "true_false": schemas.TfQuestion, "short_answer": schemas.SaQuestion}


def _item(unit: Any, target: str) -> Any:
    """The question a target such as "multiple_choice:1" names, or None."""
    kind, _, index = target.partition(":")
    items = getattr(unit, kind, None) if kind in _ITEM_SCHEMAS else None
    return items[int(index)] if items is not None and index.isdigit() and int(index) < len(items) else None


def _as_json(item: Any) -> str:
    return json.dumps(item.model_dump(), ensure_ascii=False, indent=1) if item is not None else "(none)"


def _problem_lines(problems: list[dict[str, str]]) -> str:
    return "\n".join(f"- {p['message']}" for p in problems) or "- (none)"


def _targeted_repair(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str, unit_id: str,
                     first: Any, first_problems: list[dict[str, str]], current: Any, current_problems: list[dict[str, str]]) -> Any:
    """
    The second repair of a unit (question bank): only the questions the checks still flag are rewritten, one small model
    call each, and put into the current answer; every other question is kept exactly as it is. A problem that is not
    about one question (nothing to point at) falls back to asking for the module again with the problems listed. The
    caller checks the result in full, so nothing invalid is accepted here.
    """
    targets = list(dict.fromkeys(p.get("target", "") for p in current_problems))
    if "" in targets or any(_item(current, t) is None for t in targets):
        return generate_unit(llm, template, curriculum, language, unit_id, current_problems)
    data = current.model_dump()
    for target in targets:
        kind, _, index = target.partition(":")
        remaining = [p for p in current_problems if p.get("target") == target]
        earlier = [p for p in first_problems if p.get("target") == target]
        others = [text for other_target, text in zip(validation._stem_targets(current), validation._unit_stems(current))
                  if other_target != target]
        changes = "\n".join(dict.fromkeys(prompts.QUESTION_CHANGES.get(p["code"], "Fix the problem listed above.")
                                          for p in remaining))
        messages = prompts.QUESTION_FIX.format_messages(
            language_name=prompts.LANGUAGE_NAMES[language], curriculum_block=prompts.curriculum_block(curriculum),
            unit_block=prompts.unit_block(template.unit_field, curriculum, unit_id), kind_rules=prompts.QUESTION_KIND_RULES[kind],
            others="\n".join(f"- {t}" for t in others) or "- (none)", original=_as_json(_item(first, target)),
            original_problems=_problem_lines(earlier), current=_as_json(_item(current, target)),
            current_problems=_problem_lines(remaining), changes=changes)
        data[kind][int(index)] = structured(llm, _ITEM_SCHEMAS[kind], messages, f"{template.doc_type} {unit_id} {target}").model_dump()
    return template.unit_schema.model_validate(data)


def _generate_frame(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str,
                    problems: Optional[list[dict[str, str]]] = None) -> Any:
    """The text written once for the whole document (instructor preparation, closing summary)."""
    schema = template.frame_schema
    messages = prompts.PROMPTS[schema.__name__].format_messages(
        language_name=prompts.LANGUAGE_NAMES[language], curriculum_block=prompts.curriculum_block(curriculum),
        problems_block=_problem_block(problems))
    return structured(llm, schema, messages, f"{template.doc_type} overview")


def _generate_in_units(llm: Any, template: DocumentTemplate, curriculum: dict[str, Any], language: str,
                       progress: Optional[Callable[[int, int, str], None]]) -> tuple[Any, list[dict[str, str]]]:
    """
    The document-wide text (if any), then one unit at a time. Each unit is checked as soon as it is written and gets
    one repair attempt; a unit that still fails ends the run (nothing is saved). The finished set is checked as a
    whole before it is returned. This is more reliable for a small model than one very long answer, and a failure
    names the module or assessment it happened in.
    """
    ids = units.unit_ids(template.unit_field, curriculum)
    total = len(ids) + (1 if template.frame_schema is not None else 0)
    done, frame_data = 0, {}
    if template.frame_schema is not None:
        _tell(progress, done, total, template.label[language])
        frame = _generate_frame(llm, template, curriculum, language)
        problems = check_unit_parts_frame(frame, template, curriculum)
        if problems:
            frame = _generate_frame(llm, template, curriculum, language, problems)
            problems = check_unit_parts_frame(frame, template, curriculum)
            if problems:
                return None, problems
        frame_data, done = frame.model_dump(), done + 1
    accepted: list[Any] = []
    for unit_id in ids:
        _tell(progress, done, total, unit_id)
        unit = generate_unit(llm, template, curriculum, language, unit_id)
        problems = check_unit(unit, template, curriculum, tuple(accepted))
        if problems:
            first, first_problems = unit, problems
            unit = generate_unit(llm, template, curriculum, language, unit_id, problems)
            problems = check_unit(unit, template, curriculum, tuple(accepted))
            if problems and template.targeted_repair:
                unit = _targeted_repair(llm, template, curriculum, language, unit_id, first, first_problems, unit, problems)
                problems = check_unit(unit, template, curriculum, tuple(accepted))
            if problems:
                return None, problems
        accepted.append(unit)
        done += 1
    _tell(progress, done, total, "")
    try:
        parts = template.parts_schema.model_validate({**frame_data, template.unit_field: [u.model_dump() for u in accepted]})
    except ValueError as exc:
        return None, [issue("schema", f"The assembled document is not valid: {str(exc)[:200]}", template.unit_field)]
    return parts, check_parts(parts, template, curriculum)
