"""
Revisions of a stored curriculum (replaces the old refinement.py).

``detect_changes`` turns a message into a list of changes, reusing
request_parser.detect_refinements for hours, audience, "add a module" and
"add a capstone", and adding level, language, module removal, capstone
removal and free-form instructions. ``apply_change`` applies the changes
that need no language model; the graph asks the model for the others
(new module text, capstone text, or a regenerated curriculum that follows
an instruction). Every revision is saved as a new version.

Kinds: set_hours, set_audience, remove_module, remove_capstone (here);
add_module, add_capstone, set_level, set_language, instruction (graph).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Optional

from agents.curriculum_research.request_parser import (
    detect_refinements,
    looks_like_edit_attempt,
)
from agents.curriculum_research.understanding import find_level, output_language
from agents.curriculum_research.validation import rebalance_hours
from agents.text_utils import normalize_for_matching

MAX_HOURS = 1000
DETERMINISTIC = ("set_hours", "set_audience", "remove_module", "remove_capstone")
REGENERATE = ("set_level", "set_language", "instruction")

_REMOVE_MODULE = re.compile(
    r"\b(?:remove|delete|drop|cut)\s+(?:the\s+)?(?:module|unit)\s*(?:m|number\s+|no\.?\s*)?(\d{1,2})\b"
    r"|\b(?:remove|delete|drop|cut)\s+m(\d{1,2})\b"
    r"|(?:احذف|ازل|امسح|شيل)\s+(?:ال)?(?:وحده|موديول)\s*(?:رقم\s*)?(\d{1,2})"
)
_REMOVE_CAPSTONE = re.compile(
    r"\b(?:remove|delete|drop)\s+(?:the\s+)?capstone\b|\bno\s+capstone\b"
    r"|(?:احذف|ازل|امسح|شيل)\s+(?:ال)?مشروع\s+(?:ال)?(?:ختامي|نهائي)|بدون\s+(?:ال)?مشروع\s+(?:ال)?ختامي"
)
_LEVEL_CUE = re.compile(r"\blevel\b|\bmake\s+it\s+(?:more\s+)?(?:beginner|intermediate|advanced)\b|المستوي|مستوي|اجعله|اجعلها")
_LANGUAGE_CUE = re.compile(r"\btranslate\b|\bswitch\b|\bconvert\b|\brewrite\b|\bin\s+(?:arabic|english)\b|ترجم|حول|"
                           r"اكتب|اجعل|بالعربي|بالانجليزي|باللغه")


class RevisionError(ValueError):
    """A change that cannot be applied (the message says why)."""


def detect_changes(text: str, curriculum: dict[str, Any]) -> list[dict[str, Any]]:
    """The changes a message asks for on ``curriculum`` ([] when it is not an edit)."""
    norm = normalize_for_matching(text)
    changes: list[dict[str, Any]] = []

    match = _REMOVE_MODULE.search(norm)
    if match:
        number = next(g for g in match.groups() if g)
        changes.append({"kind": "remove_module", "module_id": f"M{int(number)}"})
    elif _REMOVE_CAPSTONE.search(norm):
        changes.append({"kind": "remove_capstone"})

    language = output_language(text, curriculum.get("language", "en"))
    if language != curriculum.get("language") and _LANGUAGE_CUE.search(norm):
        changes.append({"kind": "set_language", "value": language})

    level = find_level(text)
    if level and level != curriculum.get("level") and _LEVEL_CUE.search(norm):
        changes.append({"kind": "set_level", "value": level})

    if not any(c["kind"] in ("remove_module", "remove_capstone") for c in changes):
        for refinement in detect_refinements(text):
            if refinement.kind == "set_hours":
                changes.append({"kind": "set_hours", "value": int(refinement.value)})
            elif refinement.kind == "set_audience":
                changes.append({"kind": "set_audience", "value": str(refinement.value)})
            elif refinement.kind == "add_module":
                changes.append({"kind": "add_module", "value": str(refinement.value)})
            elif refinement.kind == "add_capstone":
                changes.append({"kind": "add_capstone"} if not curriculum.get("capstone") else
                               {"kind": "no_change", "value": "capstone_exists"})
    else:
        hours = [r for r in detect_refinements(text) if r.kind == "set_hours"]
        changes += [{"kind": "set_hours", "value": int(r.value)} for r in hours]

    if not changes and looks_like_edit_attempt(text):
        # Anything else that reads as an edit ("make the outcomes more
        # practical"): the model regenerates following the instruction.
        changes.append({"kind": "instruction", "value": " ".join(text.split())[:500]})
    # The level and language changes regenerate the whole curriculum; an
    # audience change then becomes part of the instruction.
    return changes


def _drop_refs(curriculum: dict[str, Any]) -> None:
    """Remove outcome references that point to nothing (after a removal)."""
    ids = {o.get("id") for o in curriculum.get("learning_outcomes") or []}
    for part in [*curriculum.get("modules", []), *curriculum.get("assessments", []),
                 *([curriculum["capstone"]] if isinstance(curriculum.get("capstone"), dict) else [])]:
        part["outcome_ids"] = [ref for ref in part.get("outcome_ids") or [] if ref in ids]


def set_total_hours(curriculum: dict[str, Any], hours: int) -> None:
    modules = curriculum.get("modules") or []
    if not 1 <= hours <= MAX_HOURS:
        raise RevisionError(f"The total must be between 1 and {MAX_HOURS} hours.")
    if hours < len(modules):
        raise RevisionError(f"{hours} hours cannot cover {len(modules)} modules (at least one hour each).")
    for module, share in zip(modules, rebalance_hours(modules, hours)):
        module["hours"] = share
    curriculum["total_hours"] = hours


def apply_change(curriculum: dict[str, Any], change: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Apply one deterministic change; returns (new curriculum, English summary)."""
    revised = copy.deepcopy(curriculum)
    kind = change["kind"]
    if kind == "set_hours":
        before = revised.get("total_hours")
        set_total_hours(revised, change["value"])
        return revised, f"Total hours changed from {before} to {change['value']}; module hours rescaled proportionally."
    if kind == "set_audience":
        revised["target_audience"] = change["value"]
        return revised, f"Target audience changed to {change['value']}."
    if kind == "remove_module":
        modules = revised.get("modules") or []
        if change["module_id"] not in {m.get("id") for m in modules}:
            raise RevisionError(f"There is no module {change['module_id']}.")
        if len(modules) == 1:
            raise RevisionError("The only module cannot be removed.")
        removed = next(m for m in modules if m.get("id") == change["module_id"])
        revised["modules"] = [m for m in modules if m.get("id") != change["module_id"]]
        for index, module in enumerate(revised["modules"], start=1):
            module["id"] = f"M{index}"
        set_total_hours(revised, revised["total_hours"])
        _drop_refs(revised)
        return revised, (f"Module {change['module_id']} ({removed.get('title')}) removed; its hours were spread "
                         "over the remaining modules.")
    if kind == "remove_capstone":
        revised["capstone"] = None
        revised["capstone_omitted_reason"] = "Removed at the administrator's request."
        return revised, "Capstone project removed."
    raise RevisionError(f"{kind} is not a deterministic change.")


def add_module(curriculum: dict[str, Any], module: dict[str, Any], hours: Optional[int] = None) -> dict[str, Any]:
    """Append a module written by the model and keep the total hours (or set ``hours``)."""
    revised = copy.deepcopy(curriculum)
    modules = revised.setdefault("modules", [])
    average = max(1, round(sum(m.get("hours", 0) for m in modules) / len(modules))) if modules else 1
    modules.append({"id": f"M{len(modules) + 1}", "hours": average, **module})
    set_total_hours(revised, hours or revised["total_hours"])
    _drop_refs(revised)
    return revised
