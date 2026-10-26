"""
Apply refinements (change hours, add a module or capstone, change the
audience) to a generated curriculum draft.

Works on a deep copy and either applies every requested change or
raises RefinementError, so a failed request never leaves a draft
half-edited. Returns structured change records; the agent turns them
into English or Arabic text.
"""

from __future__ import annotations

import copy
from typing import Any

from agents.curriculum_research.curriculum_builder import CurriculumBuilder
from agents.curriculum_research.request_parser import Refinement

MAX_TRAINING_HOURS = 1000
CAPSTONE_TITLE = "Capstone Project"


class RefinementError(ValueError):
    """A refinement could not be applied; `code` says why."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def _redistribute(draft: dict[str, Any], total_hours: int) -> None:
    modules = draft["modules"]
    shares = CurriculumBuilder.distribute_hours(total_hours, len(modules))

    for module, hours in zip(modules, shares):
        module["hours"] = hours

    draft["training_hours"] = total_hours


def _renumber(draft: dict[str, Any]) -> None:
    for number, module in enumerate(draft["modules"], start=1):
        module["module_number"] = number


def _check_hours(draft: dict[str, Any], hours: int, module_count: int) -> None:
    if hours > MAX_TRAINING_HOURS:
        raise RefinementError("hours_too_high", maximum=MAX_TRAINING_HOURS)

    if hours < module_count:
        raise RefinementError("hours_too_low", minimum=module_count)


def _set_hours(draft: dict[str, Any], hours: int) -> dict[str, Any]:
    _check_hours(draft, hours, len(draft["modules"]))
    old = draft.get("training_hours")

    if old == hours:
        return {"type": "hours_unchanged", "hours": hours}

    _redistribute(draft, hours)
    return {"type": "hours_changed", "old": old, "new": hours}


def _append_module(
    draft: dict[str, Any], module: dict[str, Any]
) -> None:
    total = int(draft["training_hours"])
    new_count = len(draft["modules"]) + 1

    if total < new_count:
        raise RefinementError("hours_too_low_for_module", minimum=new_count)

    draft["modules"].append(module)
    _renumber(draft)
    _redistribute(draft, total)


def _add_module(draft: dict[str, Any], title: str) -> dict[str, Any]:
    wanted = title.casefold().strip()

    if any(
        str(module.get("title", "")).casefold().strip() == wanted
        for module in draft["modules"]
    ):
        return {"type": "module_exists", "title": title}

    _append_module(draft, {
        "module_number": 0,
        "title": title,
        "hours": 0,
        "topics": [
            f"Key concepts of {title}",
            "Guided practice and discussion",
        ],
        "practical_exercises": [
            f"Complete a practical exercise on {title}"
        ],
    })
    return {"type": "module_added", "title": title}


def _add_capstone(draft: dict[str, Any]) -> dict[str, Any]:
    if any(
        "capstone" in str(module.get("title", "")).casefold()
        for module in draft["modules"]
    ):
        return {"type": "capstone_exists"}

    _append_module(draft, {
        "module_number": 0,
        "title": CAPSTONE_TITLE,
        "hours": 0,
        "topics": [
            "Project planning and scoping",
            "Implementation with feedback",
            "Final presentation and review",
        ],
        "practical_exercises": [
            "Complete and present a capstone project"
        ],
    })
    return {"type": "capstone_added"}


def _set_audience(draft: dict[str, Any], audience: str) -> dict[str, Any]:
    old = draft.get("target_audience")

    if str(old or "").casefold().strip() == audience.casefold().strip():
        return {"type": "audience_unchanged", "audience": audience}

    draft["target_audience"] = audience
    return {"type": "audience_changed", "old": old, "new": audience}


_UNCHANGED_TYPES = {
    "hours_unchanged", "module_exists", "capstone_exists",
    "audience_unchanged",
}


def apply_refinements(
    draft: dict[str, Any], refinements: list[Refinement]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (updated copy of the draft, list of change records)."""
    updated = copy.deepcopy(draft)
    changes: list[dict[str, Any]] = []

    for refinement in refinements:
        if refinement.kind == "set_hours":
            changes.append(_set_hours(updated, int(refinement.value)))
        elif refinement.kind == "add_module":
            changes.append(_add_module(updated, str(refinement.value)))
        elif refinement.kind == "add_capstone":
            changes.append(_add_capstone(updated))
        elif refinement.kind == "set_audience":
            changes.append(_set_audience(updated, str(refinement.value)))
        else:
            raise RefinementError("unsupported_change")

    updated["practical_exercises"] = [
        exercise
        for module in updated["modules"]
        for exercise in module.get("practical_exercises", [])
    ]

    if any(change["type"] not in _UNCHANGED_TYPES for change in changes):
        updated["revision"] = int(updated.get("revision", 1)) + 1

    return updated, changes
