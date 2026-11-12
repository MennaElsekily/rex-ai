"""
Turns a curriculum's structured content into plain, labelled text an
admin can edit in a handful of text areas (rex_ai_app.py), and back.

Not raw JSON: each editable section has its own widget and its own
small, documented line format (one item per line for plain lists;
"key: value" lines per block, blocks separated by a line of dashes, for
modules and assessments, whose records have more than one field).
Nothing here talks to a database or a model -- these are pure
functions, independent of Streamlit, so they are unit-tested directly.
"""

from __future__ import annotations

import re
from typing import Any

_BLOCK_SEP = re.compile(r"^-{3,}\s*$", re.MULTILINE)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def format_lines(items: list[str]) -> str:
    return "\n".join(str(item) for item in items or [])


def parse_lines(text: str) -> list[str]:
    return _lines(text)


# ------------------------------------------------------------ outcomes

def format_outcomes(outcomes: list[dict[str, Any]]) -> str:
    return "\n".join(f"{o.get('id', '')}: {o.get('statement', '')}" for o in outcomes or [])


def parse_outcomes(text: str, existing_ids: list[str]) -> list[dict[str, str]]:
    """"LO1: statement" per line; a line with no "id:" prefix gets the next unused LOn id."""
    used = set()
    next_number = 1
    outcomes = []
    for line in _lines(text):
        if ":" in line and re.match(r"^[\w-]+\s*:", line):
            identifier, statement = line.split(":", 1)
            identifier = identifier.strip()
        else:
            identifier, statement = "", line
        statement = statement.strip()
        if not identifier:
            while f"LO{next_number}" in used:
                next_number += 1
            identifier = f"LO{next_number}"
        used.add(identifier)
        outcomes.append({"id": identifier, "statement": statement})
    return outcomes


# -------------------------------------------------------------- blocks

def _parse_block(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in _lines(block):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


# -------------------------------------------------------------- modules

def format_modules(modules: list[dict[str, Any]]) -> str:
    blocks = []
    for module in modules or []:
        activities = " | ".join(f"{a.get('type', '')}: {a.get('description', '')}"
                                for a in module.get("activities") or [])
        blocks.append("\n".join([
            f"id: {module.get('id', '')}", f"title: {module.get('title', '')}",
            f"hours: {module.get('hours', '')}", f"topics: {', '.join(module.get('topics') or [])}",
            f"description: {module.get('description', '')}", f"activities: {activities}",
            f"outcome_ids: {', '.join(module.get('outcome_ids') or [])}",
        ]))
    return f"\n{'-' * 20}\n".join(blocks)


def parse_modules(text: str) -> list[dict[str, Any]]:
    modules = []
    next_number = 1
    used_ids: set[str] = set()
    blocks = [b for b in _BLOCK_SEP.split(text or "") if b.strip()]
    for block in blocks:
        fields = _parse_block(block)
        if not fields.get("title"):
            continue
        module_id = fields.get("id") or ""
        if not module_id or module_id in used_ids:
            while f"M{next_number}" in used_ids:
                next_number += 1
            module_id = f"M{next_number}"
        used_ids.add(module_id)
        try:
            hours = int(float(fields.get("hours", "0") or 0))
        except ValueError:
            hours = 0
        activities = []
        for piece in (fields.get("activities", "") or "").split("|"):
            piece = piece.strip()
            if not piece:
                continue
            if ":" in piece:
                a_type, a_description = piece.split(":", 1)
            else:
                a_type, a_description = "activity", piece
            activities.append({"type": a_type.strip(), "description": a_description.strip()})
        modules.append({
            "id": module_id, "title": fields.get("title", ""), "hours": hours,
            "topics": _csv(fields.get("topics", "")), "description": fields.get("description", ""),
            "activities": activities, "outcome_ids": _csv(fields.get("outcome_ids", "")),
        })
    return modules


# --------------------------------------------------------- assessments

def format_assessments(assessments: list[dict[str, Any]]) -> str:
    blocks = []
    for assessment in assessments or []:
        blocks.append("\n".join([
            f"id: {assessment.get('id', '')}", f"type: {assessment.get('type', '')}",
            f"description: {assessment.get('description', '')}",
            f"outcome_ids: {', '.join(assessment.get('outcome_ids') or [])}",
        ]))
    return f"\n{'-' * 20}\n".join(blocks)


def parse_assessments(text: str) -> list[dict[str, Any]]:
    assessments = []
    next_number = 1
    used_ids: set[str] = set()
    blocks = [b for b in _BLOCK_SEP.split(text or "") if b.strip()]
    for block in blocks:
        fields = _parse_block(block)
        if not fields.get("description"):
            continue
        assessment_id = fields.get("id") or ""
        if not assessment_id or assessment_id in used_ids:
            while f"A{next_number}" in used_ids:
                next_number += 1
            assessment_id = f"A{next_number}"
        used_ids.add(assessment_id)
        assessments.append({
            "id": assessment_id, "type": fields.get("type", "assessment"),
            "description": fields.get("description", ""), "outcome_ids": _csv(fields.get("outcome_ids", "")),
        })
    return assessments


# ------------------------------------------------------------- capstone

def capstone_fields(capstone: Any) -> dict[str, str]:
    """Plain strings for the capstone's own widgets (title/outcome_ids/deliverables as one line each)."""
    if not isinstance(capstone, dict):
        return {"title": "", "description": "", "outcome_ids": "", "deliverables": ""}
    return {"title": capstone.get("title", ""), "description": capstone.get("description", ""),
            "outcome_ids": ", ".join(capstone.get("outcome_ids") or []),
            "deliverables": format_lines(capstone.get("deliverables") or [])}


def parse_capstone(included: bool, title: str, description: str, outcome_ids: str, deliverables: str) -> Any:
    if not included:
        return None
    return {"title": title.strip(), "description": description.strip(), "outcome_ids": _csv(outcome_ids),
            "deliverables": parse_lines(deliverables)}


# ------------------------------------------------------------- compose

def apply_edits(content: dict[str, Any], edits: dict[str, Any]) -> dict[str, Any]:
    """
    A new content dict: `content` with every field `edits` provides
    replaced, everything else (language, references, academy_evidence,
    data_notice, alignment, ...) kept exactly as it was. total_hours is
    recomputed from the edited modules, since that is what the hours a
    curriculum actually offers means.
    """
    import copy

    updated = copy.deepcopy(content)
    updated.update(edits)
    if "modules" in edits:
        updated["total_hours"] = sum(int(m.get("hours") or 0) for m in updated["modules"])
    return updated
