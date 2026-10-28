"""
Framework alignment (R7, R8): compare a curriculum with an approved
framework's requirements.

Only requirements from an approved framework file are used (frameworks.py;
none is shipped). The tool's word-overlap pre-mapping is passed to the
model as a hint; the model judges each requirement and cites curriculum
element ids. Then deterministic checks:
- every requirement gets exactly one result (a missing judgement becomes
  "unable_to_determine");
- "covered" / "partially_covered" without an existing element id becomes
  "unable_to_determine" (no evidence, no claim);
- element ids that do not exist are dropped.
Not and partially covered requirements are the gaps, each with the
model's recommendation. Alignment is never presented as accreditation.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from agents.curriculum_research import prompts
from agents.curriculum_research.schemas import ALIGNMENT_STATUSES, AlignmentJudgement
from agents.text_utils import normalize_for_matching

BATCH = 12
_GENERIC = {"framework", "frameworks", "standard", "standards", "competency", "competencies", "national",
            "qualifications", "qualification", "skills", "workforce", "for", "and", "the", "of"}


class AlignmentError(ValueError):
    """The framework or curriculum cannot be compared (the message says why)."""


def match_framework(text: str, frameworks: list[dict[str, Any]]) -> Optional[str]:
    """The id of the approved framework a message names (by id, name or a distinctive word)."""
    norm = normalize_for_matching(text)
    words = set(re.findall(r"[\w*-]+", norm))
    for framework in frameworks:
        fid = framework["framework_id"]
        if fid in norm or normalize_for_matching(framework.get("name") or "") in norm:
            return fid
    for framework in frameworks:
        distinctive = {w for w in re.split(r"[-_\s]+", framework["framework_id"]) if len(w) >= 3} | {
            w.casefold() for w in re.findall(r"\b[A-Z][A-Z0-9*]{2,}\b", framework.get("name") or "")}
        if (distinctive - _GENERIC) & words:
            return framework["framework_id"]
    return None


def _elements(curriculum: dict[str, Any]) -> tuple[str, set[str]]:
    lines, ids = [], set()
    for outcome in curriculum.get("learning_outcomes") or []:
        lines.append(f"- {outcome['id']} (outcome): {outcome['statement']}")
        ids.add(outcome["id"])
    for module in curriculum.get("modules") or []:
        lines.append(f"- {module['id']} (module, {module.get('hours')} h): {module['title']} -- "
                     f"{', '.join(module.get('topics') or [])}")
        ids.add(module["id"])
    for assessment in curriculum.get("assessments") or []:
        lines.append(f"- {assessment['id']} (assessment, {assessment.get('type')}): {assessment.get('description')}")
        ids.add(assessment["id"])
    return "\n".join(lines), ids


def align(curriculum: dict[str, Any], framework_id: str, call: Callable[..., dict[str, Any]],
          judge: Callable[..., AlignmentJudgement], reply_language: str) -> dict[str, Any]:
    """
    The alignment record stored with the curriculum. ``call`` reaches the
    tools (get_framework_requirements, compare_with_framework); ``judge``
    asks the model (prompts.ALIGN -> AlignmentJudgement).
    """
    loaded = call("get_framework_requirements", framework_id=framework_id)
    if loaded.get("status") != "ok":
        raise AlignmentError(loaded.get("message") or f"Framework {framework_id} is not available.")
    framework = loaded["framework"]
    compared = call("compare_with_framework", curriculum=curriculum, framework_id=framework_id)
    if compared.get("status") != "ok":
        raise AlignmentError(compared.get("message") or "The comparison failed.")
    hints = {m["requirement_id"]: m.get("candidates") or [] for m in compared.get("pre_mapping") or []}
    elements_block, element_ids = _elements(curriculum)
    requirements = framework["requirements"]

    judged: dict[str, dict[str, Any]] = {}
    for start in range(0, len(requirements), BATCH):
        batch = requirements[start:start + BATCH]
        lines = []
        for requirement in batch:
            hint = ", ".join(f"{c['element_id']} [{' '.join(c['shared_terms'][:4])}]" for c in hints.get(requirement["id"], []))
            lines.append(f"- {requirement['id']}: {requirement['text']}" + (f" (possible evidence: {hint})" if hint else ""))
        answer = judge(prompts.ALIGN, language_name=prompts.LANGUAGE_NAMES[reply_language],
                       framework_name=f"{framework['name']} ({framework.get('version')})",
                       elements_block=elements_block, requirements_block="\n".join(lines))
        wanted = {r["id"] for r in batch}
        for item in answer.items:
            if item.requirement_id in wanted and item.requirement_id not in judged:
                judged[item.requirement_id] = item.model_dump()

    items = []
    for requirement in requirements:
        item = judged.get(requirement["id"])
        if item is None:
            item = {"status": "unable_to_determine", "element_ids": [], "recommendation": None,
                    "explanation": "No judgement was produced for this requirement."}
        item["requirement_id"] = requirement["id"]
        item["requirement"] = requirement["text"]
        item["element_ids"] = [e for e in item.get("element_ids") or [] if e in element_ids]
        if item["status"] in ("covered", "partially_covered") and not item["element_ids"]:
            item["explanation"] = (f"Judged {item['status'].replace('_', ' ')} without valid evidence, so it "
                                   f"cannot be confirmed. {item.get('explanation') or ''}").strip()
            item["status"] = "unable_to_determine"
        if item["status"] not in ("partially_covered", "not_covered"):
            item["recommendation"] = None
        items.append(item)

    summary = {status: sum(1 for i in items if i["status"] == status) for status in ALIGNMENT_STATUSES}
    return {
        "framework": {k: framework.get(k) for k in ("framework_id", "name", "publisher", "version", "source_url",
                                                     "retrieved_on")},
        "items": items, "summary": summary,
        "gaps": [i["requirement_id"] for i in items if i["status"] in ("partially_covered", "not_covered")],
        "method": "Model judgement per requirement with a word-overlap hint; covered/partial require existing "
                  "curriculum element ids as evidence. Not accreditation.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
