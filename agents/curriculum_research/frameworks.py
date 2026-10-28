"""
Competency / accreditation frameworks for alignment, from approved files.

A framework is one JSON file in CURRICULUM_FRAMEWORKS_DIR (default:
agents/curriculum_research/frameworks/), added by an administrator from
an official source. Required provenance: name, publisher, version,
source_url, retrieved_on. No framework is shipped: until one is added,
alignment reports "unable to determine".

Files are loaded by framework id only (a validated name; never a path).
pre_mapping() is a deterministic first pass (shared terms between each
requirement and the curriculum's outcomes, modules and assessments); the
language model then judges each requirement, and the validator rejects
any "covered" judgement without real evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from agents.config import get_setting

FRAMEWORKS_SETTING = "CURRICULUM_FRAMEWORKS_DIR"
DEFAULT_FRAMEWORKS_DIR = Path(__file__).resolve().parent / "frameworks"
FRAMEWORK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")
REQUIRED_FIELDS = ("framework_id", "name", "publisher", "version", "source_url", "retrieved_on", "requirements")
MAX_REQUIREMENTS = 200
_WORD = re.compile(r"[a-z][a-z0-9+#-]{2,}")
_STOP = frozenset("the and for with that this from into such their using use able will can are".split())


def frameworks_dir() -> Path:
    return Path(get_setting(FRAMEWORKS_SETTING, "") or DEFAULT_FRAMEWORKS_DIR)


def _check(data: Any, framework_id: str) -> Optional[str]:
    if not isinstance(data, dict):
        return "The framework file is not a JSON object."
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    if missing:
        return f"The framework file lacks {', '.join(missing)} (official provenance is required)."
    if data["framework_id"] != framework_id:
        return "The framework_id inside the file does not match its file name."
    requirements = data["requirements"]
    if not isinstance(requirements, list) or not requirements or len(requirements) > MAX_REQUIREMENTS:
        return f"requirements must be a list of 1-{MAX_REQUIREMENTS} items."
    if not all(isinstance(r, dict) and r.get("id") and r.get("text") for r in requirements):
        return "Every requirement needs an id and a text."
    return None


def load_framework(framework_id: str, directory: Optional[Path] = None) -> dict[str, Any]:
    """{"status": "ok", "framework": {...}} or not_found / invalid_request / error."""
    if not isinstance(framework_id, str) or not FRAMEWORK_ID.match(framework_id):
        return {"status": "invalid_request", "message": "Unknown framework id."}
    path = (directory or frameworks_dir()) / f"{framework_id}.json"
    if not path.is_file():
        return {"status": "not_found",
                "message": f"No approved framework document named {framework_id} is available."}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "message": f"The framework file could not be read ({type(exc).__name__})."}
    problem = _check(data, framework_id)
    if problem:
        return {"status": "error", "message": problem}
    return {"status": "ok", "framework": data}


def list_frameworks(directory: Optional[Path] = None) -> list[dict[str, Any]]:
    """Approved frameworks available for alignment (valid files only)."""
    directory = directory or frameworks_dir()
    out = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        result = load_framework(path.stem, directory)
        if result["status"] == "ok":
            framework = result["framework"]
            out.append({k: framework.get(k) for k in ("framework_id", "name", "publisher", "version",
                                                       "source_url", "retrieved_on", "language")}
                       | {"requirement_count": len(framework["requirements"])})
    return out


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text or "").casefold()) if w not in _STOP}


def pre_mapping(
    curriculum: dict[str, Any],
    framework: dict[str, Any],
    translate: Optional[Callable[[str], str]] = None,
) -> list[dict[str, Any]]:
    """
    For each requirement, the curriculum elements sharing terms with it
    (a hint for the model, not a judgement). Arabic curricula are compared
    in English through the existing translator.
    """
    translate = translate or (lambda t: t)
    elements = []
    for outcome in curriculum.get("learning_outcomes") or []:
        elements.append((outcome.get("id"), translate(outcome.get("statement", ""))))
    for module in curriculum.get("modules") or []:
        elements.append((module.get("id"), translate(" ".join([module.get("title", ""), *module.get("topics", [])]))))
    for assessment in curriculum.get("assessments") or []:
        elements.append((assessment.get("id"), translate(assessment.get("description", ""))))
    element_terms = [(eid, _terms(text)) for eid, text in elements if eid]
    mapping = []
    for requirement in framework.get("requirements", []):
        wanted = _terms(requirement["text"])
        candidates = []
        for eid, terms in element_terms:
            shared = wanted & terms
            if shared:
                candidates.append({"element_id": eid, "shared_terms": sorted(shared)[:10]})
        candidates.sort(key=lambda c: len(c["shared_terms"]), reverse=True)
        mapping.append({"requirement_id": requirement["id"], "requirement": requirement["text"],
                        "candidates": candidates[:5]})
    return mapping
