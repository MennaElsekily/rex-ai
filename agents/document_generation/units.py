"""
Deterministic building blocks for the document types the model writes "per unit" (one model call per module, or per
assessment for rubrics), and the arithmetic and plans those documents use.

Everything here depends only on the approved curriculum, never on model output, so the composer, the validator
(which recomposes a document to prove curriculum text was not changed) and the generator all agree on the document's
shape.

- unit ids: which modules / assessments a document is written for;
- rubric specifications and scoring: criteria come from learning outcomes, weights and points are computed with exact
  fractions so every rubric totals exactly 100 (see rubric_weights);
- the question plan: how many questions of which type and difficulty each module gets;
- topic terms: the words of a module, used to check that a question is about the curriculum.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Optional

from agents.text_utils import normalize_for_matching

TOTAL_POINTS = 100
MAX_CRITERIA = 12            # outcomes a rubric can use (curricula have at most 8)
CAPSTONE_RUBRIC_ID = "CAP"
PERFORMANCE_LEVELS = ("beginning", "developing", "proficient", "advanced")   # in order; level k earns k/4 of the weight

# What each module gets in a question bank: (question kind, difficulty). Fixed, so the document's shape is known
# from the curriculum alone.
QUESTION_PLAN: tuple[tuple[str, str], ...] = (("mc", "easy"), ("mc", "medium"), ("tf", "easy"), ("sa", "hard"))
MC_OPTIONS = 4


# -- which units a document has ----------------------------------------------------

def module_ids(curriculum: dict[str, Any]) -> list[str]:
    return [m.get("id") for m in curriculum.get("modules") or []]


def find_module(curriculum: dict[str, Any], module_id: str) -> Optional[dict[str, Any]]:
    return next((m for m in curriculum.get("modules") or [] if m.get("id") == module_id), None)


def outcome_statement(curriculum: dict[str, Any], outcome_id: str) -> Optional[str]:
    return next((o.get("statement") for o in curriculum.get("learning_outcomes") or [] if o.get("id") == outcome_id), None)


def rubric_specs(curriculum: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One rubric per curriculum assessment, plus one for the capstone project. A rubric's criteria are the learning
    outcomes the assessment is linked to (only outcomes the curriculum defines, each once, in order).
    {"id", "kind": "assessment" | "capstone", "title", "description", "criteria": [{"outcome_id", "statement"}]}
    """
    specs = []
    sources = [("assessment", a.get("id"), f"{a.get('id')} ({a.get('type')})", a.get("description"), a.get("outcome_ids"))
               for a in curriculum.get("assessments") or []]
    capstone = curriculum.get("capstone")
    if isinstance(capstone, dict):
        sources.append(("capstone", CAPSTONE_RUBRIC_ID, str(capstone.get("title")), capstone.get("description"),
                        capstone.get("outcome_ids")))
    for kind, spec_id, title, description, outcome_ids in sources:
        criteria, seen = [], set()
        for outcome_id in outcome_ids or []:
            statement = outcome_statement(curriculum, outcome_id)
            if statement and outcome_id not in seen and len(criteria) < MAX_CRITERIA:
                seen.add(outcome_id)
                criteria.append({"outcome_id": outcome_id, "statement": statement})
        specs.append({"id": spec_id, "kind": kind, "title": title, "description": description, "criteria": criteria})
    return specs


def rubric_spec(curriculum: dict[str, Any], rubric_id: str) -> Optional[dict[str, Any]]:
    return next((s for s in rubric_specs(curriculum) if s["id"] == rubric_id), None)


def unit_ids(unit_field: str, curriculum: dict[str, Any]) -> list[str]:
    """The ids a per-unit document is written for ("modules" -> module ids, "rubrics" -> rubric ids)."""
    if unit_field == "rubrics":
        return [s["id"] for s in rubric_specs(curriculum) if s["criteria"]]
    return module_ids(curriculum)


# -- rubric scoring ----------------------------------------------------------------

def rubric_weights(criteria: int, total: int = TOTAL_POINTS) -> list[int]:
    """
    Whole-number weights for `criteria` criteria that add up to exactly `total`: an equal share each, and the
    remainder (one point at a time) to the first criteria. 3 criteria -> [34, 33, 33]; 7 -> [15, 15, 14, 14, 14, 14, 14].
    """
    if criteria < 1:
        return []
    share, remainder = divmod(total, criteria)
    return [share + (1 if i < remainder else 0) for i in range(criteria)]


def level_points(weight: int) -> list[Fraction]:
    """Points a criterion of this weight earns at each performance level: weight x 1/4, 2/4, 3/4, 4/4 (exact)."""
    return [Fraction(weight * (k + 1), len(PERFORMANCE_LEVELS)) for k in range(len(PERFORMANCE_LEVELS))]


def format_points(value: Fraction | int) -> str:
    """Exact points as text: 34, 8.5, 8.25 (quarter steps never need more than two decimals)."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


# -- topic terms (question grounding) -----------------------------------------------

_WORD = re.compile(r"[^\W_]+")
_STOP = frozenset("""a an and are as at be by for from in into is it of on or the to with using use this that these those
what which who how when why does do did can will would should about between during your you we our their its not no yes
true false correct incorrect following statement question answer option options""".split())


def terms(text: Any) -> set[str]:
    """Significant words of a text, normalized (Arabic letters unified, leading Arabic 'al-' dropped)."""
    words = set()
    for word in _WORD.findall(normalize_for_matching(str(text or ""))):
        if word.startswith("ال") and len(word) > 4:
            word = word[2:]
        if len(word) > 2 and word not in _STOP and not word.isdigit():
            words.add(word)
    return words


def module_terms(curriculum: dict[str, Any], module: dict[str, Any]) -> set[str]:
    """Words that make a question 'about' this module: its title, topics, description, activities and outcomes."""
    parts = [module.get("title"), module.get("description"), *(module.get("topics") or []),
             *(a.get("description") for a in module.get("activities") or [])]
    parts += [outcome_statement(curriculum, oid) for oid in module.get("outcome_ids") or []]
    found: set[str] = set()
    for part in parts:
        found |= terms(part)
    return found


def shares_a_term(text: str, wanted: set[str]) -> bool:
    """True if the text uses one of the wanted words (also as a prefix of 4+ letters, to allow plurals and suffixes)."""
    for word in terms(text):
        for target in wanted:
            if word == target or (min(len(word), len(target)) >= 4 and (word.startswith(target) or target.startswith(word))):
                return True
    return False


def normalized_text(text: Any) -> str:
    return " ".join(_WORD.findall(normalize_for_matching(str(text or ""))))
