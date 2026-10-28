"""
Deterministic checks for a curriculum (no LLM).

validate_curriculum(curriculum, requested_hours, framework=None) returns
{"valid", "errors", "warnings"}; each issue is {"code", "path", "message"}.
The rules are the ones that can be decided exactly:

- required sections present and of the right type          missing_section / bad_type
- module hours are whole numbers above zero                  module_hours
- the module hours add up to the requested total             hours_total
- learning outcomes start with a measurable action verb      outcome_not_measurable
  (English and Arabic; "understand / know / learn / يفهم / يعرف / يتعلم" are not)
- outcomes, modules, assessments and the capstone link up    outcome_unlinked / dangling_outcome_ref
- every cited source id exists in the references             unknown_source
- alignment entries cite real requirements and real curriculum
  elements (otherwise they cannot be "covered")              alignment_evidence
- no claim of official accreditation or certification        accreditation_claim
- the content is written in the requested language           language_mismatch

Judgements that need understanding (is this outcome *appropriate*?) are
left to the language model; nothing here guesses.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.curriculum_research.schemas import ALIGNMENT_STATUSES, LEVELS
from agents.text_utils import normalize_for_matching

_MEASURABLE_EN = frozenset("""
analyse analyze apply assess automate build calculate choose classify clean collect communicate compare
compose compute conduct configure construct create critique debug define demonstrate deploy describe
design detect develop diagnose differentiate distinguish document estimate evaluate examine execute
explain extract formulate identify illustrate implement integrate interpret justify list manage map
measure model monitor operate optimise optimize organise organize perform plan predict prepare present
prioritise prioritize produce program query recommend refactor report run secure select solve
summarise summarize test train transform troubleshoot tune use validate visualise visualize write
""".split())
_VAGUE_EN = frozenset("understand know learn appreciate grasp familiarise familiarize realise realize".split())

# Arabic: present-tense third-person verbs and verbal nouns (normalized
# by agents/text_utils: alef forms -> ا, ة -> ه, ى -> ي).
_MEASURABLE_AR = frozenset(normalize_for_matching(w) for w in """
يطبق تطبيق يحلل تحليل يصمم تصميم يبني بناء ينشئ انشاء يقيم تقييم يقارن مقارنه يشرح شرح يحدد تحديد
ينفذ تنفيذ يستخدم استخدام يطور تطوير يكتب كتابه يعرض عرض يقدم تقديم يختبر اختبار يصنف تصنيف يفسر
تفسير يحسب حساب يدرب تدريب يوظف توظيف يعالج معالجه يبرمج برمجه يصف وصف يميز تمييز ينظم تنظيم يخطط
تخطيط يحل حل يقيس قياس يستنتج استنتاج يعد اعداد يجهز تجهيز ينشر نشر يدير اداره يراقب مراقبه يوثق
توثيق يلخص تلخيص ينمذج نمذجه يقترح اقتراح يبرر تبرير يقدر تقدير يستخرج استخراج يصحح تصحيح يحسن تحسين
يتحقق تحقق يختار اختيار يستكشف استكشاف يربط ربط ينظف تنظيف يدمج دمج يؤتمت اتمته يعد يصوغ صياغه
""".split())
_VAGUE_AR = frozenset(normalize_for_matching(w) for w in "يفهم فهم يعرف معرفه يتعلم تعلم يدرك ادراك يتعرف تعرف يلم المام".split())

# Lead-ins before the verb: "Learners will be able to", "أن", "قادرا على".
_LEAD_IN = re.compile(
    r"^(?:(?:by the end[^,]*,\s*)?(?:the\s+)?(?:learners?|students?|participants?|trainees?)\s+"
    r"(?:will|should|can|are able to)\s+(?:be able to\s+)?|be able to\s+|able to\s+)",
    re.IGNORECASE,
)
_LEAD_IN_AR = re.compile(
    r"^(?:(?:في نهايه[^،,]*[،,]\s*)?(?:سيكون|يكون|ان يكون)?\s*(?:المتدرب|المتعلم|الطالب|المشارك)?\s*"
    r"(?:قادرا|قادر|قادره)?\s*(?:علي)?\s*(?:ان)?\s+)"
)

_ACCREDITATION_EN = re.compile(
    r"(?<!not )(?<!never )\b(?:is|are|has been|have been|been|be|officially)\s+(?:accredited|certified)\b"
    r"|\baccredited by\b|\bofficially (?:approved|recognised|recognized)\b|\bfully compliant with\b",
    re.IGNORECASE,
)
_ACCREDITATION_AR = re.compile(
    normalize_for_matching(r"(?<!غير )(?:معتمد|معتمده) (?:من|رسميا)|حاصل علي اعتماد|اعتماد رسمي|مطابق تماما")
)
_ARABIC_LETTER = re.compile(r"[ء-ي]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def first_verb(statement: str, language: str) -> str:
    """The outcome's leading verb, after lead-ins such as "Learners will be able to"."""
    text = str(statement or "").strip()
    if language == "ar":
        norm = normalize_for_matching(text)
        norm = _LEAD_IN_AR.sub("", norm).strip()
        return norm.split()[0] if norm.split() else ""
    text = _LEAD_IN.sub("", text).strip()
    word = re.sub(r"[^a-zA-Z-]", "", text.split()[0]) if text.split() else ""
    return word.casefold()


def is_measurable(statement: str, language: str) -> bool:
    verb = first_verb(statement, language)
    if language == "ar":
        return verb in _MEASURABLE_AR and verb not in _VAGUE_AR
    return verb in _MEASURABLE_EN and verb not in _VAGUE_EN


def _text_fields(curriculum: dict[str, Any]) -> list[tuple[str, str]]:
    """(path, text) for every model-written text of the curriculum."""
    fields = [(name, curriculum.get(name)) for name in ("title", "overview", "rationale", "target_audience")]
    fields += [(f"objectives[{i}]", t) for i, t in enumerate(curriculum.get("objectives") or [])]
    fields += [(f"learning_outcomes[{i}]", o.get("statement")) for i, o in
               enumerate(curriculum.get("learning_outcomes") or []) if isinstance(o, dict)]
    for i, module in enumerate(curriculum.get("modules") or []):
        if not isinstance(module, dict):
            continue
        fields += [(f"modules[{i}].title", module.get("title")), (f"modules[{i}].description", module.get("description"))]
        fields += [(f"modules[{i}].topics", t) for t in module.get("topics") or []]
        fields += [(f"modules[{i}].activities[{j}]", a.get("description"))
                   for j, a in enumerate(module.get("activities") or []) if isinstance(a, dict)]
    # §17 found an Arabic curriculum whose assessments were written in English.
    for i, assessment in enumerate(curriculum.get("assessments") or []):
        if isinstance(assessment, dict):
            fields.append((f"assessments[{i}]", f"{assessment.get('type') or ''} {assessment.get('description') or ''}"))
    capstone = curriculum.get("capstone")
    if isinstance(capstone, dict):
        fields.append(("capstone", f"{capstone.get('title') or ''} {capstone.get('description') or ''}"))
    return [(path, str(text)) for path, text in fields if text]


def _all_text(curriculum: dict[str, Any]) -> str:
    return " ".join(text for _, text in _text_fields(curriculum))


def _wrong_language_fields(curriculum: dict[str, Any], language: str) -> list[str]:
    """
    Fields written mostly in the other language. Short texts (under 20
    letters, e.g. "Power BI") are skipped: technical terms stay as they are.
    """
    wrong = []
    for path, text in _text_fields(curriculum):
        letters = len(_ARABIC_LETTER.findall(text)) + len(_LATIN_LETTER.findall(text))
        if letters < 20:
            continue
        ratio = _language_ratio(text)
        if (language == "ar" and ratio < 0.25) or (language == "en" and ratio > 0.5):
            wrong.append(path)
    return wrong


def _language_ratio(text: str) -> float:
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    return arabic / (arabic + latin) if arabic + latin else 0.0


def validate_curriculum(
    curriculum: dict[str, Any],
    requested_hours: int,
    framework: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Check a curriculum against the deterministic rules (see the module docstring)."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(curriculum, dict):
        return {"valid": False, "errors": [_issue("bad_type", "", "The curriculum must be an object.")],
                "warnings": []}
    language = curriculum.get("language") if curriculum.get("language") in ("ar", "en") else "en"

    # Required sections and types.
    text_fields = ("title", "overview", "rationale", "target_audience", "level")
    list_fields = {"objectives": 1, "learning_outcomes": 2, "modules": 1, "assessments": 1,
                   "tools_resources": 1, "competencies": 1}
    for field in text_fields:
        if not isinstance(curriculum.get(field), str) or not curriculum[field].strip():
            errors.append(_issue("missing_section", field, f"The {field} is missing."))
    for field, minimum in list_fields.items():
        value = curriculum.get(field)
        if not isinstance(value, list):
            errors.append(_issue("bad_type", field, f"The {field} must be a list."))
        elif len(value) < minimum:
            errors.append(_issue("missing_section", field, f"At least {minimum} {field} required."))
    if not isinstance(curriculum.get("prerequisites"), list):
        errors.append(_issue("bad_type", "prerequisites", "The prerequisites must be a list (it may say none)."))
    if curriculum.get("level") and curriculum["level"] not in LEVELS:
        errors.append(_issue("bad_type", "level", f"The level must be one of {', '.join(LEVELS)}."))
    if not curriculum.get("capstone") and not curriculum.get("capstone_omitted_reason"):
        errors.append(_issue("missing_section", "capstone", "A capstone project (or the reason for none) is required."))
    for field in ("assumptions", "open_questions", "references", "academy_evidence"):
        if not isinstance(curriculum.get(field, []), list):
            errors.append(_issue("bad_type", field, f"The {field} must be a list."))

    modules = [m for m in curriculum.get("modules") or [] if isinstance(m, dict)]
    outcomes = [o for o in curriculum.get("learning_outcomes") or [] if isinstance(o, dict)]
    assessments = [a for a in curriculum.get("assessments") or [] if isinstance(a, dict)]
    outcome_ids = {o.get("id") for o in outcomes}

    # Hours.
    total = 0
    for index, module in enumerate(modules):
        hours = module.get("hours")
        if not isinstance(hours, int) or isinstance(hours, bool) or hours <= 0:
            errors.append(_issue("module_hours", f"modules[{index}].hours",
                                 f"Module {module.get('id') or index + 1} needs a whole number of hours above zero."))
        else:
            total += hours
    if modules and total != requested_hours:
        errors.append(_issue("hours_total", "modules", f"The module hours add up to {total}, not the requested "
                                                         f"{requested_hours}."))
    if curriculum.get("total_hours") != requested_hours:
        errors.append(_issue("hours_total", "total_hours",
                             f"The total hours must be the requested {requested_hours}."))

    # Measurable outcomes.
    for index, outcome in enumerate(outcomes):
        if not is_measurable(outcome.get("statement", ""), language):
            errors.append(_issue("outcome_not_measurable", f"learning_outcomes[{index}]",
                                 f"{outcome.get('id')}: start with a measurable action verb "
                                 f"(not \"{first_verb(outcome.get('statement', ''), language)}\")."))

    # Links between outcomes, modules, assessments and the capstone.
    linked_by_module: set[str] = set()
    for index, module in enumerate(modules):
        refs = module.get("outcome_ids") or []
        if not refs:
            errors.append(_issue("outcome_unlinked", f"modules[{index}]", f"{module.get('id')} develops no outcome."))
        for ref in refs:
            if ref not in outcome_ids:
                errors.append(_issue("dangling_outcome_ref", f"modules[{index}]", f"{module.get('id')} cites "
                                                                                  f"unknown outcome {ref}."))
        linked_by_module.update(r for r in refs if r in outcome_ids)
        if not module.get("activities"):
            warnings.append(_issue("no_activities", f"modules[{index}]", f"{module.get('id')} has no practical activity."))
    if modules and not any(m.get("activities") for m in modules):
        errors.append(_issue("missing_section", "modules.activities", "No practical exercises or activities."))
    linked_by_assessment: set[str] = set()
    for index, assessment in enumerate(assessments):
        refs = assessment.get("outcome_ids") or []
        if not refs:
            errors.append(_issue("outcome_unlinked", f"assessments[{index}]", f"{assessment.get('id')} assesses no outcome."))
        for ref in refs:
            if ref not in outcome_ids:
                errors.append(_issue("dangling_outcome_ref", f"assessments[{index}]",
                                     f"{assessment.get('id')} cites unknown outcome {ref}."))
        linked_by_assessment.update(r for r in refs if r in outcome_ids)
    capstone = curriculum.get("capstone")
    if isinstance(capstone, dict):
        refs = capstone.get("outcome_ids") or []
        if not refs:
            errors.append(_issue("outcome_unlinked", "capstone", "The capstone demonstrates no outcome."))
        linked_by_assessment.update(r for r in refs if r in outcome_ids)
        for ref in refs:
            if ref not in outcome_ids:
                errors.append(_issue("dangling_outcome_ref", "capstone", f"The capstone cites unknown outcome {ref}."))
    for outcome in outcomes:
        oid = outcome.get("id")
        if oid not in linked_by_module:
            errors.append(_issue("outcome_unlinked", "learning_outcomes", f"{oid} is not developed by any module."))
        if oid not in linked_by_assessment:
            errors.append(_issue("outcome_unlinked", "learning_outcomes", f"{oid} is not assessed."))

    # Source references.
    known_sources = {r.get("source_id") for r in curriculum.get("references") or [] if isinstance(r, dict)}
    for index, competency in enumerate(curriculum.get("competencies") or []):
        for sid in (competency or {}).get("source_ids") or []:
            if sid not in known_sources:
                errors.append(_issue("unknown_source", f"competencies[{index}]", f"Unknown source {sid}."))
    for index, reference in enumerate(curriculum.get("references") or []):
        if not (isinstance(reference, dict) and reference.get("url") and reference.get("retrieved_at")):
            errors.append(_issue("unknown_source", f"references[{index}]",
                                 "A reference must come from a retrieved source (URL and retrieval time)."))

    # Framework alignment.
    alignment = curriculum.get("alignment")
    if alignment:
        requirement_ids = {r.get("id") for r in (framework or {}).get("requirements", [])}
        element_ids = outcome_ids | {m.get("id") for m in modules} | {a.get("id") for a in assessments}
        for index, item in enumerate(alignment.get("items", [])):
            status = item.get("status")
            if status not in ALIGNMENT_STATUSES:
                errors.append(_issue("alignment_evidence", f"alignment.items[{index}]", f"Unknown status {status}."))
            if item.get("requirement_id") not in requirement_ids:
                errors.append(_issue("alignment_evidence", f"alignment.items[{index}]",
                                     f"{item.get('requirement_id')} is not a requirement of the framework."))
            if status in ("covered", "partially_covered") and not (
                item.get("element_ids") and set(item["element_ids"]) <= element_ids
            ):
                errors.append(_issue("alignment_evidence", f"alignment.items[{index}]",
                                     f"{item.get('requirement_id')}: \"{status}\" needs existing curriculum "
                                     "elements as evidence."))

    # Accreditation claims and language.
    text = _all_text(curriculum)
    # Alignment explanations are model-written too: no compliance claims there either.
    claims = " ".join([text, *[f"{i.get('explanation') or ''} {i.get('recommendation') or ''}"
                               for i in (alignment or {}).get("items", []) if isinstance(i, dict)]])
    if _ACCREDITATION_EN.search(claims) or _ACCREDITATION_AR.search(normalize_for_matching(claims)):
        errors.append(_issue("accreditation_claim", "", "The curriculum must not claim official accreditation, "
                                                       "certification or compliance."))
    ratio = _language_ratio(text)
    if language == "ar" and ratio < 0.4:
        errors.append(_issue("language_mismatch", "", "The curriculum must be written in Arabic."))
    if language == "en" and ratio > 0.2:
        errors.append(_issue("language_mismatch", "", "The curriculum must be written in English."))
    wrong = _wrong_language_fields(curriculum, language)
    if wrong and not any(e["code"] == "language_mismatch" for e in errors):
        errors.append(_issue("language_mismatch", wrong[0],
                             f"Written in the wrong language: {', '.join(wrong[:5])}"
                             + (f" and {len(wrong) - 5} more" if len(wrong) > 5 else "") + "."))

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def rebalance_hours(modules: list[dict[str, Any]], total_hours: int) -> list[int]:
    """
    Module hours scaled to exactly ``total_hours`` (largest-remainder
    rounding, at least one hour each), keeping the modules' proportions.
    Used only after the one repair attempt; the caller records it.
    """
    if not modules:
        return []
    if total_hours < len(modules):
        raise ValueError(f"{total_hours} hours cannot cover {len(modules)} modules.")
    weights = [m["hours"] if isinstance(m.get("hours"), int) and m["hours"] > 0 else 1 for m in modules]
    raw = [total_hours * w / sum(weights) for w in weights]
    shares = [int(r) for r in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - shares[i], reverse=True)[: total_hours - sum(shares)]:
        shares[index] += 1
    # A module never drops to zero: take the hour from the largest module.
    for index, share in enumerate(shares):
        if share == 0:
            shares[shares.index(max(shares))] -= 1
            shares[index] = 1
    return shares
