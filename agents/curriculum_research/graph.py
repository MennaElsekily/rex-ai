"""
Agent 3's LangGraph workflow.

Routes (decided in ``understand``):
- generate  (Workflow A): academy_context -> [clarify] -> research ->
  generate -> validate -> [repair -> validate] -> save -> render_pdf -> reply
- revise:   revise -> [generate] -> validate -> [repair] -> save -> render_pdf -> reply
- suggest   (Workflow B): catalogue + gap report + research -> model
  shortlist -> overlap check -> reply
- show / align / approve_attempt / help / unsupported -> reply

The graph reaches tools only through ``tools`` (CurriculumMcpClient, or
the in-process CurriculumToolbox in tests); the model only through
``llm`` (LangChain chat model). Deterministic checks and decisions are in
validation.py, revisions.py and the tools; the model writes content.
Nothing here can approve a curriculum.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

from agents.curriculum_research import prompts, understanding as rules
from agents.curriculum_research.llm import CurriculumModelError, structured
from agents.curriculum_research.revisions import (
    DETERMINISTIC,
    REGENERATE,
    RevisionError,
    add_module,
    apply_change,
    detect_changes,
)
from agents.curriculum_research.alignment import AlignmentError, align as run_alignment, match_framework
from agents.curriculum_research.schemas import (
    LEVELS,
    AlignmentJudgement,
    CapstoneItem,
    CurriculumDetails,
    CurriculumExtras,
    CurriculumOutline,
    ModuleDetailList,
    NewModule,
    RepairPatch,
    RequestSlots,
    SuggestionList,
    assemble_curriculum,
)
from agents.curriculum_research.state import CurriculumState
from agents.curriculum_research.validation import rebalance_hours
from agents.text_utils import normalize_for_matching
from knowledge_base.language import detect_conversation_language

logger = logging.getLogger(__name__)

MAX_HOURS = 1000
MAX_SOURCES = 5
MAX_SUGGESTIONS = 5
STATUS_LABELS = {
    "draft": ("draft (validation problems remain)", "مسودة (توجد مشكلات تحقق)"),
    "pending_review": ("pending review", "بانتظار المراجعة"),
    "changes_requested": ("changes requested", "مطلوب تعديلات"),
    "approved": ("approved", "معتمد"),
}
LEVEL_AR = {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"}
_LEVEL_FROM_TEXT = {normalize_for_matching(k): v for k, v in {
    "beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced",
    "مبتدئ": "Beginner", "متوسط": "Intermediate", "متقدم": "Advanced"}.items()}


def _t(language: str, english: str, arabic: str) -> str:
    return arabic if language == "ar" else english


def _ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "ok"


def _framework_of(curriculum: dict[str, Any]) -> Optional[str]:
    alignment = curriculum.get("alignment") or {}
    return (alignment.get("framework") or {}).get("framework_id") if alignment.get("items") else None


def _level(value: Any, fallback: Optional[str]) -> Optional[str]:
    if value in LEVELS:
        return value
    return _LEVEL_FROM_TEXT.get(normalize_for_matching(str(value or "")).strip(), fallback)


# ---------------------------------------------------------------------
# Prompt blocks
# ---------------------------------------------------------------------

def _academy_block(state: CurriculumState) -> str:
    lines = [f"- {e['course_id']} {e['field']}: {e['value']}" for e in state.get("academy_evidence") or []]
    for course in state.get("related_courses") or []:
        lines.append(f"- Related existing course {course['course_id']} {course.get('name')} ({course.get('level')})")
    if not lines:
        return "None."
    statuses = ", ".join(state.get("data_status") or [])
    return "\n".join(lines) + (f"\n(Catalogue data status: {statuses})" if statuses else "")


def _sources_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "None (no research performed): leave every source_ids list empty."
    return "\n".join(f"- {s['source_id']}: {s.get('title')} ({s.get('publisher')}): {str(s.get('excerpt') or '')[:400]}"
                     for s in sources)


def _outcomes_block(curriculum: dict[str, Any]) -> str:
    return "\n".join(f"- {o['id']}: {o['statement']}" for o in curriculum.get("learning_outcomes") or [])


def _modules_block(curriculum: dict[str, Any]) -> str:
    return "\n".join(f"- {m['id']}: {m['title']} ({m['hours']} h): {', '.join(m.get('topics') or [])}"
                     for m in curriculum.get("modules") or [])


def _summary_for_revision(curriculum: dict[str, Any]) -> str:
    return (f"Current curriculum: {curriculum.get('title')} ({curriculum.get('level')}, "
            f"{curriculum.get('total_hours')} hours) for {curriculum.get('target_audience')}.\n"
            f"Outcomes:\n{_outcomes_block(curriculum)}\nModules:\n{_modules_block(curriculum)}")


# ---------------------------------------------------------------------
# Deterministic clean-up of model output
# ---------------------------------------------------------------------

def _renumber(items: list[Any], prefix: str) -> dict[str, str]:
    """Give items ids prefix1..n; returns old id -> new id."""
    mapping: dict[str, str] = {}
    for index, item in enumerate(items, start=1):
        new = f"{prefix}{index}"
        if item.id and item.id not in mapping:
            mapping[item.id] = new
        item.id = new
    return mapping


# Numbering labels a small model sometimes prefixes to text ("module1: ...",
# "M2 - ...", "الوحدة 3: ..."). Never content: the ids carry the numbering.
_LABEL_PREFIX = re.compile(r"^\s*(?:(?:module|unit|lesson|section|m)\s*\d+|(?:ال)?(?:وحدة|وحده)\s*\d+)\s*[:：\-–—.)]\s*",
                           re.IGNORECASE)


def _strip_label(text: str) -> str:
    stripped = _LABEL_PREFIX.sub("", text or "", count=1)
    return stripped if stripped.strip() else text


def _clean_outline(outline: CurriculumOutline, level: Optional[str], hours: int) -> dict[str, str]:
    for field in ("title", "overview", "rationale", "target_audience"):
        setattr(outline, field, _strip_label(getattr(outline, field)))
    for module in outline.modules:
        module.title = _strip_label(module.title)
    outline.learning_outcomes = [o for o in outline.learning_outcomes if o.statement.strip()]
    outline.modules = [m for m in outline.modules if m.title.strip()]
    if not outline.modules:
        raise CurriculumModelError("The language model returned no modules.")
    if len(outline.modules) > hours:
        outline.modules = outline.modules[:hours]
    outcome_map = _renumber(outline.learning_outcomes, "LO")
    module_map = _renumber(outline.modules, "M")
    for module in outline.modules:
        module.hours = module.hours if isinstance(module.hours, int) and module.hours > 0 else 1
    outline.level = _level(outline.level, level) or level or "Beginner"
    if level:
        outline.level = level  # the admin's or the catalogue's level wins over the model's
    return {**outcome_map, **{f"module:{k}": v for k, v in module_map.items()}}


def _map_refs(refs: list[str], mapping: dict[str, str], known: set[str]) -> list[str]:
    out = []
    for ref in refs:
        ref = mapping.get(ref, ref)
        if ref in known and ref not in out:
            out.append(ref)
    return out


# ---------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------

def build_graph(tools: Any, llm: Any):
    """Compile the workflow. ``tools``: the MCP client (or toolbox); ``llm``: a LangChain chat model."""

    def call(name: str, **kwargs: Any) -> dict[str, Any]:
        result = getattr(tools, name)(**kwargs)
        return result if isinstance(result, dict) else {"status": "error", "message": f"{name} returned nothing."}

    def ask(schema, template, step: str, **values):
        return structured(llm, schema, template.format_messages(**values), step)

    # ------------------------------------------------------- understand

    def understand(state: CurriculumState) -> dict[str, Any]:
        text = state["request_text"]
        memory = dict(state.get("memory") or {})
        language = detect_conversation_language(text) or memory.get("language") or "en"
        update: dict[str, Any] = {"language": language, "notes": [], "repairs": [], "repair_attempts": 0,
                                  "academy_evidence": [], "related_courses": [], "sources": [], "data_status": []}
        pending = memory.get("pending")
        active_id = memory.get("active_curriculum_id")

        def generate(slots: dict[str, Any], out_language: str) -> dict[str, Any]:
            return {**update, "intent": "generate", "slots": slots, "output_language": out_language}

        if rules.is_approve_attempt(text):
            return {**update, "intent": "approve_attempt"}
        if rules.is_send_request(text):
            return {**update, "intent": "unsupported"}

        choice = rules.suggestion_choice(text)
        suggestions = memory.get("last_suggestions") or []
        if choice and suggestions:
            if not 1 <= choice <= len(suggestions):
                return {**update, "intent": "help", "notes": [_t(
                    language, f"There is no suggestion {choice}; the last list had {len(suggestions)}.",
                    f"لا يوجد اقتراح رقم {choice}؛ القائمة الأخيرة فيها {len(suggestions)} اقتراحات.")]}
            picked = suggestions[choice - 1]
            slots = {"topic": picked["title"], "level": _level(picked.get("level"), None),
                     "target_audience": picked.get("target_audience"), "total_hours": picked.get("proposed_hours"),
                     "origin": "suggestion", "suggestion": picked}
            stated = rules.request_slots(text)
            if stated["total_hours"]:
                slots["total_hours"] = stated["total_hours"]
            return generate(slots, rules.output_language(text, language))

        if rules.is_suggest_request(text):
            return {**update, "intent": "suggest"}
        if rules.is_alignment_request(text) and not rules.is_creation_request(text):
            return {**update, "intent": "align"}

        if rules.is_creation_request(text) and not (active_id and _edit_first(text)):
            slots = rules.request_slots(text)
            slots["origin"] = "request"
            if not slots["topic"] and not slots["course_ref"]:
                slots.update(_model_slots(text))
            return generate(slots, rules.output_language(text, language))

        if rules.is_show_request(text) or rules.find_curriculum_id(text):
            return {**update, "intent": "show"}

        if active_id:
            loaded = call("get_curriculum", curriculum_id=active_id)
            if _ok(loaded):
                base = loaded["curriculum"]
                changes = detect_changes(text, base["content"])
                if changes:
                    return {**update, "intent": "revise", "base": base, "revision": {"changes": changes},
                            "output_language": base["content"].get("language", language)}

        if pending:
            slots = dict(pending.get("slots") or {})
            answer = rules.answer_slots(text)
            for key in ("total_hours", "level", "target_audience", "course_ref"):
                if answer.get(key) and not slots.get(key):
                    slots[key] = answer[key]
            if answer.get("total_hours"):
                slots["total_hours"] = answer["total_hours"]
            if answer.get("topic") and "topic" in (pending.get("missing") or []):
                slots["topic"] = answer["topic"]
            if rules.accepts_defaults(text):
                slots["defaults_accepted"] = True
            return generate(slots, pending.get("output_language") or language)

        return {**update, "intent": "help"}

    def _edit_first(text: str) -> bool:
        from agents.curriculum_research.request_parser import refinement_takes_priority
        return refinement_takes_priority(text)

    def _model_slots(text: str) -> dict[str, Any]:
        """LLM fallback when the rules found no topic; values must appear in the message."""
        try:
            found = ask(RequestSlots, prompts.SLOTS, "request understanding", text=text)
        except CurriculumModelError:
            return {}
        norm = normalize_for_matching(text)
        out: dict[str, Any] = {}
        topic_words = [w for w in normalize_for_matching(found.topic or "").split() if len(w) > 2]
        if topic_words and all(w in norm for w in topic_words):
            out["topic"] = found.topic.strip()
        if found.total_hours and str(found.total_hours) in norm:
            out["total_hours"] = found.total_hours
        level = _level(found.level, None)
        if level and rules.find_level(text) == level:
            out["level"] = level
        return out

    def route(state: CurriculumState) -> str:
        return state["intent"]

    # -------------------------------------------------- academy context

    def academy_context(state: CurriculumState) -> dict[str, Any]:
        slots = dict(state["slots"])
        language = state["language"]
        evidence: list[dict[str, Any]] = []
        notes = list(state.get("notes") or [])
        statuses: set[str] = set()
        reference = slots.get("course_ref") or slots.get("topic")
        course = None
        if reference:
            details = call("get_course_details", course=reference)
            # Only the course id or its exact name makes it "the" course
            # ("Python" is not "Python for Data Analysis"); anything else
            # is at most related (found by the overlap check below).
            if _ok(details) and (details.get("matched_by") == "id" or (
                    details.get("matched_by") == "name"
                    and normalize_for_matching(details["course"].get("name") or "").strip()
                    == normalize_for_matching(reference).strip())):
                course = details["course"]
        if course:
            cid = course["course_id"]
            source = ", ".join(course.get("source_files") or []) or "catalogue"
            if course.get("data_status"):
                statuses.add(course["data_status"])
            for field in ("name", "department", "level", "target_audience", "total_hours", "learning_outcomes",
                          "topics", "skills", "prerequisite_courses"):
                value = course.get(field)
                if value not in (None, "", []):
                    evidence.append({"course_id": cid, "field": field,
                                     "value": "; ".join(map(str, value)) if isinstance(value, list) else value,
                                     "source": source})
            slots["topic"] = slots.get("topic") or course.get("name")
            for slot, field in (("total_hours", "total_hours"), ("level", "level"),
                                ("target_audience", "target_audience")):
                if not slots.get(slot) and course.get(field):
                    slots[slot] = _level(course[field], None) if slot == "level" else course[field]
                    notes.append(_t(language, f"{slot.replace('_', ' ').capitalize()} taken from catalogue course {cid}.",
                                    f"أُخذت قيمة ({slot}) من مقرر الكتالوج {cid}."))
            slots["course_ref"] = cid
        related: list[dict[str, Any]] = []
        if slots.get("topic"):
            overlap = call("check_course_overlap", title=slots["topic"], topics=[], outcomes=[])
            if _ok(overlap):
                for item in overlap.get("overlaps") or []:
                    if course and item["course_id"] == course["course_id"]:
                        continue
                    if item["name_similarity"] >= 0.5 or item["term_coverage"] >= 0.5:
                        related.append(item)
                if overlap.get("duplicate") and not course and related:
                    notes.append(_t(language,
                                    f"A similar course already exists in the catalogue: {related[0]['course_id']} "
                                    f"{related[0].get('name')}.",
                                    f"يوجد في الكتالوج مقرر مشابه: {related[0]['course_id']} {related[0].get('name')}."))
        if related:
            listed = call("list_academy_courses", limit=200)
            if _ok(listed):
                statuses.update(s for s in listed.get("data_status") or [])
        # "... aligned with the NICE framework": align after generating, but
        # only when the message asks for alignment and names an approved one.
        if rules.is_alignment_request(state["request_text"]):
            listed = call("list_frameworks")
            framework_id = match_framework(state["request_text"], listed.get("frameworks") or []) if _ok(listed) else None
            if framework_id:
                slots["framework_id"] = framework_id
            else:
                notes.append(_t(language, "No approved framework document matches the request, so alignment is "
                                          "unable to determine.",
                                "لا توجد وثيقة إطار معتمدة تطابق الطلب، لذا يتعذر تحديد المواءمة."))
        return {"slots": slots, "academy_evidence": evidence, "related_courses": related[:3],
                "data_status": sorted(statuses), "notes": notes}

    def essentials(state: CurriculumState) -> str:
        return "clarify" if _missing(state["slots"]) else "research"

    def _missing(slots: dict[str, Any]) -> list[str]:
        missing = []
        if not slots.get("topic"):
            missing.append("topic")
        hours = slots.get("total_hours")
        if not isinstance(hours, int) or not 1 <= hours <= MAX_HOURS:
            missing.append("total_hours")
        if not slots.get("level") and not slots.get("defaults_accepted"):
            missing.append("level")
        return missing

    def clarify(state: CurriculumState) -> dict[str, Any]:
        language, slots = state["language"], state["slots"]
        missing = _missing(slots)
        questions = {
            "topic": _t(language, "What topic (or existing course) should the curriculum cover?",
                        "ما موضوع المنهج (أو المقرر الموجود) المطلوب؟"),
            "total_hours": _t(language, "How many training hours in total?", "كم عدد ساعات التدريب الإجمالية؟"),
            "level": _t(language, "Which level: Beginner, Intermediate or Advanced?",
                        "ما المستوى: مبتدئ أم متوسط أم متقدم؟"),
        }
        known = []
        if slots.get("topic"):
            known.append(_t(language, f"Topic: {slots['topic']}", f"الموضوع: {slots['topic']}"))
        if isinstance(slots.get("total_hours"), int):
            if not 1 <= slots["total_hours"] <= MAX_HOURS:
                known.append(_t(language, f"{slots['total_hours']} hours is outside 1–{MAX_HOURS}.",
                                f"عدد الساعات {slots['total_hours']} خارج النطاق 1–{MAX_HOURS}."))
            else:
                known.append(_t(language, f"Total hours: {slots['total_hours']}",
                                f"الساعات: {slots['total_hours']}"))
        if slots.get("level"):
            known.append(_t(language, f"Level: {slots['level']}", f"المستوى: {LEVEL_AR.get(slots['level'])}"))
        if slots.get("hours_rate_ignored"):
            known.append(_t(language, "Hours per week/day were given; I need the total.",
                            "ذُكرت ساعات أسبوعية/يومية؛ أحتاج العدد الإجمالي."))
        lines = [_t(language, "To draft this curriculum I need:", "لإعداد مسودة هذا المنهج أحتاج إلى:")]
        lines += [f"- {questions[m]}" for m in missing]
        if known:
            lines += ["", _t(language, "Already known:", "معلومات معروفة:")] + [f"- {k}" for k in known]
        if "level" in missing:
            lines += ["", _t(language, "You can also reply \"use defaults\" to use the Beginner level.",
                             "يمكنك أيضًا الرد بـ «استخدم الافتراضي» لاعتماد المستوى المبتدئ.")]
        memory_out = dict(state.get("memory") or {})
        memory_out["pending"] = {"slots": dict(slots), "missing": missing,
                                 "output_language": state["output_language"], "request": state["request_text"]}
        memory_out["language"] = language
        return {"status": "success", "memory_out": memory_out, "missing": missing,
                "response": {"kind": "question", "message": "\n".join(lines), "missing": missing}}

    # ---------------------------------------------------------- research

    def research(state: CurriculumState) -> dict[str, Any]:
        slots = state["slots"]
        query = f"{slots['topic']} skills curriculum {slots.get('level') or ''} training".replace("  ", " ")
        result = call("search_curriculum_sources", query=query, max_results=MAX_SOURCES)
        if _ok(result):
            return {"sources": (result.get("sources") or [])[:MAX_SOURCES], "research_status": "ok"}
        return {"sources": [], "research_status": result.get("status", "error")}

    # ---------------------------------------------------------- generate

    def generate(state: CurriculumState) -> dict[str, Any]:
        slots, language = state["slots"], state["output_language"]
        hours = int(slots["total_hours"])
        level = slots.get("level") or "Beginner"
        audience = slots.get("target_audience")
        revision_block = ""
        if state.get("revision", {}).get("instruction"):
            revision_block = (f"\n{_summary_for_revision(state['base']['content'])}\n"
                              f"Requested change: {state['revision']['instruction']}\n"
                              "Keep everything else as close to the current curriculum as possible.\n")
        values = {"language_name": prompts.LANGUAGE_NAMES[language], "verb_examples": prompts.VERB_EXAMPLES[language]}
        outline = ask(CurriculumOutline, prompts.OUTLINE, "outline", **values, topic=slots["topic"], level=level,
                      audience=audience or "not stated: propose a suitable audience for this level",
                      total_hours=hours, revision_block=revision_block,
                      academy_block=_academy_block(state), sources_block=_sources_block(state.get("sources") or []))
        mapping = _clean_outline(outline, level, hours)
        skeleton = {"learning_outcomes": [o.model_dump() for o in outline.learning_outcomes],
                    "modules": [m.model_dump() for m in outline.modules]}
        common = {**values, "title": outline.title, "level": outline.level, "total_hours": hours,
                  "audience": outline.target_audience or audience or "", "outcomes_block": _outcomes_block(skeleton),
                  "modules_block": _modules_block(skeleton)}
        module_details = ask(ModuleDetailList, prompts.MODULE_DETAILS, "module details", **common)
        capstone_rule = "" if slots.get("capstone") is not False else (
            "; the administrator asked for NO capstone: leave capstone empty and give capstone_omitted_reason")
        extras = ask(CurriculumExtras, prompts.EXTRAS, "assessments", **common, capstone_rule=capstone_rule,
                     sources_block=_sources_block(state.get("sources") or []))

        outcome_ids = {o.id for o in outline.learning_outcomes}
        module_ids = [m.id for m in outline.modules]
        details = []
        by_id = {}
        for position, detail in enumerate(module_details.modules):
            module_id = mapping.get(f"module:{detail.module_id}", detail.module_id)
            if module_id not in module_ids and position < len(module_ids):
                module_id = module_ids[position]  # the model renamed the id: keep the order
            detail.module_id = module_id
            detail.outcome_ids = _map_refs(detail.outcome_ids, mapping, outcome_ids)
            by_id.setdefault(module_id, detail)
        details = list(by_id.values())
        for assessment in extras.assessments:
            assessment.outcome_ids = _map_refs(assessment.outcome_ids, mapping, outcome_ids)
        _renumber(extras.assessments, "A")
        if extras.capstone:
            extras.capstone.outcome_ids = _map_refs(extras.capstone.outcome_ids, mapping, outcome_ids)
        if slots.get("capstone") is False:
            extras.capstone = None
            extras.capstone_omitted_reason = extras.capstone_omitted_reason or _t(
                language, "Not included at the administrator's request.", "لم يُضمَّن بناءً على طلب المسؤول.")
        source_ids = {s["source_id"] for s in state.get("sources") or []}
        repairs = list(state.get("repairs") or [])
        dropped = 0
        for competency in extras.competencies:
            kept = [sid for sid in competency.source_ids if sid in source_ids]
            dropped += len(competency.source_ids) - len(kept)
            competency.source_ids = kept
        # With no research there is nothing the model could cite; ids it
        # wrote anyway are cleared without a note (the assumptions already
        # say no sources were used). With research, removals are reported.
        if dropped and source_ids:
            repairs.append(_t(state["language"], f"Removed {dropped} citation(s) to sources that were not retrieved.",
                              f"حُذفت {dropped} إحالة إلى مصادر لم تُسترجع."))
        if extras.capstone and extras.capstone_omitted_reason:
            extras.capstone_omitted_reason = None

        draft = assemble_curriculum(
            outline, CurriculumDetails(modules=details, **extras.model_dump()),
            language=language, total_hours=hours, sources=state.get("sources") or [],
            academy_evidence=state.get("academy_evidence") or [], data_notice=_data_notice(state, language),
            extra_assumptions=_assumptions(state, language))
        draft["data_status"] = list(state.get("data_status") or [])
        return {"draft": draft, "repairs": repairs}

    def _data_notice(state: CurriculumState, language: str) -> Optional[str]:
        statuses = state.get("data_status") or []
        if not statuses:
            return None
        return _t(language, f"Academy catalogue data used here is labelled \"{', '.join(statuses)}\"; it is not "
                            "verified academy information.",
                  f"بيانات كتالوج الأكاديمية المستخدمة هنا موسومة بأنها «{', '.join(statuses)}» وليست معلومات "
                  "أكاديمية موثقة.")

    def _assumptions(state: CurriculumState, language: str) -> list[str]:
        slots = state["slots"]
        out = []
        if not slots.get("level") and slots.get("defaults_accepted"):
            out.append(_t(language, "Level not stated: the default Beginner level was used.",
                          "لم يُحدَّد المستوى: استُخدم المستوى المبتدئ افتراضيًا."))
        if not slots.get("target_audience"):
            out.append(_t(language, "The target audience was not stated; the audience shown is proposed.",
                          "لم يُحدَّد الجمهور المستهدف؛ الجمهور المذكور مقترح."))
        if not state.get("sources"):
            out.append(_t(language, "No external research sources were used (research is not configured or "
                                    "returned nothing); industry relevance is the model's proposal.",
                          "لم تُستخدم مصادر بحث خارجية (البحث غير مفعّل أو بلا نتائج)؛ الصلة بسوق العمل اقتراح من "
                          "النموذج."))
        if slots.get("origin") == "suggestion":
            out.append(_t(language, "Based on a catalogue suggestion; its duration was proposed by the model.",
                          "مبني على اقتراح من تحليل الكتالوج؛ مدته مقترحة من النموذج."))
        return out

    # ---------------------------------------------------------- validate

    def validate(state: CurriculumState) -> dict[str, Any]:
        draft = state["draft"]
        result = call("validate_curriculum", curriculum=draft, requested_hours=draft["total_hours"],
                      framework_id=_framework_of(draft))
        if not _ok(result):
            return {"validation": {"valid": False, "errors": [{"code": "tool_error", "path": "",
                                                               "message": result.get("message", "")}],
                                   "warnings": []}}
        return {"validation": {k: result[k] for k in ("valid", "errors", "warnings")}}

    def after_validate(state: CurriculumState) -> str:
        if state["validation"]["valid"] or state.get("repair_attempts", 0) >= 1:
            # A generation request that named an approved framework is
            # aligned once the content is final (R15: ... validate -> align -> save).
            if (state.get("slots") or {}).get("framework_id") and not state["draft"].get("alignment"):
                return "attach_alignment"
            return "save"
        return "repair"

    def _judge(template, **values) -> AlignmentJudgement:
        return ask(AlignmentJudgement, template, "alignment", **values)

    def attach_alignment(state: CurriculumState) -> dict[str, Any]:
        draft = dict(state["draft"])
        try:
            draft["alignment"] = run_alignment(draft, state["slots"]["framework_id"], call, _judge, state["language"])
        except AlignmentError as exc:
            notes = list(state.get("notes") or []) + [_t(state["language"], f"Alignment was not performed: {exc}",
                                                         f"لم تُجرَ المواءمة: {exc}")]
            return {"notes": notes}
        return {"draft": draft}

    def repair(state: CurriculumState) -> dict[str, Any]:
        """One model pass on the listed errors, then deterministic hour and reference fixes."""
        draft = json.loads(json.dumps(state["draft"]))
        language = draft["language"]
        errors = state["validation"]["errors"]
        repairs = list(state.get("repairs") or [])
        content_errors = [e for e in errors if e["code"] not in ("hours_total", "module_hours")]
        if content_errors:
            compact = {k: draft.get(k) for k in ("title", "level", "learning_outcomes", "assessments", "capstone",
                                                  "tools_resources", "competencies")}
            compact["modules"] = [{k: m.get(k) for k in ("id", "title", "topics", "outcome_ids")} for m in draft["modules"]]
            try:
                patch = ask(RepairPatch, prompts.REPAIR, "repair", language_name=prompts.LANGUAGE_NAMES[language],
                            verb_examples=prompts.VERB_EXAMPLES[language],
                            errors_block="\n".join(f"- [{e['code']}] {e['path']}: {e['message']}" for e in content_errors),
                            curriculum_json=json.dumps(compact, ensure_ascii=False))
                _apply_patch(draft, patch)
                codes = ", ".join(sorted({e["code"] for e in content_errors}))
                repairs.append(_t(state["language"], f"One automatic repair pass was applied to: {codes}.",
                                  f"طُبّقت جولة إصلاح تلقائية واحدة على: {codes}."))
            except CurriculumModelError as exc:
                repairs.append(_t(state["language"], f"The automatic repair pass failed ({exc}).",
                                  f"فشلت جولة الإصلاح التلقائي ({exc})."))
        modules = draft.get("modules") or []
        if modules and sum(m.get("hours") or 0 for m in modules) != draft["total_hours"] \
                and draft["total_hours"] >= len(modules):
            before = [m.get("hours") for m in modules]
            for module, share in zip(modules, rebalance_hours(modules, draft["total_hours"])):
                module["hours"] = share
            note = _t(language,
                      f"Module hours were rescaled proportionally to keep the requested {draft['total_hours']} hours "
                      f"(model proposed {before}).",
                      f"أعيد توزيع ساعات الوحدات بالتناسب للحفاظ على {draft['total_hours']} ساعة المطلوبة "
                      f"(اقترح النموذج {before}).")
            draft.setdefault("assumptions", []).append(note)
            repairs.append(_t(state["language"], "Module hours rescaled to the requested total.",
                              "أعيد توزيع ساعات الوحدات لتطابق الإجمالي المطلوب."))
        ids = {o.get("id") for o in draft.get("learning_outcomes") or []}
        for part in [*modules, *draft.get("assessments", []),
                     *([draft["capstone"]] if isinstance(draft.get("capstone"), dict) else [])]:
            part["outcome_ids"] = [r for r in part.get("outcome_ids") or [] if r in ids]
        return {"draft": draft, "repairs": repairs, "repair_attempts": state.get("repair_attempts", 0) + 1}

    def _apply_patch(draft: dict[str, Any], patch: RepairPatch) -> None:
        known_ids = {o.get("id") for o in draft.get("learning_outcomes") or []}
        if patch.learning_outcomes:
            by_id = {o.id: o.statement for o in patch.learning_outcomes if o.statement.strip()}
            for outcome in draft["learning_outcomes"]:
                outcome["statement"] = by_id.get(outcome["id"], outcome["statement"])
        links = {link.module_id: [r for r in link.outcome_ids if r in known_ids] for link in patch.module_links}
        for module in draft["modules"]:
            if links.get(module["id"]):
                module["outcome_ids"] = links[module["id"]]
        if patch.assessments:
            draft["assessments"] = [a.model_dump() for a in patch.assessments]
            for index, assessment in enumerate(draft["assessments"], start=1):
                assessment["id"] = f"A{index}"
        if patch.capstone and draft.get("capstone_omitted_reason") is None:
            draft["capstone"] = patch.capstone.model_dump()
        if patch.tools_resources:
            draft["tools_resources"] = list(patch.tools_resources)
        if patch.competencies:
            known_sources = {r.get("source_id") for r in draft.get("references") or []}
            draft["competencies"] = [{"name": c.name, "source_ids": [s for s in c.source_ids if s in known_sources]}
                                     for c in patch.competencies]

    # -------------------------------------------------------------- save

    def save(state: CurriculumState) -> dict[str, Any]:
        draft = state["draft"]
        base = state.get("base")
        slots = state.get("slots") or {}
        saved = call("save_curriculum_draft", curriculum=draft, sources=draft.get("references") or [],
                     curriculum_id=base["curriculum_id"] if base else None,
                     change_summary=state.get("change_summary") or ("Generated from the request." if not base else None),
                     alignment=draft.get("alignment"), framework_id=_framework_of(draft),
                     topic=slots.get("topic") or draft.get("title"), origin=slots.get("origin") or "request",
                     source_request=state["request_text"])
        if not _ok(saved):
            return {"status": "error", "saved": None, "response": {"kind": "error", "message": _t(
                state["language"], f"The curriculum could not be saved: {saved.get('message')}",
                f"تعذر حفظ المنهج: {saved.get('message')}")}}
        return {"saved": saved, "validation": saved["validation"]}

    def after_save(state: CurriculumState) -> str:
        return "render_pdf" if state.get("saved") else "finish"

    def render_pdf(state: CurriculumState) -> dict[str, Any]:
        saved = state["saved"]
        result = call("render_curriculum_pdf", curriculum_id=saved["curriculum_id"], version=saved["version"])
        if _ok(result):
            return {"document": result["document"], "document_status": "ok"}
        return {"document": None, "document_status": result.get("message") or result.get("status")}

    def curriculum_reply(state: CurriculumState) -> dict[str, Any]:
        saved, draft, language = state["saved"], state["draft"], state["language"]
        memory_out = dict(state.get("memory") or {})
        memory_out.update({"language": language, "active_curriculum_id": saved["curriculum_id"],
                           "active_version": saved["version"], "active_language": draft["language"],
                           "active_level": draft.get("level"), "pending": None})
        if state.get("intent") == "align":
            header = _t(language, "Alignment added", "أضيفت المواءمة")
        elif state.get("base"):
            header = _t(language, "Curriculum revised", "تم تعديل المنهج")
        else:
            header = _t(language, "Curriculum drafted", "تم إعداد مسودة المنهج")
        message = _curriculum_message(draft, saved["curriculum_id"], saved["version"], saved["review_status"],
                                      state["validation"], language, header=header,
                                      change=state.get("change_summary") if state.get("base") else None,
                                      notes=state.get("notes") or [], repairs=state.get("repairs") or [],
                                      document=state.get("document"), document_status=state.get("document_status"))
        return {"status": "success", "memory_out": memory_out, "response": {
            "kind": "curriculum", "message": message, "curriculum_id": saved["curriculum_id"],
            "version": saved["version"], "review_status": saved["review_status"], "validation": state["validation"],
            "document": state.get("document"), "curriculum": draft}}

    # ------------------------------------------------------------ revise

    def revise(state: CurriculumState) -> dict[str, Any]:
        base, language = state["base"], state["language"]
        content = base["content"]
        changes = state["revision"]["changes"]
        if all(c["kind"] == "no_change" for c in changes):
            return {"status": "success", "memory_out": dict(state.get("memory") or {}), "response": {
                "kind": "no_change", "message": _t(language, "The curriculum already has a capstone project; nothing "
                                                             "was changed.",
                                                   "يتضمن المنهج مشروعًا ختاميًا بالفعل؛ لم يتغير شيء.")}}
        changes = [c for c in changes if c["kind"] != "no_change"]
        hours_change = next((c["value"] for c in changes if c["kind"] == "set_hours"), None)
        summaries: list[str] = []
        try:
            if any(c["kind"] in REGENERATE for c in changes):
                level = next((c["value"] for c in changes if c["kind"] == "set_level"), content.get("level"))
                out_language = next((c["value"] for c in changes if c["kind"] == "set_language"), content["language"])
                audience = next((c["value"] for c in changes if c["kind"] == "set_audience"),
                                content.get("target_audience"))
                instruction_parts = []
                for change in changes:
                    if change["kind"] == "set_level":
                        instruction_parts.append(f"Rewrite it for the {change['value']} level.")
                    elif change["kind"] == "set_language":
                        instruction_parts.append(f"Write it in {prompts.LANGUAGE_NAMES[change['value']]}.")
                    elif change["kind"] == "instruction":
                        instruction_parts.append(change["value"])
                    else:
                        instruction_parts.append(_describe(change, "en"))
                instruction = " ".join(instruction_parts)
                summary = instruction + (" Framework alignment cleared because the content changed; ask to align "
                                         "it again." if content.get("alignment") else "")
                slots = {"topic": content.get("title"), "level": level, "target_audience": audience,
                         "total_hours": hours_change or content["total_hours"], "origin": "request",
                         "capstone": False if any(c["kind"] == "remove_capstone" for c in changes) else None}
                return {"slots": slots, "output_language": out_language, "change_summary": summary,
                        "revision": {"changes": changes, "instruction": instruction},
                        "academy_evidence": content.get("academy_evidence") or [],
                        "sources": content.get("references") or [], "research_status": "reused",
                        "data_status": content.get("data_status") or [], "regenerate": True}
            revised = content
            for change in changes:
                if change["kind"] in DETERMINISTIC and change["kind"] != "set_hours":
                    revised, summary = apply_change(revised, change)
                    summaries.append(summary)
            for change in changes:
                if change["kind"] == "add_module":
                    new = ask(NewModule, prompts.NEW_MODULE, "new module",
                              language_name=prompts.LANGUAGE_NAMES[revised["language"]], title=revised["title"],
                              level=revised.get("level"), audience=revised.get("target_audience"),
                              module_titles=", ".join(m["title"] for m in revised["modules"]),
                              outcomes_block=_outcomes_block(revised), subject=change["value"])
                    revised = add_module(revised, {"title": new.title, "topics": new.topics,
                                                   "description": new.description,
                                                   "activities": [a.model_dump() for a in new.activities],
                                                   "outcome_ids": new.outcome_ids}, hours_change)
                    summaries.append(f"Module added: {new.title}.")
                elif change["kind"] == "add_capstone":
                    capstone = ask(CapstoneItem, prompts.CAPSTONE, "capstone",
                                   language_name=prompts.LANGUAGE_NAMES[revised["language"]], title=revised["title"],
                                   level=revised.get("level"), audience=revised.get("target_audience"),
                                   module_titles=", ".join(m["title"] for m in revised["modules"]),
                                   outcomes_block=_outcomes_block(revised), instruction="Write the capstone project.")
                    revised = json.loads(json.dumps(revised))
                    known = {o["id"] for o in revised["learning_outcomes"]}
                    revised["capstone"] = {**capstone.model_dump(),
                                           "outcome_ids": [r for r in capstone.outcome_ids if r in known]}
                    revised["capstone_omitted_reason"] = None
                    summaries.append("Capstone project added.")
            if hours_change is not None and not any(c["kind"] == "add_module" for c in changes):
                revised, summary = apply_change(revised, {"kind": "set_hours", "value": hours_change})
                summaries.append(summary)
        except RevisionError as exc:
            return {"status": "error", "response": {"kind": "error", "message": _t(
                language, f"I could not apply that change: {exc}", f"تعذر تطبيق هذا التعديل: {exc}")}}
        if revised.get("alignment"):
            # The comparison described the old content: it must be run again.
            revised = {**revised, "alignment": None}
            summaries.append("Framework alignment cleared because the content changed; ask to align it again.")
        return {"draft": revised, "change_summary": " ".join(summaries), "regenerate": False,
                "slots": {"topic": content.get("title"), "origin": "request"}}

    def after_revise(state: CurriculumState) -> str:
        if state.get("status") == "error" or state.get("response"):
            return "finish"
        return "generate" if state.get("regenerate") else "validate"

    def _describe(change: dict[str, Any], language: str) -> str:
        kind = change["kind"]
        if kind == "set_audience":
            return _t(language, f"Target the audience: {change['value']}.", f"الجمهور المستهدف: {change['value']}.")
        if kind == "set_hours":
            return _t(language, f"Use {change['value']} hours in total.", f"إجمالي الساعات {change['value']}.")
        if kind == "remove_module":
            return _t(language, f"Remove module {change['module_id']}.", f"احذف الوحدة {change['module_id']}.")
        if kind == "remove_capstone":
            return _t(language, "No capstone project.", "بدون مشروع ختامي.")
        if kind == "add_module":
            return _t(language, f"Add a module about {change['value']}.", f"أضف وحدة عن {change['value']}.")
        if kind == "add_capstone":
            return _t(language, "Add a capstone project.", "أضف مشروعًا ختاميًا.")
        return str(change.get("value") or kind)

    # ----------------------------------------------------------- suggest

    def suggest(state: CurriculumState) -> dict[str, Any]:
        language, text = state["language"], state["request_text"]
        listed = call("list_academy_courses", limit=200)
        if not _ok(listed):
            return _error_reply(state, listed)
        courses = listed["courses"]
        norm = normalize_for_matching(text)
        departments = sorted({c.get("department") for c in courses if c.get("department")})
        department = next((d for d in departments if normalize_for_matching(d) in norm), None)
        report = call("catalogue_gap_report", department=department)
        if not _ok(report):
            return _error_reply(state, report)
        gaps = report.get("gaps") or []
        data_status = ", ".join(report.get("data_status") or []) or "of unknown status"
        research = call("search_curriculum_sources",
                        query=f"in-demand skills new training courses {department or 'technology'}", max_results=MAX_SOURCES)
        sources = (research.get("sources") or [])[:MAX_SOURCES] if _ok(research) else []
        scoped = [c for c in courses if not department or c.get("department") == department]
        courses_block = "\n".join(f"- {c['course_id']} {c.get('name')} ({c.get('department')}, {c.get('level')}, "
                                  f"{c.get('total_hours') or '?'} h)" for c in scoped) or "None."
        gaps_block = "\n".join(f"- {g['id']} [{g['type']}] {g['detail']}" for g in gaps) or "None found."
        try:
            answer = ask(SuggestionList, prompts.SUGGEST, "suggestions",
                         language_name=prompts.LANGUAGE_NAMES[language], max_suggestions=MAX_SUGGESTIONS,
                         data_status=data_status, request=text, courses_block=courses_block, gaps_block=gaps_block,
                         sources_block=_sources_block(sources))
        except CurriculumModelError as exc:
            return _model_error_reply(state, exc)
        gap_ids = {g["id"]: g for g in gaps}
        source_ids = {s["source_id"]: s for s in sources}
        course_ids = {c["course_id"] for c in courses}
        suggestions = []
        for item in answer.suggestions[:MAX_SUGGESTIONS]:
            entry = item.model_dump()
            entry["gap_ids"] = [g for g in item.gap_ids if g in gap_ids]
            entry["source_ids"] = [s for s in item.source_ids if s in source_ids]
            entry["prerequisite_course_ids"] = [c for c in item.prerequisite_course_ids if c in course_ids]
            entry["level"] = _level(item.level, None)
            entry["proposed_hours"] = item.proposed_hours if 1 <= (item.proposed_hours or 0) <= MAX_HOURS else None
            entry["basis"] = ("catalogue_gap" if entry["gap_ids"] else "research" if entry["source_ids"]
                              else "model_proposal")
            if entry["basis"] == "model_proposal":
                entry["confidence"] = "low"
            overlap = call("check_course_overlap", title=item.title, topics=item.modules_outline,
                           outcomes=item.outcomes)
            entry["overlaps"] = [o for o in (overlap.get("overlaps") or [])[:2]
                                 if o["name_similarity"] >= 0.5 or o["term_coverage"] >= 0.4] if _ok(overlap) else []
            entry["duplicate"] = bool(overlap.get("duplicate")) if _ok(overlap) else False
            suggestions.append(entry)
        suggestions.sort(key=lambda s: s["duplicate"])  # likely duplicates last
        memory_out = dict(state.get("memory") or {})
        memory_out.update({"language": language, "last_suggestions": suggestions, "pending": None})
        message = _suggestions_message(suggestions, gaps, sources, report, research, language)
        return {"status": "success", "memory_out": memory_out, "gap_report": report, "suggestions": suggestions,
                "response": {"kind": "suggestions", "message": message, "suggestions": suggestions,
                             "gaps": gaps, "data_status": report.get("data_status") or [],
                             "research_status": research.get("status")}}

    # -------------------------------------------------------------- show

    def show(state: CurriculumState) -> dict[str, Any]:
        language, text = state["language"], state["request_text"]
        memory = state.get("memory") or {}
        curriculum_id = rules.find_curriculum_id(text) or (None if rules.is_list_request(text)
                                                            else memory.get("active_curriculum_id"))
        if not curriculum_id:
            listed = call("list_curricula", limit=10)
            if not _ok(listed):
                return _error_reply(state, listed)
            rows = listed["curricula"]
            if not rows:
                message = _t(language, "No curricula have been saved yet.", "لم تُحفظ أي مناهج بعد.")
            else:
                lines = [_t(language, "Saved curricula (latest first):", "المناهج المحفوظة (الأحدث أولًا):")]
                for row in rows:
                    label = STATUS_LABELS.get(row["status"], (row["status"], row["status"]))
                    lines.append(f"- {row['curriculum_id']} · {row['title']} · v{row['current_version']} · "
                                 f"{_t(language, label[0], label[1])}")
                message = "\n".join(lines)
            return {"status": "success", "memory_out": dict(memory),
                    "response": {"kind": "list", "message": message, "curricula": rows}}
        loaded = call("get_curriculum", curriculum_id=curriculum_id)
        if not _ok(loaded):
            return _error_reply(state, loaded)
        record = loaded["curriculum"]
        documents = [d for d in loaded.get("documents") or [] if d.get("version") == record["version"]]
        message = _curriculum_message(record["content"], record["curriculum_id"], record["version"], record["status"],
                                      record.get("validation") or {"valid": True, "errors": []}, language,
                                      header=_t(language, "Stored curriculum", "المنهج المحفوظ"),
                                      document=documents[-1] if documents else None,
                                      document_status="ok" if documents else _t(language, "not rendered",
                                                                               "لم يُنشأ بعد"))
        memory_out = dict(memory)
        memory_out.update({"active_curriculum_id": record["curriculum_id"], "active_version": record["version"],
                           "active_language": record["content"].get("language"),
                           "active_level": record["content"].get("level")})
        return {"status": "success", "memory_out": memory_out, "response": {
            "kind": "curriculum", "message": message, "curriculum_id": record["curriculum_id"],
            "version": record["version"], "review_status": record["status"], "curriculum": record["content"],
            "document": documents[-1] if documents else None, "versions": loaded.get("versions")}}

    # ------------------------------------------------------------- align

    def align(state: CurriculumState) -> dict[str, Any]:
        """Compare a stored curriculum with an approved framework; the result is saved as a new version."""
        language, text = state["language"], state["request_text"]
        memory = state.get("memory") or {}

        def reply(message: str, **extra: Any) -> dict[str, Any]:
            return {"status": "success", "memory_out": dict(memory),
                    "response": {"kind": "alignment", "message": message, **extra}}

        listed = call("list_frameworks")
        frameworks = (listed.get("frameworks") or []) if _ok(listed) else []
        if not frameworks:
            return reply(_t(
                language,
                "Framework alignment: unable to determine. No approved framework or accreditation documents are "
                "loaded, so there are no verified requirements to compare against. An administrator can add an "
                "official framework file (with its source and version). Curricula never claim accreditation.",
                "المواءمة مع الأطر: يتعذر التحديد. لا توجد وثائق أطر أو معايير اعتماد معتمدة محمّلة، فلا توجد متطلبات "
                "موثقة للمقارنة بها. يمكن للمسؤول إضافة ملف إطار رسمي (مع مصدره وإصداره). لا تدّعي المناهج أي اعتماد."),
                frameworks=[], alignment_status="unable_to_determine")
        framework_id = match_framework(text, frameworks) or (frameworks[0]["framework_id"] if len(frameworks) == 1
                                                             else None)
        names = "\n".join(f"- {f['framework_id']}: {f['name']} ({f.get('publisher')}, {f.get('version')})"
                          for f in frameworks)
        if not framework_id:
            return reply(_t(language, f"Which approved framework should I compare with?\n{names}",
                            f"مع أي إطار معتمد أقارن؟\n{names}"), frameworks=frameworks)
        curriculum_id = rules.find_curriculum_id(text) or memory.get("active_curriculum_id")
        if not curriculum_id:
            return reply(_t(language, "Which curriculum? Draft one first, or name its id (CUR-...).",
                            "أي منهج؟ أنشئ منهجًا أولًا أو اذكر معرّفه (CUR-...)."))
        loaded = call("get_curriculum", curriculum_id=curriculum_id)
        if not _ok(loaded):
            return _error_reply(state, loaded)
        record = loaded["curriculum"]
        content = json.loads(json.dumps(record["content"]))
        try:
            content["alignment"] = run_alignment(content, framework_id, call, _judge, language)
        except AlignmentError as exc:
            return reply(_t(language, f"Alignment could not be performed: {exc}", f"تعذر إجراء المواءمة: {exc}"))
        name = content["alignment"]["framework"]["name"]
        return {"draft": content, "base": record, "repair_attempts": 1,
                "slots": {"topic": content.get("title"), "origin": "request", "framework_id": framework_id},
                "change_summary": f"Alignment with {name} ({framework_id}) added; content unchanged."}

    def after_align(state: CurriculumState) -> str:
        return "finish" if state.get("response") or state.get("status") == "error" else "validate"

    # ------------------------------------------------------ simple replies

    def simple_reply(state: CurriculumState) -> dict[str, Any]:
        language, intent = state["language"], state["intent"]
        memory = dict(state.get("memory") or {})
        if intent == "approve_attempt":
            message = _t(language,
                         "I can't approve curricula. Only an administrator can approve a reviewed version, in the "
                         "admin dashboard's Curricula tab. I can revise the draft or show its current status.",
                         "لا أستطيع اعتماد المناهج. يعتمد المسؤول فقط النسخة بعد مراجعتها من تبويب «المناهج» في لوحة "
                         "الإدارة. يمكنني تعديل المسودة أو عرض حالتها الحالية.")
            if memory.get("active_curriculum_id"):
                message += "\n" + _t(language, f"Current curriculum: {memory['active_curriculum_id']} "
                                               f"(version {memory.get('active_version')}).",
                                     f"المنهج الحالي: {memory['active_curriculum_id']} "
                                     f"(النسخة {memory.get('active_version')}).")
        elif intent == "unsupported":
            message = _t(language, "I can't send or share curricula. Download the PDF from the Curricula tab instead.",
                         "لا أستطيع إرسال المناهج أو مشاركتها. يمكنك تنزيل ملف PDF من تبويب «المناهج».")
        else:
            notes = state.get("notes") or []
            message = "\n".join(notes + [_t(
                language,
                "I'm the Curriculum Agent. I can:\n- draft a curriculum: \"Create a 40-hour beginner curriculum on "
                "Python for data analysis\"\n- revise the current draft: \"make it 30 hours\", \"add a module on "
                "Git\", \"translate it to Arabic\"\n- suggest new courses from the catalogue: \"Suggest new courses "
                "for the Data Analytics department\"\n- show saved curricula: \"list curricula\"",
                "أنا وكيل المناهج. يمكنني:\n- إعداد منهج: «أنشئ منهج عن تحليل البيانات للمبتدئين لمدة 40 ساعة»\n"
                "- تعديل المسودة الحالية: «اجعلها 30 ساعة»، «أضف وحدة عن Git»، «ترجمه إلى الإنجليزية»\n"
                "- اقتراح دورات جديدة من الكتالوج: «اقترح دورات جديدة لقسم تحليل البيانات»\n"
                "- عرض المناهج المحفوظة: «اعرض المناهج»")])
        # A refused action is never reported as a success (approval, sending).
        status = "error" if intent in ("approve_attempt", "unsupported") else "success"
        return {"status": status, "memory_out": memory, "response": {"kind": intent, "message": message}}

    def _error_reply(state: CurriculumState, result: dict[str, Any]) -> dict[str, Any]:
        return {"status": "error", "memory_out": dict(state.get("memory") or {}), "response": {
            "kind": "error", "message": _t(state["language"], f"The request could not be completed: "
                                                              f"{result.get('message') or result.get('status')}",
                                           f"تعذر إكمال الطلب: {result.get('message') or result.get('status')}")}}

    def _model_error_reply(state: CurriculumState, exc: Exception) -> dict[str, Any]:
        return {"status": "error", "memory_out": dict(state.get("memory") or {}), "response": {
            "kind": "error", "message": _t(state["language"],
                                           f"The language model is not available or returned unusable output ({exc}). "
                                           "Nothing was saved.",
                                           f"نموذج اللغة غير متاح أو أعاد ناتجًا غير صالح ({exc}). لم يُحفظ شيء.")}}

    def finish(state: CurriculumState) -> dict[str, Any]:
        return {}

    # Nodes that call the model fail as a clear error reply, never a crash.
    def guarded(node: Callable[[CurriculumState], dict[str, Any]]):
        def run(state: CurriculumState) -> dict[str, Any]:
            if state.get("status") == "error":
                return {}
            try:
                return node(state)
            except CurriculumModelError as exc:
                return _model_error_reply(state, exc)
        run.__name__ = node.__name__
        return run

    def unless_error(target: str) -> Callable[[CurriculumState], str]:
        return lambda state: "finish" if state.get("status") == "error" else target

    graph = StateGraph(CurriculumState)
    for name, node in (("understand", understand), ("academy_context", academy_context), ("clarify", clarify),
                       ("research", research), ("generate", generate), ("validate", validate), ("repair", repair),
                       ("save", save), ("render_pdf", render_pdf), ("curriculum_reply", curriculum_reply),
                       ("revise", revise), ("suggest", suggest), ("show", show), ("align", align),
                       ("attach_alignment", attach_alignment), ("simple_reply", simple_reply), ("finish", finish)):
        graph.add_node(name, guarded(node))
    graph.set_entry_point("understand")
    graph.add_conditional_edges("understand", route, {
        "generate": "academy_context", "revise": "revise", "suggest": "suggest", "show": "show", "align": "align",
        "approve_attempt": "simple_reply", "unsupported": "simple_reply", "help": "simple_reply"})
    graph.add_conditional_edges("academy_context", essentials, {"clarify": "clarify", "research": "research"})
    graph.add_edge("research", "generate")
    graph.add_conditional_edges("generate", unless_error("validate"), {"validate": "validate", "finish": "finish"})
    graph.add_conditional_edges("validate", after_validate, {"save": "save", "repair": "repair",
                                                           "attach_alignment": "attach_alignment"})
    graph.add_edge("attach_alignment", "save")
    graph.add_conditional_edges("align", after_align, {"validate": "validate", "finish": "finish"})
    graph.add_edge("repair", "validate")
    graph.add_conditional_edges("save", after_save, {"render_pdf": "render_pdf", "finish": "finish"})
    graph.add_edge("render_pdf", "curriculum_reply")
    graph.add_conditional_edges("revise", after_revise, {"generate": "generate", "validate": "validate",
                                                         "finish": "finish"})
    for terminal in ("clarify", "curriculum_reply", "suggest", "show", "simple_reply", "finish"):
        graph.add_edge(terminal, END)
    return graph.compile()


# ---------------------------------------------------------------------
# Reply text
# ---------------------------------------------------------------------

def _curriculum_message(curriculum: dict[str, Any], curriculum_id: str, version: int, review_status: str,
                        validation: dict[str, Any], language: str, *, header: str, change: Optional[str] = None,
                        notes: Optional[list[str]] = None, repairs: Optional[list[str]] = None,
                        document: Optional[dict[str, Any]] = None, document_status: Optional[str] = None) -> str:
    label = STATUS_LABELS.get(review_status, (review_status, review_status))
    level = curriculum.get("level")
    lines = [f"{header}: {curriculum.get('title')}",
             _t(language, f"ID: {curriculum_id} · version {version} · status: {label[0]}",
                f"المعرّف: {curriculum_id} · النسخة {version} · الحالة: {label[1]}"),
             _t(language, f"Level: {level} · Audience: {curriculum.get('target_audience')} · "
                          f"Total: {curriculum.get('total_hours')} hours",
                f"المستوى: {LEVEL_AR.get(level, level)} · الجمهور: {curriculum.get('target_audience')} · "
                f"الإجمالي: {curriculum.get('total_hours')} ساعة")]
    if change:
        lines.append(_t(language, f"Change: {change}", f"التعديل: {change}"))
    lines += ["", _t(language, "Modules:", "الوحدات:")]
    lines += [f"- {m['id']} {m['title']} — {m['hours']} {_t(language, 'h', 'ساعة')}" for m in curriculum.get("modules") or []]
    lines.append(_t(language,
                    f"Learning outcomes: {len(curriculum.get('learning_outcomes') or [])} · "
                    f"Assessments: {len(curriculum.get('assessments') or [])} · "
                    f"Capstone: {'yes' if curriculum.get('capstone') else 'no'}",
                    f"مخرجات التعلم: {len(curriculum.get('learning_outcomes') or [])} · "
                    f"التقييمات: {len(curriculum.get('assessments') or [])} · "
                    f"مشروع ختامي: {'نعم' if curriculum.get('capstone') else 'لا'}"))
    alignment = curriculum.get("alignment") or {}
    if alignment.get("items"):
        framework, summary = alignment.get("framework") or {}, alignment.get("summary") or {}
        names = {"en": {"covered": "covered", "partially_covered": "partially covered", "not_covered": "not covered",
                        "unable_to_determine": "unable to determine"},
                 "ar": {"covered": "مغطى", "partially_covered": "مغطى جزئيًا", "not_covered": "غير مغطى",
                        "unable_to_determine": "يتعذر التحديد"}}["ar" if language == "ar" else "en"]
        counts = ", ".join(f"{names[k]} {v}" for k, v in summary.items() if v)
        lines += ["", _t(language, f"Alignment with {framework.get('name')} ({framework.get('version')}): {counts}.",
                         f"المواءمة مع {framework.get('name')} ({framework.get('version')}): {counts}.")]
        gaps = [i for i in alignment["items"] if i["status"] in ("not_covered", "partially_covered")]
        if gaps:
            lines.append(_t(language, "Gaps:", "الفجوات:"))
            lines += [f"- {g['requirement_id']} ({names[g['status']]}): {g.get('recommendation') or g.get('explanation')}"
                      for g in gaps[:8]]
        lines.append(_t(language, "Alignment is an evidence-based comparison, not accreditation.",
                        "المواءمة مقارنة قائمة على الأدلة وليست اعتمادًا."))
    errors = validation.get("errors") or []
    lines.append("")
    if errors:
        lines.append(_t(language, f"Validation: {len(errors)} problem(s) remain, so it is saved as a draft:",
                        f"التحقق: تبقى {len(errors)} مشكلة، لذا حُفظ كمسودة:"))
        lines += [f"- {e['message']}" for e in errors[:8]]
    else:
        lines.append(_t(language, "Validation: passed.", "التحقق: ناجح."))
    if repairs:
        lines.append(_t(language, "Automatic fixes: ", "إصلاحات تلقائية: ") + " ".join(repairs))
    extra = [*(notes or []), *(curriculum.get("assumptions") or [])]
    if curriculum.get("data_notice"):
        extra.append(curriculum["data_notice"])
    if extra:
        lines += ["", _t(language, "Notes and assumptions:", "ملاحظات وافتراضات:")] + [f"- {n}" for n in extra]
    lines.append("")
    if document:
        lines.append(_t(language, f"PDF: {document.get('filename')} ({document.get('pages')} pages).",
                        f"ملف PDF: {document.get('filename')} ({document.get('pages')} صفحات)."))
    elif document_status:
        lines.append(_t(language, f"PDF: not available ({document_status}).", f"ملف PDF: غير متاح ({document_status})."))
    lines.append(_t(language, "Approval: only an administrator can approve it, in the Curricula tab.",
                    "الاعتماد: يعتمده المسؤول فقط من تبويب «المناهج»."))
    return "\n".join(lines)


def _suggestions_message(suggestions: list[dict[str, Any]], gaps: list[dict[str, Any]], sources: list[dict[str, Any]],
                         report: dict[str, Any], research: dict[str, Any], language: str) -> str:
    gap_by_id = {g["id"]: g for g in gaps}
    statuses = ", ".join(report.get("data_status") or []) or _t(language, "unknown", "غير معروفة")
    lines = [_t(language, f"Course suggestions from the catalogue ({report.get('course_count')} courses reviewed):",
                f"اقتراحات دورات من الكتالوج (رُوجع {report.get('course_count')} مقررًا):")]
    if not suggestions:
        lines.append(_t(language, "No suggestions could be made from the available evidence.",
                        "تعذر تقديم اقتراحات من الأدلة المتاحة."))
    for number, item in enumerate(suggestions, start=1):
        level = item.get("level") or "?"
        lines += ["", f"{number}. {item['title']} — {item.get('department')} · "
                      f"{level if language != 'ar' else LEVEL_AR.get(level, level)} · "
                      f"{item.get('proposed_hours') or '?'} {_t(language, 'h', 'ساعة')}",
                  f"   {item.get('rationale')}"]
        basis = [f"{g}: {gap_by_id[g]['detail']}" for g in item["gap_ids"]] + item["source_ids"]
        lines.append("   " + _t(language, "Basis: ", "الأساس: ") + ("; ".join(basis) if basis else _t(
            language, "model proposal (no catalogue gap or source cited)", "اقتراح من النموذج (دون فجوة أو مصدر)")))
        lines.append("   " + _t(language, f"Confidence: {item['confidence']}", f"الثقة: {item['confidence']}"))
        if item.get("overlaps"):
            names = ", ".join(f"{o['course_id']} {o.get('name')}" for o in item["overlaps"])
            lines.append("   " + _t(language, f"Possible overlap with: {names}" + (" (likely duplicate)"
                                                                                  if item["duplicate"] else ""),
                                    f"تداخل محتمل مع: {names}" + (" (غالبًا مكرر)" if item["duplicate"] else "")))
    lines += ["", _t(language, f"Catalogue data status: {statuses}. Gaps come from the catalogue data only and are not "
                               "verified academy or market needs.",
                     f"حالة بيانات الكتالوج: {statuses}. الفجوات مستخرجة من بيانات الكتالوج فقط وليست احتياجات "
                     "موثقة للأكاديمية أو السوق.")]
    if research.get("status") != "ok":
        lines.append(_t(language, "External research was not used (not configured).",
                        "لم يُستخدم البحث الخارجي (غير مفعّل)."))
    if suggestions:
        lines.append(_t(language, "Reply \"generate suggestion 1\" (or another number) to draft one.",
                        "اكتب «أنشئ الاقتراح 1» (أو رقمًا آخر) لإعداد مسودته."))
    return "\n".join(lines)
