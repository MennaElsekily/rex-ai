"""
LangGraph workflow for Agent 4.

    understand -> fetch_curriculum -> write -> save -> render -> reply

Any node can end the run early by setting "response" (a clarifying
question, or an error that says which step failed). Nothing is reported
as a success unless the document was saved and its PDF rendered; a model
failure or a failed check saves nothing.

Tools (the document MCP server or the in-process toolbox) are the only
way to read curricula and to store documents. The language model writes
only the narrative parts (generator.py); the rest is copied from the
approved curriculum (composer.py).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.curriculum_research.llm import CurriculumModelError
from agents.document_generation import understanding as rules
from agents.document_generation.composer import compose
from agents.document_generation.document_store import DOCUMENT_ID
from agents.document_generation.generator import generate_with_repair
from agents.document_generation.templates import TEMPLATES, UNSUPPORTED, field_label, type_label
from agents.document_generation.validation import clean_extra, missing_curriculum_fields

AGENT_NAME = "document_generation"
CURRICULUM_PAGE = 200       # the tool's page size limit
MAX_CURRICULUM_PAGES = 10   # at most 2000 approved curricula are read to match a name
MAX_OFFERED = 20            # choices shown in a clarifying question


class DocumentState(TypedDict, total=False):
    request_text: str
    request_id: str
    ui_language: str
    form: dict[str, Any]
    memory: dict[str, Any]
    doc_type: str
    curriculum_id: str
    language: str
    extra: dict[str, str]
    document_id: Optional[str]
    record: dict[str, Any]
    newer_version: bool
    parts: Any
    saved: dict[str, Any]
    document: dict[str, Any]
    response: dict[str, Any]
    status: str
    memory_out: Optional[dict[str, Any]]
    progress: Any   # optional callable(done, total, label): lets the admin page show how far the writing has got


def _problem_text(problems: list[dict[str, str]]) -> str:
    """The problems as one message; a problem found in one module or assessment names it ("[M2] ...")."""
    return " ".join((f"[{p['where']}] " if p.get("where") else "") + p["message"] for p in problems)


def _t(language: str, english: str, arabic: str) -> str:
    return arabic if language == "ar" else english


def build_graph(tools: Any, llm: Any):
    """Compile the workflow. ``tools``: the MCP client (or toolbox); ``llm``: a LangChain chat model."""

    def call(name: str, **kwargs: Any) -> dict[str, Any]:
        result = getattr(tools, name)(**kwargs)
        return result if isinstance(result, dict) else {"status": "error", "message": f"{name} returned nothing."}

    def stop(state: DocumentState, kind: str, message: str, *, status: str = "error",
             memory: Optional[dict[str, Any]] = None, **extra: Any) -> dict[str, Any]:
        return {"status": status, "response": {"kind": kind, "message": message, **extra}, "memory_out": memory}

    # ------------------------------------------------------- understand

    def understand(state: DocumentState) -> dict[str, Any]:
        text, form = state["request_text"], state.get("form") or {}
        memory = dict(state.get("memory") or {})
        pending = memory.get("pending") or {}
        ui = state.get("ui_language", "en")

        if form:
            doc_type, curriculum_id = form.get("document_type"), form.get("curriculum_id")
            language, raw_extra, document_id = form.get("language"), form.get("extra"), form.get("document_id")
        else:
            doc_type = rules.find_document_type(text) or pending.get("doc_type")
            curriculum_id = rules.find_curriculum_id(text) or rules.choose_from_options(text, pending.get("options") or [])
            language = rules.find_language(text) or pending.get("language")
            raw_extra, document_id = pending.get("extra"), None

        if not doc_type:
            options = ", ".join(f"{type_label(t, ui)} ({t})" for t in TEMPLATES)
            return stop(state, "clarify", _t(ui, f"Which document should I generate? Available: {options}.",
                                             f"أي وثيقة تريد إنشاءها؟ المتاح: {options}."),
                        status="success", memory={"pending": {"doc_type": None, "language": language}})
        if doc_type in UNSUPPORTED:
            return stop(state, "unsupported", UNSUPPORTED[doc_type]["reason"])
        if doc_type not in TEMPLATES:
            return stop(state, "unsupported", _t(ui, f"Unknown document type: {doc_type}.", f"نوع وثيقة غير معروف: {doc_type}."))
        if language not in (None, "ar", "en"):
            return stop(state, "invalid", _t(ui, "The language must be English (en) or Arabic (ar).",
                                             "يجب أن تكون اللغة الإنجليزية (en) أو العربية (ar)."))
        if document_id is not None and not (isinstance(document_id, str) and DOCUMENT_ID.match(document_id)):
            return stop(state, "invalid", _t(ui, "That is not a valid document id.", "معرّف الوثيقة غير صالح."))
        extra, problems = clean_extra(raw_extra)
        if problems:
            return stop(state, "invalid", " ".join(p["message"] for p in problems))

        update = {"doc_type": doc_type, "language": language, "extra": extra, "document_id": document_id}
        if curriculum_id:
            return {**update, "curriculum_id": curriculum_id}

        # No curriculum named: offer the approved ones, or pick the one the request describes.
        approved: list[dict[str, Any]] = []
        for _ in range(MAX_CURRICULUM_PAGES):   # every approved curriculum, not just the first page
            listed = call("list_approved_curricula", limit=CURRICULUM_PAGE, offset=len(approved))
            if listed.get("status") != "ok":
                return stop(state, "error", listed.get("message") or "The approved curricula could not be listed.")
            approved += listed.get("curricula") or []
            if not listed.get("has_more"):
                break
        if not approved:
            return stop(state, "not_approved", _t(
                ui, "No curriculum is approved yet. Approve one in the Curricula tab first: documents are only "
                    "generated from approved curricula.",
                "لا يوجد منهج معتمد بعد. اعتمد منهجًا في تبويب المناهج أولًا: لا تُنشأ الوثائق إلا من مناهج معتمدة."))
        matches = [] if rules.find_bad_reference(text) else rules.match_curricula(text, approved)
        if len(matches) == 1:
            return {**update, "curriculum_id": matches[0]["curriculum_id"]}
        offered_all = matches or approved
        offered = offered_all[:MAX_OFFERED]
        bad = rules.find_bad_reference(text)
        lines = "\n".join(f"{i}. {c['title']} — {c['curriculum_id']} (v{c['approved_version']})"
                          for i, c in enumerate(offered, start=1))
        intro = _t(ui, f"“{bad}” is not a curriculum id (they look like CUR-0123456789ab). " if bad else "",
                   f"«{bad}» ليس معرّف منهج (يبدو مثل CUR-0123456789ab). " if bad else "")
        question = _t(ui, "Which approved curriculum should the document use? Reply with its number or id:\n",
                      "أي منهج معتمد تريد استخدامه؟ أجب برقمه أو معرّفه:\n")
        if len(offered_all) > len(offered):
            more = len(offered_all) - len(offered)
            lines += "\n" + _t(ui, f"… and {more} more: reply with a curriculum id.",
                               f"… و{more} أخرى: أجب بمعرّف المنهج.")
        return stop(state, "clarify", intro + question + lines, status="success", memory={"pending": {
            "doc_type": doc_type, "language": language, "extra": extra, "options": [c["curriculum_id"] for c in offered]}})

    # -------------------------------------------------- fetch_curriculum

    def fetch_curriculum(state: DocumentState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        found = call("get_approved_curriculum", curriculum_id=state["curriculum_id"])
        outcome = found.get("status")
        if outcome in ("not_approved", "not_found", "invalid_request"):
            return stop(state, outcome, found.get("message") or _t(ui, "The curriculum cannot be used.", "لا يمكن استخدام المنهج."))
        if outcome != "ok":
            return stop(state, "error", _t(ui, "The curriculum could not be retrieved: ", "تعذر استرجاع المنهج: ")
                        + str(found.get("message") or outcome))
        record = found["curriculum"]
        content = record["content"]
        language = state.get("language") or content.get("language")
        if language != content.get("language"):
            return stop(state, "invalid", _t(
                ui, f"The document is written in the curriculum's own language ({content.get('language')}); "
                    "revise the curriculum to change its language.",
                f"تُكتب الوثيقة بلغة المنهج نفسها ({content.get('language')})؛ عدّل المنهج لتغيير لغته."))
        template = TEMPLATES[state["doc_type"]]
        gaps = missing_curriculum_fields(content, template.required)
        if gaps:
            return stop(state, "missing_information", _t(
                ui, "The approved curriculum lacks information this document needs: " + ", ".join(gaps)
                    + ". Nothing was invented; revise and re-approve the curriculum.",
                "ينقص المنهج المعتمد معلومات تحتاجها هذه الوثيقة: " + "، ".join(gaps)
                + ". لم يُختلق شيء؛ عدّل المنهج وأعد اعتماده."), missing=gaps)
        return {"record": record, "language": language, "newer_version": bool(found.get("newer_unapproved_version"))}

    # ------------------------------------------------------------ write

    def write(state: DocumentState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        template = TEMPLATES[state["doc_type"]]
        if template.parts_schema is None:
            return {"parts": None}
        try:
            parts, problems = generate_with_repair(llm, template, state["record"]["content"], state["language"],
                                                   progress=state.get("progress"))
        except CurriculumModelError as exc:
            return stop(state, "generation_failed", _t(
                ui, f"The language model failed ({exc}). Nothing was saved.",
                f"فشل نموذج اللغة ({exc}). لم يُحفظ شيء."))
        if problems:
            two = template.targeted_repair          # (the Question Bank gets a second, targeted repair)
            return stop(state, "validation_failed", _t(
                ui, f"The generated text did not pass the checks, even after {'two repair attempts' if two else 'one repair attempt'}. "
                    "Nothing was saved: " + _problem_text(problems),
                f"لم يجتز النص المُنشأ الفحوص حتى بعد {'محاولتي إصلاح' if two else 'محاولة إصلاح واحدة'}. لم يُحفظ شيء: "
                + _problem_text(problems)))
        return {"parts": parts}

    # ------------------------------------------------------------- save

    def save(state: DocumentState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        content = compose(TEMPLATES[state["doc_type"]], state["record"], state["language"], state.get("parts"),
                          state.get("extra"))
        summary = "Regenerated" if state.get("document_id") else None
        saved = call("save_document_draft", content=content, document_id=state.get("document_id"),
                     change_summary=summary)
        if saved.get("status") != "ok":
            return stop(state, saved.get("status", "error"), _t(ui, "The document could not be saved: ", "تعذر حفظ الوثيقة: ")
                        + str(saved.get("message") or saved.get("status")))
        return {"saved": saved}

    # ----------------------------------------------------------- render

    def render(state: DocumentState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        saved = state["saved"]
        rendered = call("render_document_pdf", document_id=saved["document_id"], version=saved["version"])
        if rendered.get("status") != "ok":
            return stop(state, "render_failed", _t(
                ui, f"The document {saved['document_id']} (version {saved['version']}) was saved, but its PDF could "
                    f"not be created: {rendered.get('message') or rendered.get('status')}",
                f"حُفظت الوثيقة {saved['document_id']} (النسخة {saved['version']}) لكن تعذر إنشاء ملف PDF: "
                f"{rendered.get('message') or rendered.get('status')}"))
        return {"document": rendered["document"]}

    # ------------------------------------------------------------ reply

    def reply(state: DocumentState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        saved, document, record = state["saved"], state["document"], state["record"]
        validation = saved.get("validation") or {}
        content = document["content"]
        label = type_label(state["doc_type"], ui)
        lines = [_t(ui, f"{label} generated for “{record['content']['title']}” "
                        f"({record['curriculum_id']}, approved version {record['version']}).",
                    f"تم إنشاء «{label}» للمنهج «{record['content']['title']}» "
                    f"({record['curriculum_id']}، النسخة المعتمدة {record['version']}).")]
        lines.append(_t(ui, f"Document {saved['document_id']}, version {saved['version']}: {saved['review_status']}. "
                            "Review and edit it in the Documents tab, then approve it to publish.",
                        f"الوثيقة {saved['document_id']}، النسخة {saved['version']}: {saved['review_status']}. "
                        "راجعها وعدّلها في تبويب الوثائق ثم اعتمدها."))
        if content.get("missing"):
            fields = ("، " if ui == "ar" else ", ").join(sorted({field_label(m["field"], ui) for m in content["missing"]}))
            lines.append(_t(ui, f"Not specified in the curriculum (marked in the document): {fields}.",
                            f"غير محدد في المنهج (موضّح في الوثيقة): {fields}."))
        if validation.get("errors"):
            lines.append(_t(ui, "Validation problems remain, so it is a draft: ", "توجد مشكلات تحقق لذا هي مسودة: ")
                         + " ".join(e["message"] for e in validation["errors"]))
        if state.get("newer_version"):
            lines.append(_t(ui, "A newer version of this curriculum is waiting for approval; this document uses the "
                                "approved one.", "هناك نسخة أحدث من المنهج بانتظار الاعتماد؛ تستخدم الوثيقة النسخة المعتمدة."))
        response = {"kind": "document", "message": "\n".join(lines), "document_id": saved["document_id"],
                    "version": saved["version"], "review_status": saved["review_status"],
                    "document_type": state["doc_type"], "curriculum_id": record["curriculum_id"],
                    "curriculum_version": record["version"], "missing": content.get("missing") or [],
                    "validation": {"errors": validation.get("errors") or [], "warnings": validation.get("warnings") or []},
                    "pdf": {"filename": document.get("filename"), "pages": document.get("pages")}}
        return {"status": "success", "response": response,
                "memory_out": {"last_document_id": saved["document_id"], "pending": None}}

    def next_unless_done(target: str) -> Callable[[DocumentState], str]:
        return lambda state: "end" if state.get("response") else target

    graph = StateGraph(DocumentState)
    for name, node in (("understand", understand), ("fetch_curriculum", fetch_curriculum), ("write", write),
                       ("save", save), ("render", render), ("reply", reply)):
        graph.add_node(name, node)
    graph.set_entry_point("understand")
    for source, target in (("understand", "fetch_curriculum"), ("fetch_curriculum", "write"), ("write", "save"),
                           ("save", "render"), ("render", "reply")):
        graph.add_conditional_edges(source, next_unless_done(target), {target: target, "end": END})
    graph.add_edge("reply", END)
    return graph.compile()
