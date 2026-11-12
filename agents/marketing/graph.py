"""
LangGraph workflow for Agent 5 (Marketing).

    understand -> fetch_course -> audience -> write -> save -> reply

Any node can end the run early by setting "response" (a clarifying question, or an error that says which step
failed). Nothing is reported as a success unless the campaign was written, passed every check and was saved as pending
review. A model failure or a failed check saves nothing. The agent never approves, publishes, e-mails or sends anything.

Tools (the marketing MCP server or the in-process toolbox) are the only way to read approved courses and to store
campaigns; the language model writes only the strategy and the channel texts (generator.py).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.curriculum_research.llm import CurriculumModelError
from agents.marketing import composer, schemas, understanding as rules
from agents.marketing.generator import generate_campaign
from agents.marketing.marketing_store import CAMPAIGN_ID
from agents.marketing.validation import MAX_AUDIENCE, problem_text, validate_content

AGENT_NAME = "marketing"
COURSE_PAGE = 200
MAX_COURSE_PAGES = 10
MAX_OFFERED = 20
MAX_NOTE = 300
REQUIRED = ("title", "level", "total_hours", "overview", "learning_outcomes", "modules")


class MarketingState(TypedDict, total=False):
    request_text: str
    request_id: str
    ui_language: str
    form: dict[str, Any]
    memory: dict[str, Any]
    previous_results: list[dict[str, Any]]
    options: dict[str, Any]       # tone, goal, notes, channels, variations, audience override, campaign_id
    curriculum_id: str
    record: dict[str, Any]
    newer_version: bool
    audience: str
    audience_source: str
    strategy: dict[str, Any]
    written: dict[str, Any]
    campaign: dict[str, Any]
    saved: dict[str, Any]
    response: dict[str, Any]
    status: str
    memory_out: Optional[dict[str, Any]]
    progress: Any   # optional callable(done, total, label) so the Marketing tab can show how far the writing has got


def _t(language: str, english: str, arabic: str) -> str:
    return arabic if language == "ar" else english


def _clean(value: Any, limit: int) -> Optional[str]:
    text = " ".join(str(value or "").split())
    return text[:limit] or None


def read_options(raw: dict[str, Any], ui: str) -> tuple[dict[str, Any], Optional[str]]:
    """The request's choices with defaults filled in; (options, None) or ({}, the message about what is wrong)."""
    tone = raw.get("tone") or schemas.DEFAULT_TONE
    if tone not in schemas.TONES:
        return {}, _t(ui, f"The tone must be one of: {', '.join(schemas.TONES)}.", "النبرة يجب أن تكون من: " + "، ".join(schemas.TONES))
    channels = raw.get("channels")
    channels = list(schemas.CHANNELS) if channels in (None, [], ()) else channels
    if not isinstance(channels, (list, tuple)) or not all(c in schemas.CHANNELS for c in channels):
        return {}, _t(ui, f"Channels must be from: {', '.join(schemas.CHANNELS)}.", "القنوات يجب أن تكون من: " + "، ".join(schemas.CHANNELS))
    count = raw.get("variations")
    count = schemas.DEFAULT_VARIATIONS if count is None else count
    if isinstance(count, bool) or not isinstance(count, int) or not schemas.MIN_VARIATIONS <= count <= schemas.MAX_VARIATIONS:
        return {}, _t(ui, f"The number of variations must be {schemas.MIN_VARIATIONS}, 2 or {schemas.MAX_VARIATIONS}.",
                      f"عدد الصيغ يجب أن يكون من {schemas.MIN_VARIATIONS} إلى {schemas.MAX_VARIATIONS}.")
    campaign_id = raw.get("campaign_id")
    if campaign_id is not None and not (isinstance(campaign_id, str) and CAMPAIGN_ID.match(campaign_id)):
        return {}, _t(ui, "That is not a valid campaign id.", "معرّف الحملة غير صالح.")
    audience = _clean(raw.get("audience"), MAX_AUDIENCE + 1)
    if audience and rules.unsafe_audience(audience):
        return {}, _t(ui, "The audience must be a short description of a group of people, without links, addresses or markup.",
                      "يجب أن يكون الجمهور وصفًا قصيرًا لفئة من الناس، دون روابط أو عناوين أو علامات.")
    if audience and len(audience) > MAX_AUDIENCE:
        return {}, _t(ui, f"The audience is longer than {MAX_AUDIENCE} characters.", f"وصف الجمهور أطول من {MAX_AUDIENCE} حرفًا.")
    return {"tone": tone, "channels": [c for c in schemas.CHANNELS if c in channels], "variations": count,
            "goal": _clean(raw.get("goal"), MAX_NOTE), "notes": _clean(raw.get("notes"), MAX_NOTE),
            "audience": audience, "campaign_id": campaign_id}, None


def _audience_again(kind: str, ui: str) -> str:
    """The question asked again when a reply cannot be used as the audience."""
    why = {"ambiguous": ("That is too general to write for.", "هذا عام جدًا للكتابة له."),
           "invalid": ("That does not look like a description of an audience (it is too long, or contains a link or an address).",
                       "هذا لا يبدو وصفًا للجمهور (طويل جدًا أو يحتوي رابطًا أو عنوانًا)."),
           }.get(kind, ("I could not use that as the audience.", "لم أستطع استخدام هذا كجمهور."))
    return _t(ui, why[0] + " Who is this campaign for? Describe a specific group in a few words (for example: new graduates in "
                           "accounting), or say cancel.",
              why[1] + " لمن هذه الحملة؟ صف فئة محددة في بضع كلمات (مثال: خريجون جدد في المحاسبة) أو قل إلغاء.")


def build_graph(tools: Any, llm: Any):
    """Compile the workflow. ``tools``: the MCP client (or toolbox); ``llm``: a LangChain chat model."""

    def call(name: str, **kwargs: Any) -> dict[str, Any]:
        result = getattr(tools, name)(**kwargs)
        return result if isinstance(result, dict) else {"status": "error", "message": f"{name} returned nothing."}

    def stop(kind: str, message: str, *, status: str = "error", memory: Optional[dict[str, Any]] = None, **extra: Any) -> dict[str, Any]:
        return {"status": status, "response": {"kind": kind, "message": message, **extra}, "memory_out": memory}

    # ------------------------------------------------------- understand

    def understand(state: MarketingState) -> dict[str, Any]:
        text, form = state["request_text"], state.get("form") or {}
        memory = dict(state.get("memory") or {})
        pending = memory.get("pending") or {}
        ui = state.get("ui_language", "en")

        if form:
            curriculum_id, raw = form.get("curriculum_id"), form
        else:
            awaiting = pending.get("awaiting")
            kind, cleaned = rules.classify_reply(text) if awaiting in ("audience", "course") else ("", text)
            if awaiting in ("audience", "course") and kind == "cancel":
                return stop("cancelled", _t(ui, "Okay, I cancelled that campaign request. Nothing was created.",
                                            "تم إلغاء طلب الحملة. لم يُنشأ شيء."), status="success", memory={"pending": None})
            if awaiting == "audience" and kind == "audience":
                raw = {**(pending.get("options") or {}), "audience": cleaned}      # a real answer to the open question
                curriculum_id = pending.get("curriculum_id")
            elif awaiting == "audience" and kind in ("ambiguous", "invalid", "empty", "small_talk"):
                return stop("need_audience", _audience_again(kind, ui), status="success", missing=["audience"],
                            memory={"pending": pending})                                   # asked again; nothing is generated
            else:
                # A new request (or a course answer): it is read on its own. The open audience question is dropped, and an
                # audience typed for the earlier request is never carried over.
                raw = dict(pending.get("options") or {}) if awaiting == "course" else {}
                curriculum_id = rules.find_curriculum_id(text) or rules.choose_from_options(text, pending.get("course_ids") or [])
        options, problem = read_options(raw, ui)
        if problem:
            return stop("invalid", problem)
        if curriculum_id:
            return {"options": options, "curriculum_id": curriculum_id}

        # No course named: offer the approved ones, or pick the one the request describes.
        approved: list[dict[str, Any]] = []
        for _ in range(MAX_COURSE_PAGES):
            listed = call("list_approved_courses", limit=COURSE_PAGE, offset=len(approved))
            if listed.get("status") != "ok":
                return stop("error", listed.get("message") or _t(ui, "The approved courses could not be listed.", "تعذر عرض المقررات المعتمدة."))
            approved += listed.get("courses") or []
            if not listed.get("has_more"):
                break
        if not approved:
            return stop("not_approved", _t(
                ui, "No course is approved yet. Approve a curriculum in the Curricula tab first: marketing content is only "
                    "created from approved courses.",
                "لا يوجد مقرر معتمد بعد. اعتمد منهجًا في تبويب المناهج أولًا: لا يُنشأ المحتوى التسويقي إلا من مقررات معتمدة."))
        drafted = rules.drafted_a_curriculum(state.get("previous_results"))
        matches = [] if drafted else rules.match_courses(text, approved)
        if len(matches) == 1:
            return {"options": options, "curriculum_id": matches[0]["curriculum_id"]}
        offered_all = matches or approved
        offered = offered_all[:MAX_OFFERED]
        lines = "\n".join(f"{i}. {c['title']} — {c['curriculum_id']} (v{c['approved_version']})" for i, c in enumerate(offered, start=1))
        intro = _t(ui, "The curriculum drafted just now is not approved, so no marketing content is created from it. Approve it in "
                       "the Curricula tab, then ask again naming it. ", "المنهج الذي صيغ للتو غير معتمد، لذا لا يُنشأ محتوى تسويقي منه. "
                       "اعتمده في تبويب المناهج ثم اطلب مجددًا مع ذكر معرّفه. ") if drafted else ""
        question = _t(ui, "Which approved course should the campaign promote? Reply with its number or id:\n",
                      "أي مقرر معتمد تريد الترويج له؟ أجب برقمه أو معرّفه:\n")
        if len(offered_all) > len(offered):
            lines += "\n" + _t(ui, f"… and {len(offered_all) - len(offered)} more: reply with a course id.",
                                f"… و{len(offered_all) - len(offered)} أخرى: أجب بمعرّف المقرر.")
        return stop("clarify", intro + question + lines, status="success", memory={"pending": {
            "awaiting": "course", "options": options, "course_ids": [c["curriculum_id"] for c in offered]}})

    # ------------------------------------------------------ fetch_course

    def fetch_course(state: MarketingState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        found = call("get_approved_course", curriculum_id=state["curriculum_id"])
        outcome = found.get("status")
        if outcome in ("not_approved", "not_found", "invalid_request"):
            # Kept so a chained request that paused here (admin_service.py's
            # pending_workflows.py) can resume with this exact marketing
            # request once the curriculum is approved, instead of
            # re-matching free text against every approved course.
            return stop(outcome, found.get("message") or _t(ui, "The course cannot be used.", "لا يمكن استخدام المقرر."),
                       curriculum_id=state["curriculum_id"], options=state.get("options"))
        if outcome != "ok":
            return stop("error", _t(ui, "The course could not be retrieved: ", "تعذر استرجاع المقرر: ") + str(found.get("message") or outcome))
        record = found["course"]
        content = record["content"]
        gaps = [f for f in REQUIRED if content.get(f) in (None, "", [], {})]
        if gaps:
            return stop("missing_information", _t(
                ui, "The approved course lacks information a campaign needs: " + ", ".join(gaps) + ". Nothing was invented; "
                    "revise and re-approve the curriculum.",
                "ينقص المقرر المعتمد معلومات تحتاجها الحملة: " + "، ".join(gaps) + ". لم يُختلق شيء؛ عدّل المنهج وأعد اعتماده."), missing=gaps)
        return {"record": record, "newer_version": bool(found.get("newer_unapproved_version"))}

    # ----------------------------------------------------------- audience

    def audience(state: MarketingState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        options, content = state["options"], state["record"]["content"]
        from_course = (content.get("target_audience") or "").strip()
        supplied = options.get("audience")
        if supplied:
            return {"audience": supplied, "audience_source": "curriculum" if supplied == from_course else "admin"}
        if from_course:
            return {"audience": from_course, "audience_source": "curriculum"}
        # In the Marketing tab the form's own audience field is the answer, so nothing is remembered; in chat the question
        # is remembered with the request it belongs to (the agent stamps it with the time, so it expires).
        memory = None if state.get("form") else {"pending": {
            "awaiting": "audience", "curriculum_id": state["curriculum_id"], "options": options,
            "request_text": state["request_text"][:200]}}
        return stop("need_audience", _t(
            ui, "The approved course does not state its target audience, and I will not invent one. Who is this campaign for? "
                "Reply with a short description (for example: new graduates in accounting), or say cancel.",
            "لا يذكر المقرر المعتمد جمهوره المستهدف ولن أختلقه. لمن هذه الحملة؟ أجب بوصف قصير (مثال: خريجون جدد في المحاسبة) "
            "أو قل إلغاء."), status="success", missing=["audience"], memory=memory)

    # -------------------------------------------------------------- write

    def write(state: MarketingState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        options, content = state["options"], state["record"]["content"]
        labels = {c: schemas.CHANNEL_LABELS[c][ui] for c in schemas.CHANNELS}
        labels["strategy"] = _t(ui, "Campaign strategy", "استراتيجية الحملة")
        try:
            strategy, written, problems = generate_campaign(
                llm, content, audience=state["audience"], audience_source=state["audience_source"], tone=options["tone"],
                goal=options["goal"], notes=options["notes"], channels=options["channels"], count=options["variations"],
                labels=labels, progress=state.get("progress"))
        except CurriculumModelError as exc:
            return stop("generation_failed", _t(ui, f"The language model failed ({exc}). Nothing was saved.",
                                                f"فشل نموذج اللغة ({exc}). لم يُحفظ شيء."))
        if problems:
            failed = problems[0].get("where") or "strategy"
            name = labels.get(failed.split("[")[0], failed)
            return stop("validation_failed", _t(
                ui, f"{name} did not pass the checks, even after one repair attempt. Nothing was saved: " + problem_text(problems),
                f"لم يجتز «{name}» الفحوص حتى بعد محاولة إصلاح واحدة. لم يُحفظ شيء: " + problem_text(problems)),
                failed_content=failed.split("[")[0], problems=problems)
        return {"strategy": strategy, "written": written}

    # --------------------------------------------------------------- save

    def save(state: MarketingState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        options, record = state["options"], state["record"]
        campaign = composer.compose(record, state["strategy"], state["written"], audience=state["audience"],
                                    audience_source=state["audience_source"], tone=options["tone"], goal=options["goal"],
                                    notes=options["notes"], variations=options["variations"])
        checked = validate_content(campaign, record)
        if not checked["valid"]:   # never saved, not even as a draft
            return stop("validation_failed", _t(ui, "The campaign did not pass the final checks. Nothing was saved: ",
                                                "لم تجتز الحملة الفحوص النهائية. لم يُحفظ شيء: ") + problem_text(checked["errors"]),
                        problems=checked["errors"])
        saved = call("save_campaign_draft", content=campaign, campaign_id=options.get("campaign_id"),
                     change_summary="Regenerated" if options.get("campaign_id") else None)
        if saved.get("status") != "ok":
            return stop(saved.get("status", "error"), _t(ui, "The campaign could not be saved: ", "تعذر حفظ الحملة: ")
                        + str(saved.get("message") or saved.get("status")))
        return {"campaign": campaign, "saved": saved}

    # -------------------------------------------------------------- reply

    def reply(state: MarketingState) -> dict[str, Any]:
        ui = state.get("ui_language", "en")
        saved, record, campaign = state["saved"], state["record"], state["campaign"]
        lines = [_t(ui, f"Marketing campaign written for “{record['content']['title']}” ({record['curriculum_id']}, approved "
                        f"version {record['version']}), for: {campaign['audience']['text']}.",
                    f"تمت كتابة حملة تسويقية للمقرر «{record['content']['title']}» ({record['curriculum_id']}، النسخة المعتمدة "
                    f"{record['version']}) لجمهور: {campaign['audience']['text']}.")]
        lines.append(_t(ui, f"Campaign {saved['campaign_id']}, version {saved['version']}: {saved['review_status']}. Nothing has been "
                            "published or sent. Review and edit it in the Marketing tab, then approve it.",
                        f"الحملة {saved['campaign_id']}، النسخة {saved['version']}: {saved['review_status']}. لم يُنشر ولم يُرسل شيء. "
                        "راجعها وعدّلها في تبويب التسويق ثم اعتمدها."))
        if state.get("newer_version"):
            lines.append(_t(ui, "A newer version of this curriculum is waiting for approval; the campaign uses the approved one.",
                            "هناك نسخة أحدث من المنهج بانتظار الاعتماد؛ تستخدم الحملة النسخة المعتمدة."))
        response = {"kind": "campaign", "message": "\n".join(lines), "campaign_id": saved["campaign_id"], "version": saved["version"],
                    "review_status": saved["review_status"], "curriculum_id": record["curriculum_id"],
                    "curriculum_version": record["version"], "audience": campaign["audience"], "tone": campaign["tone"],
                    "variations": campaign["variations"], "channels": list(campaign["channels"]), "strategy": campaign["strategy"],
                    "validation": {"errors": (saved.get("validation") or {}).get("errors") or [], "warnings": []}}
        return {"status": "success", "response": response,
                "memory_out": {"last_campaign_id": saved["campaign_id"], "pending": None}}

    def next_unless_done(target: str) -> Callable[[MarketingState], str]:
        return lambda state: "end" if state.get("response") else target

    graph = StateGraph(MarketingState)
    for name, node in (("understand", understand), ("fetch_course", fetch_course), ("audience", audience), ("write", write),
                       ("save", save), ("reply", reply)):
        graph.add_node(name, node)
    graph.set_entry_point("understand")
    for source, target in (("understand", "fetch_course"), ("fetch_course", "audience"), ("audience", "write"),
                           ("write", "save"), ("save", "reply")):
        graph.add_conditional_edges(source, next_unless_done(target), {target: target, "end": END})
    graph.add_edge("reply", END)
    return graph.compile()
