
import re
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.multilingual import detect_conversation_language
from agents.multilingual import detect_language as _detect_language
from agents.course_intent import GENERAL, classify_course_intent
from agents.routing import deliverables, split_clauses
from agents.text_utils import normalize_for_matching


# Common response format from docs/agent_interfaces.md
AgentHandler = Callable[[dict[str, Any]], dict[str, Any]]

# Optional per-agent check: "does this message continue a conversation
# I already have open?" It receives {"user_input", "context"} and is only
# asked when no keyword rule matched.
ContinuationCheck = Callable[[dict[str, Any]], bool]

# "create/design/... a (new) <topic> course" -- course-creation requests
# that contain no "curriculum" keyword. Excluded below when the request
# is really about recommendations or documents.
_COURSE_CREATION_RE = re.compile(
    r"\b(?:create|generate|design|build|develop|make|prepare|draft|write)"
    r"\s+(?:me\s+|us\s+)?(?:a|an|the)\s+"
    r"(?:(?!(?:for|about|on|in|of|to|with|and)\b)[\w+#.-]+\s+){0,4}?course\b"
)
_COURSE_CREATION_AR = (
    "انشاء دوره", "انشئ دوره", "صمم دوره", "اعداد دوره", "اعمل دوره",
    "ابني دوره", "اكتب دوره",
)
_COURSE_CREATION_BLOCKERS = (
    "recommend", "suggest", "proposal", "guide", "pdf", "document",
    "question bank", "tender", "rfp", "rfq",
    "رشح", "اقترح", "مقترح", "مستند", "دليل", "بنك اسئله",
)


# Catalogue planning (Agent 3, Workflow B): "suggest new courses for our
# AI track", "identify missing learning paths in the catalogue". Needs
# "new/missing courses" PLUS an academy-planning cue, so a learner's
# "recommend new courses for me" still reaches REX.
_PLANNING_NEW = re.compile(
    r"\b(?:new|additional|more|future|missing)\s+(?:[\w-]+\s+){0,2}?"
    r"(?:courses?|curricula|programs?|programmes?|learning\s+paths?|tracks?|offerings?)\b"
)
_PLANNING_CUE = re.compile(
    r"\b(?:we|our|academy|catalogue|catalog|portfolio|department|track|offer|add|launch|introduce)\b"
)
_PLANNING_GAP = re.compile(
    r"\b(?:gaps?|missing)\b.{0,40}\b(?:catalogue|catalog|courses|learning\s+paths?|offerings?|portfolio)\b"
    r"|\b(?:catalogue|catalog)\b.{0,40}\b(?:gaps?|review|analysis|analy[sz]e|missing)\b"
    r"|\b(?:review|analy[sz]e|audit)\b.{0,20}\b(?:catalogue|catalog)\b"
)
# Picking one of Agent 3's catalogue suggestions ("generate suggestion 1").
# Without this, REX's course-intent rule reads it as a recommendation request.
_PLANNING_PICK = re.compile(
    r"\b(?:generate|create|build|draft|develop|design|use|choose|pick|select)\s+(?:the\s+)?"
    r"(?:suggestion|suggested\s+course|proposal|option)\s*(?:no\.?|number|#)?\s*\d\b"
    r"|\b(?:first|second|third|fourth|fifth)\s+suggestion\b"
    r"|(?:انشئ|اعد|صمم|طور|اختر|اختار|استخدم)\s+(?:ال)?(?:اقتراح|مقترح)\s*(?:رقم\s*)?(?:\d|الاول|الثاني|الثالث|الرابع|الخامس)"
)
_PLANNING_NEW_AR = re.compile(r"(?:دورات|مناهج|كورسات|مسارات|برامج)\s+(?:تدريبيه\s+)?(?:جديده|ناقصه|اضافيه)")
_PLANNING_CUE_AR = re.compile(r"الاكاديميه|للاكاديميه|الكتالوج|نضيف|نطلق|نقدم|للقسم|اضافتها")
_PLANNING_GAP_AR = re.compile(
    r"(?:فجوات|فجوه|نواقص).{0,40}(?:الكتالوج|الدورات|المسارات|البرامج|الاكاديميه)"
    r"|(?:الكتالوج|كتالوج).{0,30}(?:فجوات|مراجعه|تحليل)|(?:راجع|حلل)\s+(?:ال)?كتالوج|المسارات\s+الناقصه"
)


class OrchestratorState(TypedDict, total=False):
    request: dict[str, Any]
    plan: list[str]
    excluded: list[str]
    agent_inputs: dict[str, str]
    results: list[dict[str, Any]]
    current_index: int
    response: dict[str, Any]


_EXCLUDED_MESSAGE = {
    "rex_assistant": {
        "en": (
            "Course recommendations and course questions are handled by "
            "REX Assistant in the User interface, not in the Admin dashboard."
        ),
        "ar": (
            "ترشيح الدورات والأسئلة عنها يتولاها مساعد ريكس في واجهة "
            "المستخدم، وليس في لوحة الإدارة."
        ),
    },
}


_STEP_NAMES = {
    "rfp_research": ("Tender research", "البحث عن المناقصات"),
    "curriculum_research": ("Curriculum research", "إعداد المناهج"),
    "document_generation": ("Document generation", "إنشاء المستندات"),
    "marketing": ("Marketing", "التسويق"),
    "rex_assistant": ("REX Assistant", "مساعد ريكس"),
    "orchestrator": ("Orchestrator", "المنسّق"),
}


# Agents that build on the curriculum an earlier step of the same request produced.
_USES_PRIOR_CURRICULUM = ("document_generation", "marketing")
_STATUS_WORDS = {"pending_review": ("pending review", "بانتظار المراجعة"), "draft": ("draft", "مسودة"),
                 "approved": ("approved", "معتمد")}


def _artifacts(body: dict[str, Any], ar: bool) -> str:
    """What a step produced, from its own result: ids and review status (nothing invented)."""
    parts = []
    if body.get("curriculum_id"):
        status = _STATUS_WORDS.get(str(body.get("review_status") or ""), (body.get("review_status"),) * 2)
        parts.append(f"{body['curriculum_id']}" + (f", {status[1 if ar else 0]}" if status[0] else ""))
    if body.get("bilingual_companion_id"):
        parts.append(("النسخة الأخرى " if ar else "companion ") + str(body["bilingual_companion_id"]))
    for key in ("document_id", "campaign_id"):
        if body.get(key):
            parts.append(str(body[key]))
    return f" [{'; '.join(parts)}]" if parts else ""


def _warnings(body: dict[str, Any]) -> list[str]:
    warnings = body.get("warnings")
    if isinstance(warnings, list):
        return [str(w.get("message") if isinstance(w, dict) else w) for w in warnings if w][:3]
    return []


def synthesize_results(results: list[dict[str, Any]], language: str) -> str:
    """
    One coherent account of a request, in the order the agents ran: what
    each step did (its own first line) and produced (ids, review status),
    its warnings, which step is waiting for an approval, which failed and
    why, which steps built on an earlier step's curriculum (or that they
    ran independently), and a final tally. Built only from the agents' own
    results; nothing is invented.
    """
    ar = language == "ar"
    lines = []
    done = waiting = failed = 0
    curriculum_step = None
    dependent = []
    for number, result in enumerate(results, start=1):
        agent = result.get("agent_name")
        name = _STEP_NAMES.get(agent, (agent,) * 2)[1 if ar else 0]
        body = result.get("result") if isinstance(result.get("result"), dict) else {}
        first = str(body.get("message") or "").strip().splitlines()
        if result.get("status") == "success":
            done += 1
            outcome = (first[0] if first else ("تم." if ar else "Done.")) + _artifacts(body, ar)
        elif result.get("status") == "paused":
            waiting += 1
            outcome = ("بانتظار اعتماد المسؤول؛ تُستأنف الخطوات التالية بعده."
                       if ar else "Waiting for an administrator's approval; the next steps resume after it.")
        elif result.get("status") == "waiting_for_input":
            waiting += 1
            outcome = first[0] if first else ("بانتظار إجابتك." if ar else "Waiting for your answer.")
        else:
            failed += 1
            outcome = (("فشلت: " if ar else "Failed: ") + str(result.get("error_message") or ""))
        lines.append(f"{number}. {name}: {outcome}")
        lines.extend(("   ⚠ " + w) for w in _warnings(body))
        if agent == "curriculum_research" and curriculum_step is None and result.get("status") == "success":
            curriculum_step = number
        elif agent in _USES_PRIOR_CURRICULUM and curriculum_step is not None:
            dependent.append(number)
    if len(results) > 1:
        if dependent:
            steps = ", ".join(str(n) for n in dependent)
            lines.append(f"استخدمت الخطوة/الخطوات {steps} المنهج الناتج عن الخطوة {curriculum_step}."
                         if ar else f"Step{'s' if len(dependent) > 1 else ''} {steps} used the curriculum "
                                    f"from step {curriculum_step}.")
        else:
            lines.append("نُفذت هذه الخطوات مستقلةً عن بعضها." if ar else "These steps ran independently.")
        tally = [f"{'تم' if ar else 'Done'}: {done}/{len(results)}"]
        if waiting:
            tally.append(f"{'بانتظار الاعتماد' if ar else 'Waiting'}: {waiting}")
        if failed:
            tally.append(f"{'فشلت' if ar else 'Failed'}: {failed}")
        lines.append(" · ".join(tally))
    return "\n".join(lines)


_TENDER_TERMS = (
    "tender", "tenders", "rfp", "rfq", "request for proposal", "request for quotation", "مناقصة", "مناقصات",
    "مناقصه", "مناقصات تعليمية", "عطاء", "عطاءات", "طلب تقديم عروض", "طلب عروض أسعار", "طلب عروض اسعار",
)


def _mentions_tenders(folded_text: str) -> bool:
    return any(term in folded_text for term in _TENDER_TERMS)


def _agent_inputs(plan: list[str], text: str) -> dict[str, str]:
    """
    For a request that mixes independent jobs (a tender search and a new
    curriculum), each agent's own part: the tender clauses for RFP
    Research, the rest for Curriculum Research -- so the curriculum topic
    is not "tenders in Egypt". Chained requests (curriculum -> document ->
    marketing) keep the whole text: the later agents need it.
    """
    if plan != ["rfp_research", "curriculum_research"]:
        return {}
    clauses = split_clauses(text)
    tender = [c for c in clauses if _mentions_tenders(c.casefold())]
    other = [c for c in clauses if not _mentions_tenders(c.casefold())]
    if not tender or not other:
        return {}
    return {"rfp_research": ". ".join(tender), "curriculum_research": ". ".join(other)}


class Orchestrator:
    """
    The orchestrator: routes Arabic and English requests to the five specialist agents.

    Language detection and routing use deterministic Python rules.
    RAG is not implemented here; specialist agents can call the
    team's shared RAG component when it becomes available.
    """

    AGENT_NAMES = {
        "rex_assistant",
        "rfp_research",
        "curriculum_research",
        "document_generation",
        "marketing",
    }

    def __init__(self, excluded_agents: tuple[str, ...] | frozenset[str] = ()) -> None:
        """
        excluded_agents: agents this orchestrator must never run, even if
        a request is about them (the Admin orchestrator excludes
        "rex_assistant", which users reach directly). Such requests get
        an explanation instead of being routed elsewhere.
        """
        self.agents: dict[str, AgentHandler] = {}
        self._continuation_checks: dict[str, ContinuationCheck] = {}
        self.excluded_agents = frozenset(excluded_agents)
        unknown = self.excluded_agents - self.AGENT_NAMES
        if unknown:
            raise ValueError(f"Unknown specialist agent(s): {sorted(unknown)}")
        graph = StateGraph(OrchestratorState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._next_step,
            {"execute": "execute", "finish": END},
        )

        self.graph = graph.compile()
        # The last confidently detected language per conversation_id, so a
        # short reply ("ok", "5000", "yes") keeps the conversation's
        # language instead of falling back to English. In memory, bounded.
        self._conversation_languages: dict[str, str] = {}

    MAX_REMEMBERED_CONVERSATIONS = 1000

    def _resolve_language(self, user_input: str, context: dict[str, Any]) -> str:
        """
        An explicit UI choice first, then a confident detection of this
        message (knowledge_base/language.py, the same detector REX uses),
        then this conversation's previous language, then English.
        """
        if context.get("preferred_language") in ("ar", "en"):
            return context["preferred_language"]
        detected = detect_conversation_language(user_input)
        conversation_id = context.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            return detected or "en"
        if detected is None:
            return self._conversation_languages.get(conversation_id, "en")
        self._conversation_languages.pop(conversation_id, None)  # most recent last
        self._conversation_languages[conversation_id] = detected
        while len(self._conversation_languages) > self.MAX_REMEMBERED_CONVERSATIONS:
            self._conversation_languages.pop(next(iter(self._conversation_languages)))
        return detected

    @staticmethod
    def detect_language(text: str) -> str:
        """
        "ar" for Arabic script (MSA, Egyptian, mixed with English) or
        confident Arabizi, otherwise "en". See agents/multilingual.py.
        """
        return _detect_language(text)

    def register_agent(
        self,
        name: str,
        handler: AgentHandler,
        continuation_check: ContinuationCheck | None = None,
    ) -> None:
        """
        Register an implemented specialist agent.

        continuation_check (optional): lets the agent claim a follow-up
        message that matches no keyword rule, because it has an open
        conversation (for example a draft the user is refining). It is
        only consulted when every keyword rule found nothing, so
        explicit requests for other agents always take priority.
        """
        if name not in self.AGENT_NAMES:
            raise ValueError(f"Unknown specialist agent: {name}")

        if name in self.excluded_agents:
            raise ValueError(f"Agent '{name}' is excluded from this orchestrator")

        self.agents[name] = handler

        if continuation_check is None:
            self._continuation_checks.pop(name, None)
        else:
            self._continuation_checks[name] = continuation_check

    def handle_request(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Detect the language, route the request, and return a response."""
        request_id = request_id or str(uuid4())
        user_input = user_input.strip()

        if not user_input:
            return self._error_response(
                request_id, "Please provide a request."
            )

        request_context = dict(context or {})
        # An explicit UI language choice wins over per-message detection;
        # unclear short replies keep the conversation's language.
        language = self._resolve_language(user_input, request_context)
        request_context["language"] = language

        request = {
            "request_id": request_id,
            "agent_name": "orchestrator",
            "user_input": user_input,
            "context": request_context,
            "language": language,
        }

        try:
            state = self.graph.invoke(
                {
                    "request": request,
                    "plan": [],
                    "results": [],
                    "current_index": 0,
                }
            )
            return state["response"]
        except Exception as error:
            return self._error_response(
                request_id, f"Orchestrator error: {error}"
            )

    def _plan_node(self, state: OrchestratorState) -> dict[str, Any]:
        request = state["request"]
        # A document type and curriculum chosen in the admin form are an
        # explicit request: no keyword guessing.
        if isinstance((request.get("context") or {}).get("document_request"), dict):
            plan = ["document_generation"]
        # Likewise a course, audience and channels chosen in the Marketing tab.
        elif isinstance((request.get("context") or {}).get("marketing_request"), dict):
            plan = ["marketing"]
        else:
            plan = self._make_plan(request["user_input"])

        if not plan:
            plan = self._plan_from_open_conversation(request)

        # A resumed workflow (admin_service.py's pending_workflows.py):
        # the agent(s) already run before a pause are never run again.
        skip = (request.get("context") or {}).get("skip_agents") or []
        if skip:
            plan = [name for name in plan if name not in skip]

        # Excluded agents are never run; the request is answered with an
        # explanation instead of being sent to some other agent.
        excluded = [name for name in plan if name in self.excluded_agents]
        if excluded:
            plan = []

        return {
            "plan": plan,
            "excluded": excluded,
            "agent_inputs": _agent_inputs(plan, request["user_input"]),
            "results": [],
            "current_index": 0,
        }

    def _plan_from_open_conversation(
        self, request: dict[str, Any]
    ) -> list[str]:
        """
        No keyword matched: ask the agents that registered a
        continuation check whether this message continues their open
        conversation. A failing check is ignored, never fatal.
        """
        probe = {
            "user_input": request["user_input"],
            "context": dict(request.get("context") or {}),
        }

        for name, check in list(self._continuation_checks.items()):
            if name not in self.agents:
                continue

            try:
                if check(probe):
                    return [name]
            except Exception:
                continue

        return []

    @staticmethod
    def _is_course_creation(text: str) -> bool:
        """
        "Create/design/... a (new) course" requests. Not matched when the
        request is really about recommendations or documents, so REX and
        document generation keep those.
        """
        normalized = normalize_for_matching(text)

        if any(blocker in normalized for blocker in _COURSE_CREATION_BLOCKERS):
            return False

        return bool(
            _COURSE_CREATION_RE.search(normalized)
            or any(phrase in normalized for phrase in _COURSE_CREATION_AR)
        )

    @staticmethod
    def _is_catalogue_planning(text: str) -> bool:
        """Requests to plan new academy courses or review catalogue gaps (Agent 3)."""
        normalized = normalize_for_matching(text)
        return bool(
            (_PLANNING_NEW.search(normalized) and _PLANNING_CUE.search(normalized))
            or _PLANNING_GAP.search(normalized)
            or _PLANNING_PICK.search(normalized)
            or (_PLANNING_NEW_AR.search(normalized) and _PLANNING_CUE_AR.search(normalized))
            or _PLANNING_GAP_AR.search(normalized)
        )

    def _make_plan(self, text: str) -> list[str]:
        """
        Route Arabic or English requests using keywords.

        Explicit multi-agent workflows are checked before individual
        agent routes. This is rule-based routing, not semantic
        understanding by an LLM.
        """
        original_text = text
        text = text.casefold()

        # Multi-agent workflow: curriculum followed by marketing.
        curriculum_terms = (
            "curriculum",
            "curricula",
            "course outline",
            "syllabus",
            "منهج",
            "مناهج",
            "منهاج",
            "مخرجات التعلم",
            "خطة دراسية",
        )

        # "Create/design a course ..." has no curriculum keyword but is
        # a request the Curriculum Research Agent handles.
        course_creation = self._is_course_creation(original_text)

        # What the request asks FOR (agents/routing.py). A topic such as "a curriculum about digital marketing" is a
        # subject, not a marketing request; a sentence with two deliverables ("a question bank and a marketing campaign",
        # "a curriculum and a course proposal") runs every agent that was asked for, in the order curriculum, documents,
        # marketing. Nothing is routed to Marketing because of the bare word "marketing".
        asked = deliverables(original_text, self._is_course_creation)
        if len(asked) >= 2:
            return asked

        # Agent 2: RFP and tender research -- with Agent 3 too when another
        # part of the request asks for a curriculum ("Find training tenders
        # in Egypt and create a data analytics curriculum").
        if _mentions_tenders(text):
            other_parts = [c for c in split_clauses(original_text) if not _mentions_tenders(c.casefold())]
            if any("curriculum_research" in deliverables(c, self._is_course_creation) for c in other_parts):
                return ["rfp_research", "curriculum_research"]
            return ["rfp_research"]

        # One explicit deliverable: a supported document (Agent 4) or marketing content (Agent 5). Before the curriculum
        # keywords, because such a request often names a curriculum ("a course description for curriculum CUR-...").
        if asked in (["document_generation"], ["marketing"]):
            return asked

        # Agent 3: curriculum research and accreditation alignment, and
        # catalogue planning (new courses, catalogue gaps).
        if course_creation or self._is_catalogue_planning(original_text) or any(term in text for term in curriculum_terms + (
            "learning outcomes",
            "accreditation",
            "learning objectives",
            "training plan",
            "أهداف التعلم",
            "الأهداف التعليمية",
            "اعتماد",
            "الاعتماد الأكاديمي",
            "الاعتماد الاكاديمي",
            "تصميم منهج",
            "تصميم منهج دراسي",
            "منهج تدريبي",
        )):
            return ["curriculum_research"]

        # Agent 4: educational document generation.
        if any(term in text for term in (
            "generate a document",
            "create a document",
            "make a pdf",
            "participant guide",
            "instructor guide",
            "question bank",
            "proposal document",
            "create a proposal",
            "أنشئ مستند",
            "انشئ مستند",
            "إنشاء مستند",
            "إنشاء ملف pdf",
            "ملف pdf",
            "دليل المتدرب",
            "دليل المدرب",
            "بنك أسئلة",
            "بنك اسئلة",
            "إنشاء مقترح",
            "إنشاء عرض",
            "إعداد وثيقة",
        )):
            return ["document_generation"]

        # Agent 1: REX Assistant -- course recommendations and course
        # questions (agents/course_intent.py). Checked last, so every more
        # specific intent above keeps priority.
        if classify_course_intent(original_text) != GENERAL or any(term in text for term in (
            "recommend a course",
            "recommend courses",
            "which course",
            "course recommendation",
            "course recommendations",
            "what course should",
            "find me a course",
            "suggest a course",
            "training course",
            "courses for beginners",
            "رشح لي دورة",
            "رشح دورة",
            "رشح لي كورس",
            "اقترح دورة",
            "اقترح كورس",
            "أريد دورة",
            "اريد دورة",
            "أريد كورس",
            "اريد كورس",
            "الدورات المناسبة",
            "دورات للمبتدئين",
            "دورات تدريبية",
            "ما هي الدورات",
            "ماهي الدورات",
            "تعلم البرمجة",
            "تعلم تحليل البيانات",
        )):
            return ["rex_assistant"]

        # Unknown requests are not sent to an arbitrary agent.
        return []

    def _execute_node(
        self, state: OrchestratorState
    ) -> dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("current_index", 0)
        results = list(state.get("results", []))
        request = state["request"]

        if not plan:
            excluded = state.get("excluded") or []
            if excluded:
                language = "ar" if request.get("language") == "ar" else "en"
                error_message = _EXCLUDED_MESSAGE.get(
                    excluded[0], {}
                ).get(language, f"Agent '{excluded[0]}' is not available here.")
            else:
                error_message = (
                    "I couldn't identify the right agent. Please specify "
                    "course recommendations, tender research, curriculum "
                    "research, document generation, or marketing. "
                    "يمكنك توضيح ما تحتاج إليه: ترشيح الدورات، أو البحث "
                    "عن المناقصات، أو إعداد المناهج، أو إنشاء المستندات، "
                    "أو التسويق."
                )
            results = [{
                "request_id": request["request_id"],
                "agent_name": "orchestrator",
                "status": "error",
                "result": None,
                "error_message": error_message,
            }]

            return {
                "results": results,
                "current_index": index,
                "response": self._combine_results(request, results),
            }

        if index >= len(plan):
            return {
                "results": results,
                "current_index": index,
                "response": self._combine_results(request, results),
            }

        agent_name = plan[index]
        handler = self.agents.get(agent_name)

        if handler is None:
            agent_result = {
                "request_id": request["request_id"],
                "agent_name": agent_name,
                "status": "error",
                "result": None,
                "error_message": (
                    f"Agent '{agent_name}' is not implemented or "
                    "registered yet."
                ),
            }
        else:
            agent_request = {
                "request_id": request["request_id"],
                "agent_name": agent_name,
                # Its own part of a mixed request (see _agent_inputs); the
                # whole request otherwise. The full text stays in context.
                "user_input": (state.get("agent_inputs") or {}).get(agent_name, request["user_input"]),
                "context": {
                    **request.get("context", {}),
                    "language": request.get("language", "en"),
                    # A snapshot, not a live reference: results keeps
                    # growing after this call returns, and an agent
                    # must only see what ran before it.
                    "previous_results": list(results),
                },
            }

            try:
                agent_result = handler(agent_request)

                if not isinstance(agent_result, dict):
                    raise TypeError("Agent must return a dictionary.")

                agent_result = {
                    "request_id": request["request_id"],
                    "agent_name": agent_result.get(
                        "agent_name", agent_name
                    ),
                    "status": agent_result.get("status", "success"),
                    "result": agent_result.get("result"),
                    "error_message": agent_result.get("error_message"),
                }
            except Exception as error:
                agent_result = {
                    "request_id": request["request_id"],
                    "agent_name": agent_name,
                    "status": "error",
                    "result": None,
                    "error_message": str(error),
                }

        results.append(agent_result)
        next_index = index + 1

        updates: dict[str, Any] = {
            "results": results,
            "current_index": next_index,
        }

        if next_index >= len(plan):
            updates["response"] = self._combine_results(request, results)

        return updates

    def _combine_results(
        self,
        request: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures = [
            result for result in results
            if result["status"] != "success"
        ]

        return {
            "request_id": request["request_id"],
            "agent_name": "orchestrator",
            "status": "error" if failures else "success",
            "result": results,
            "summary": synthesize_results(results, request.get("language", "en")),
            "error_message": (
                "; ".join(
                    result.get("error_message") or "Agent failed"
                    for result in failures
                )
                if failures
                else None
            ),
        }

    def _next_step(self, state: OrchestratorState) -> str:
        if state.get("current_index", 0) < len(state.get("plan", [])):
            return "execute"

        return "finish"

    def _error_response(
        self, request_id: str, message: str
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "agent_name": "orchestrator",
            "status": "error",
            "result": None,
            "error_message": message,
        }


# Convenient default instance for importing elsewhere.
orchestrator = Orchestrator()
