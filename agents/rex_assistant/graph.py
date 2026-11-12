"""
LangGraph workflow for Agent 1 (REX Assistant).

    detect_language -> handle_consent                       (consent reply)
                    -> understand -> ask_question            (discovery)
                                  -> general                 (greeting/help)
                                  -> clarify_course          ("which course?")
                                  -> retrieve -> generate    (both answer modes)
                                              -> clarify_course

Two answer modes share one retrieval path and one answer writer:
- Personalized recommendation: the six discovery fields (official brief,
  section 10) are collected one question at a time with fixed bilingual
  templates (discovery.py); then courses are retrieved for the profile.
- Direct course question: answered straight away from retrieved evidence,
  without the questionnaire. A question asked mid-discovery is answered
  and the questionnaire resumes from the same pending field afterwards.

Knowledge is reached only through the MCP tool search_knowledge_base
(agents/rag_mcp_server.py), which returns evidence, never an answer.
REX writes the final answer itself in `generate` (generate_answer()
below): it checks the evidence, builds its prompt from the retrieved
chunks (prompts.py), calls its own LangChain chat model, and for
recommendations applies its course checks (recommendation.py).

Which course a direct question is about is decided without the LLM
(course_context.py), against the courses in the indexed metadata (MCP
tool list_courses): one course named -> its evidence only; several
matching equally -> "Which course do you mean?" (clarify_course); "How
long is it?" -> the course under discussion (state course_focus).
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents.config import REX_DEFAULT_MODEL
from agents.course_intent import (
    COURSE_QUESTION,
    GENERAL,
    RECOMMENDATION,
    classify_course_intent,
    is_catalogue_listing,
    is_greeting,
    is_small_talk,
    looks_like_question,
    names_a_course,
)
from agents.rex_assistant import course_context, discovery, grounded_recommendation
from agents.rex_assistant.course_context import AMBIGUOUS, NAMED, NONE
from agents.rex_assistant.lead_store import LeadStore
from agents.rex_assistant.mcp_client import RagMcpClient
from agents.rex_assistant.prompts import (
    FOREIGN_SCRIPT,
    LANGUAGE_REMINDER,
    MODE_CATALOGUE,
    MODE_QUESTION,
    MODE_RECOMMENDATION,
    build_messages,
    cited_sources,
    course_question_task,
    evidence_context,
    is_absence_answer,
    language_rule,
    strip_citations,
    fallback_key,
    message,
)
from agents.rex_assistant.state import RexAssistantState
from agents.text_utils import normalize_for_matching
from knowledge_base.language import (
    contains_arabic,
    detect_conversation_language,
    detect_explicit_language_request,
    looks_like_arabizi,
    resolve_conversation_language,
    transliterate_arabizi,
)

logger = logging.getLogger(__name__)

DISCOVERY = "discovery"

# Consent words are matched as whole words of the normalized reply
# (agents.text_utils.normalize_for_matching), never as substrings: "ok"
# inside "look" or "no" inside "know" must not decide consent.
_CONSENT_REFUSAL_PHRASES = (
    "do not agree", "don't agree", "dont agree", "not agree",
    "لا اوافق", "مش موافق", "مش موافقه", "غير موافق", "مش عايز", "لا اريد",
)
# Agreement idioms that contain a negative word ("I don't mind").
_CONSENT_AGREE_PHRASES = (
    "no problem", "لا مانع", "مفيش مانع", "ما عندي مانع", "معنديش مانع",
)
_CONSENT_YES_WORDS = frozenset({
    "yes", "yeah", "yep", "sure", "ok", "okay", "agree", "agreed",
    "نعم", "موافق", "موافقه", "اوافق", "تمام", "حسنا", "ايوه", "ايوا", "اكيد",
})
_CONSENT_NO_WORDS = frozenset({
    "no", "nope", "nah", "not", "don", "dont", "never", "disagree",
    "decline", "refuse",
    "لا", "ارفض", "مش",
})
# Deferring or uncertain replies are never treated as consent.
_CONSENT_HEDGE_WORDS = frozenset({
    "maybe", "perhaps", "later", "wait", "unsure",
    "ربما", "يمكن", "بعدين", "لاحقا",
})
_WORD_RE = re.compile(r"[^\W_]+")

_TEXT = {
    "greeting": {
        "en": (
            "Hello! I'm REX, the academy's course advisor. I can recommend "
            "courses that fit your goals, or answer questions about our "
            "courses (content, duration, prerequisites, format...). How "
            "can I help?"
        ),
        "ar": (
            "أهلاً! أنا ريكس، مستشار الدورات في الأكاديمية. يمكنني ترشيح "
            "دورات تناسب أهدافك، أو الإجابة عن أسئلتك حول الدورات (المحتوى، "
            "المدة، المتطلبات، طريقة التدريب...). كيف أساعدك؟"
        ),
    },
    "resume": {
        "en": "Whenever you're ready, we can continue: ",
        "ar": "عندما تكون مستعدًا نكمل: ",
    },
    "incomplete": {
        "en": "Note: these suggestions use only what you've shared so far ({fields} not provided).",
        "ar": "ملاحظة: هذه الترشيحات مبنية فقط على ما شاركته حتى الآن (لم تُذكر: {fields}).",
    },
    "consent": {
        "en": (
            "Would you like us to save your details (background, goal and "
            "preferences) so the academy can follow up with you? Nothing is "
            "saved unless you say yes. (yes / no)"
        ),
        "ar": (
            "هل توافق على أن نحفظ بياناتك (الخلفية والهدف والتفضيلات) لتتابع "
            "معك الأكاديمية لاحقًا؟ لن يتم حفظ أي شيء إلا إذا وافقت. (نعم / لا)"
        ),
    },
    "saved": {
        "en": "Thank you! Your details have been saved for follow-up.",
        "ar": "شكرًا لك! تم حفظ بياناتك للمتابعة.",
    },
    "not_saved": {
        "en": "No problem, nothing has been saved.",
        "ar": "لا مشكلة، لم يتم حفظ أي معلومات.",
    },
    "save_failed": {
        "en": "Sorry, I couldn't save your details right now. Nothing was stored.",
        "ar": "عذرًا، تعذّر حفظ بياناتك الآن. لم يتم تخزين أي شيء.",
    },
    "confirm": {
        "en": "Sorry, could you confirm with yes or no?",
        "ar": "عذرًا، هل توافق على حفظ بياناتك للمتابعة؟ (نعم / لا)",
    },
}


def _t(key: str, language: Optional[str]) -> str:
    return _TEXT[key]["ar" if language == "ar" else "en"]


# Tokens the model may read per call. A recommendation prompt is about
# 1,700 tokens; history and longer evidence can roughly double that.
# Set explicitly so a smaller server default never silently cuts the
# prompt.
LLM_CONTEXT_TOKENS = 8192


def _supports_thinking(model: str) -> bool:
    """
    Whether the Ollama model has a "thinking" mode (e.g. Qwen3). Asked from
    the local Ollama server; False when it cannot be determined.
    """
    try:
        request = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return "thinking" in (json.load(response).get("capabilities") or [])
    except Exception:
        return False


@lru_cache(maxsize=4)
def _get_llm(model: str) -> ChatOllama:
    """
    Reuse one ChatOllama client per model name across graph calls.
    Thinking models answer directly (no hidden reasoning pass first), so
    replies stay fast.
    """
    options: dict[str, Any] = {"model": model, "temperature": 0, "num_ctx": LLM_CONTEXT_TOKENS}
    if _supports_thinking(model):
        options["reasoning"] = False
    return ChatOllama(**options)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse an LLM's JSON reply, tolerating markdown code fences."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _detect_consent(text: str) -> Optional[bool]:
    """
    True only for a clear yes, False for a clear no, otherwise None
    (the caller re-asks). Mixed replies such as "yes, but not now" are
    None: a lead is never saved on an unclear answer.
    """
    normalized = normalize_for_matching(text).replace("’", "'")
    words = set(_WORD_RE.findall(normalized))
    has_hedge = bool(words & _CONSENT_HEDGE_WORDS)

    def has_phrase(phrases: tuple[str, ...]) -> bool:
        return any(
            re.search(rf"(?<![^\W_]){re.escape(phrase)}(?![^\W_])", normalized)
            for phrase in phrases
        )

    if has_phrase(_CONSENT_REFUSAL_PHRASES):
        return False
    if has_phrase(_CONSENT_AGREE_PHRASES):
        return None if has_hedge else True

    has_yes = bool(words & _CONSENT_YES_WORDS)
    has_no = bool(words & _CONSENT_NO_WORDS)

    if has_no and not has_yes:
        return False
    if has_yes and not has_no and not has_hedge:
        return True

    return None


# ---------------------------------------------------------------------
# Which course is a direct question about?
# ---------------------------------------------------------------------

COURSE_LIST_TTL_SECONDS = 600
COURSE_LIST_RETRY_SECONDS = 60


class CourseCatalogue:
    """
    The courses named in the indexed metadata (MCP tool list_courses),
    cached for a few minutes. Shared by all conversations: it holds no
    learner data. Empty when the tool is unavailable -- REX then uses
    the courses named in the retrieved evidence instead.
    """

    def __init__(self, mcp_client: Any, clock=time.monotonic) -> None:
        self._client = mcp_client
        self._clock = clock
        self._courses: list[dict[str, Any]] = []
        self._next_refresh = 0.0
        self._records: list[dict[str, Any]] = []
        self._records_status = "unavailable"
        self._next_records_refresh = 0.0

    def records(self) -> tuple[list[dict[str, Any]], str]:
        """
        (verified course records, status) from the get_course_records MCP
        tool: the only courses REX may recommend. Status is "ok",
        "empty_index" or "unavailable" (tool missing or failed).
        """
        now = self._clock()
        if now < self._next_records_refresh:
            return self._records, self._records_status
        getter = getattr(self._client, "get_course_records", None)
        records, status = [], "unavailable"
        if callable(getter):
            try:
                result = getter()
                if isinstance(result, dict) and result.get("status") in ("ok", "empty_index"):
                    records = grounded_recommendation.verified_records(result.get("records"))
                    status = result["status"]
            except Exception:
                logger.warning("The course records are unavailable", exc_info=True)
        if status != "unavailable":
            self._records, self._records_status = records, status
            self._next_records_refresh = now + COURSE_LIST_TTL_SECONDS
        else:
            self._next_records_refresh = now + COURSE_LIST_RETRY_SECONDS
            if not self._records:
                self._records_status = status
        return self._records, self._records_status

    def courses(self) -> list[dict[str, Any]]:
        now = self._clock()
        if now < self._next_refresh:
            return self._courses
        fetched: list[dict[str, Any]] = []
        lister = getattr(self._client, "list_courses", None)
        if callable(lister):
            try:
                result = lister()
                if isinstance(result, dict) and result.get("status") == "ok":
                    fetched = course_context.clean_courses(result.get("courses"))
            except Exception:
                logger.warning("The course list is unavailable", exc_info=True)
        if fetched:
            self._courses, self._next_refresh = fetched, now + COURSE_LIST_TTL_SECONDS
        else:
            # Keep the last good list; ask again a little later.
            self._next_refresh = now + COURSE_LIST_RETRY_SECONDS
        return self._courses


def _saved_choice(choice: Any) -> Optional[dict[str, Any]]:
    """The pending "Which course do you mean?" from state, or None if unusable."""
    if not isinstance(choice, dict):
        return None
    question = choice.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    return {"question": question, "candidates": course_context.clean_courses(choice.get("candidates"))}


def _clarify(question: str, options: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask which course is meant; the question is answered once one is chosen."""
    return {
        "next_step": "clarify_course",
        "answer_mode": None,
        "course_options": options,
        "course_choice": {"question": question, "candidates": options},
    }


def _course_updates(text: str, state: RexAssistantState, catalogue: CourseCatalogue) -> dict[str, Any]:
    """
    For a direct question: the course(s) it is about (course_scope), a
    clarification, or "decide after retrieval" (course_post_match).
    """
    if is_catalogue_listing(text):
        # The whole catalogue is the topic now; no single course is.
        return {"course_focus": []}

    courses = catalogue.courses()
    reference = course_context.resolve_reference(text, courses)
    if reference.kind == NAMED:
        return {"course_scope": reference.courses, "course_focus": reference.courses, "course_question": text}
    if reference.kind == AMBIGUOUS:
        return _clarify(text, reference.courses)

    focus = course_context.known_courses(course_context.clean_courses(state.get("course_focus")), courses)
    position = course_context.ordinal_reference(text, len(focus))
    if position is not None:
        # "Tell me more about the first course": the first one just listed or recommended.
        chosen = [focus[position]]
        return {"course_scope": chosen, "course_focus": chosen, "course_question": text}
    if focus and course_context.refers_back(text):
        if len(focus) == 1:
            return {"course_scope": focus, "course_focus": focus, "course_question": text}
        return _clarify(text, focus)   # e.g. after a comparison or a recommendation
    if not focus and not state.get("active_topic") and course_context.needs_a_course(text):
        return _clarify(text, [])      # "How much does it cost?" -- which course?

    # Not decided by the words alone: an Arabic description ("دورة تحليل
    # البيانات") is matched on the English search query after retrieval,
    # and without a course list the courses in the evidence are used.
    return {"course_focus": focus, "course_post_match": True}


def _choice_updates(text: str, state: RexAssistantState, catalogue: CourseCatalogue) -> Optional[dict[str, Any]]:
    """The reply to "Which course do you mean?", or None when it is not one."""
    choice = _saved_choice(state.get("course_choice"))
    if choice is None or classify_course_intent(text) == RECOMMENDATION:
        return None

    courses = catalogue.courses()
    candidates = course_context.known_courses(choice["candidates"], courses)
    picked = course_context.parse_choice(text, candidates, courses)
    updates: dict[str, Any] = {"intent": COURSE_QUESTION, "clarify": False, "consent_pending": False}

    if picked.kind == NAMED:
        # "2" or "Python for Data Analysis" answers the question asked
        # before; "the second one, how much is it?" asks its own.
        own_question = classify_course_intent(text, course_context=True) == COURSE_QUESTION
        return {
            **updates,
            "next_step": "retrieve",
            "answer_mode": MODE_QUESTION,
            "course_scope": picked.courses,
            "course_focus": picked.courses,
            "course_question": text if own_question else choice["question"],
            "course_choice": None,
        }
    if picked.kind == AMBIGUOUS:
        return {**updates, **_clarify(choice["question"], picked.courses)}
    return None


MAX_BARE_COURSE_WORDS = 6


def _bare_course_reply(text: str, state: RexAssistantState, catalogue: CourseCatalogue) -> bool:
    """
    A short message that is only a course name ("python foundations",
    "Python for Data Analysis"): answered as a course question when it
    names a known course, or when a course conversation is already open.
    Greetings, thanks and acknowledgements ("ok") are not.
    """
    words = text.split()
    if not words or len(words) > MAX_BARE_COURSE_WORDS:
        return False
    if is_greeting(text) or is_small_talk(text) or discovery.is_acknowledgement(text):
        return False
    if not course_context.content_words(text):
        return False
    if state.get("active_topic") or state.get("course_focus"):
        return True
    return course_context.resolve_reference(text, catalogue.courses()).kind != NONE


def _search(mcp_client: RagMcpClient, **arguments: Any) -> dict[str, Any]:
    """The search_knowledge_base MCP tool; a failure becomes an error result."""
    try:
        # The only route to the knowledge base: the MCP tool. It returns
        # evidence and source metadata, never an answer.
        evidence = mcp_client.search_knowledge_base(**arguments)
    except Exception as exc:
        # Details (which may hold paths) go to the log; the state the
        # caller keeps gets the error type only.
        logger.warning("Knowledge-base tool call failed", exc_info=True)
        return {"status": "error", "chunks": [], "courses": [],
                "error": f"The knowledge-base search is unavailable ({type(exc).__name__})."}

    if not isinstance(evidence, dict) or "status" not in evidence:
        return {"status": "error", "chunks": [], "courses": [], "error": "Unexpected tool response."}
    return evidence


def _evidence_for_courses(
    mcp_client: RagMcpClient,
    question: str,
    courses: list[dict[str, Any]],
    catalogue: list[dict[str, Any]],
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Evidence about the selected course(s) only, plus general academy
    passages. A question that does not name the course itself ("How long
    is it?", or the original "the Python course" question after a choice)
    is searched together with the course's name. The course's own passages
    are then searched directly too: the RAG tool is given the course ids
    and limits the search to that course's passages (REX builds no database
    filter itself); general passages (no course) of the first search are
    kept after them.
    """
    named = course_context.resolve_reference(question, catalogue or courses)
    names_them = named.kind == NAMED and course_context.same_courses(named.courses, courses)
    labels = "; ".join(course_context.course_label(c) for c in courses)
    retrieval_query = None if names_them else f"{question} (course: {labels})"

    if evidence is None:
        evidence = _search(mcp_client, question=question, retrieval_query=retrieval_query)
    scoped, _ = course_context.scope_evidence(evidence, courses)

    course_ids = [c["course_id"] for c in courses if c.get("course_id")]
    if not course_ids or evidence.get("status") not in ("ok", "no_results"):
        return scoped
    # Also always the course's own passages: in a catalogue cut into small
    # passages, the course's heading can rank while the passage that answers
    # (its entry requirements, say) does not (live-index trace).

    filtered = _search(mcp_client, question=question, retrieval_query=retrieval_query,
                       course_ids=course_ids)
    own, found_own = course_context.scope_evidence(filtered, courses)
    return course_context.merge_evidence(own, scoped) if found_own else scoped


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def make_detect_language_node():
    def detect_language_node(state: RexAssistantState) -> dict[str, Any]:
        """
        Reply language = an explicit request, else the language this
        message is confidently written in, else the conversation's
        previous language (so "ok" or "5000" never flips it).
        """
        text = state["user_input"]
        preferred = (
            detect_explicit_language_request(text)
            or state.get("preferred_language")
        )
        detected = detect_conversation_language(text)

        return {
            "detected_language": detected,
            "conversation_language": detected or state.get("conversation_language"),
            "preferred_language": preferred,
            "response_language": resolve_conversation_language(
                text, state.get("conversation_language"), preferred
            ),
        }

    return detect_language_node


def route_after_language(state: RexAssistantState) -> str:
    """
    A reply to a pending consent question goes straight to
    handle_consent, so "yes"/"no" is never read as a profile fact.

    A message that is clearly something else -- a course question, a
    new request, or more than a short reply -- is handled normally
    instead of being answered with "please confirm yes or no". Nothing
    is saved: consent stays unanswered.
    """
    if not state.get("consent_pending"):
        return "understand"
    text = state.get("user_input", "")
    if _detect_consent(text) is None and (
        classify_course_intent(text) != GENERAL or len(text.split()) > 3
    ):
        return "understand"
    return "handle_consent"


# Fields an extracted value is accepted for only when the message itself
# answers them (normalize_answer on the message text).
_EVIDENCE_FIELDS = ("skill_level", "training_preference", "budget", "available_time")
MAX_SHORT_REPLY_WORDS = 4


def _only_a_level(value: str) -> bool:
    """'beginner', 'مبتدئ', 'an intermediate': a skill level and nothing else."""
    return bool(discovery.normalize_answer("skill_level", value)) and len(value.split()) <= 2


def _extract_profile_facts(llm, text: str) -> dict[str, str]:
    """LLM extraction of several profile facts from one free-form message."""
    instruction = (
        "Extract any of these fields the user just stated: background, "
        "career_goal, skill_level, training_preference (online/offline), "
        "budget, available_time, interests. Respond with ONLY a JSON "
        "object using these exact keys. Omit a key entirely if it was not "
        "stated -- never write placeholder text such as 'unknown'. Do not "
        "guess. Keep values in the user's own language.\n\n"
        f"User message: {text}"
    )

    if looks_like_arabizi(text):
        instruction += (
            "\n(This message is Egyptian Arabic written in Latin letters. "
            f"Partial Arabic-script reading: {transliterate_arabizi(text)})"
        )

    try:
        reply = llm.invoke([
            SystemMessage(content=(
                "You extract structured facts from a single user message. "
                "Reply with raw JSON only, no commentary."
            )),
            HumanMessage(content=instruction),
        ])
        return _parse_json_object(reply.content)
    except Exception:
        # Extraction failure must not block the conversation.
        return {}


def make_understand_node(model: str, catalogue: Optional[CourseCatalogue] = None):
    catalogue = catalogue or CourseCatalogue(None)

    def understand_node(state: RexAssistantState) -> dict[str, Any]:
        text = state["user_input"]

        # 0. The reply to "Which course do you mean?" (asked last turn)
        #    comes first: "2" is a choice here, not a discovery answer.
        chosen = _choice_updates(text, state, catalogue)
        if chosen is not None:
            return chosen

        profile = dict(state.get("profile") or {})
        declined = list(state.get("declined_fields") or [])
        attempts = dict(state.get("field_attempts") or {})
        pending = state.get("pending_field")
        in_discovery = state.get("mode") == DISCOVERY
        active_topic = state.get("active_topic")

        # explicit_question: the message itself asks about a course.
        # intent: the same, but a follow-up ("is it online?") also counts
        # when a course is already being discussed.
        explicit_question = classify_course_intent(text) == COURSE_QUESTION
        intent = classify_course_intent(
            text, course_context=bool(active_topic or state.get("course_focus"))
        )
        # "Python for Data Analysis" typed in the middle of the
        # questionnaire is a course question, not the pending answer.
        if in_discovery and pending and intent == GENERAL and course_context.clearly_names(
            text, catalogue.courses()
        ):
            explicit_question, intent = True, COURSE_QUESTION
        recommend_now = discovery.wants_recommendation_now(text)
        answered = False
        unclear = False

        # 1. A reply to the pending discovery question ("online", "5000",
        #    "skip") is tried first, unless it explicitly asks about a course.
        if in_discovery and pending and not recommend_now and not explicit_question:
            if discovery.wants_to_skip(text):
                declined.append(pending)
                pending, answered = None, True
            else:
                value = discovery.normalize_answer(pending, text)
                if value:
                    profile[pending] = value
                    declined = [f for f in declined if f != pending]
                    pending, answered = None, True

        question_turn = not answered and intent == COURSE_QUESTION
        # A short reply that answered the question asked ("Computer Engineer",
        # "مهندس كمبيوتر") answers only that question: extracting more fields
        # from it let the model copy the same words into another field (the
        # background became the career goal, and the goal was never asked).
        short_reply = answered and len(text.split()) <= MAX_SHORT_REPLY_WORDS

        # 2. Free-form messages can state several facts at once ("I'm an
        #    accountant, beginner, I want to become a data analyst").
        if not short_reply and (intent == RECOMMENDATION or (in_discovery and not question_turn)):
            for field, value in _extract_profile_facts(_get_llm(model), text).items():
                if field not in discovery.ALL_FIELDS or not isinstance(value, str):
                    continue
                if field in _EVIDENCE_FIELDS:
                    # Small models fill these in unasked ("online",
                    # "beginner"): keep them only when the learner's own
                    # words support them, and use the learner's wording.
                    normalized = discovery.normalize_answer(field, text) and (
                        discovery.normalize_answer(field, value)
                        if field in ("budget", "available_time")
                        else discovery.normalize_answer(field, text)
                    )
                elif _only_a_level(value):
                    # "أنا مبتدئ" ("I am a beginner") is a skill level, never a
                    # background, goal or interest (live trace: background = "مبتدئ").
                    normalized = None
                else:
                    normalized = discovery.normalize_answer(field, value)
                if normalized:
                    profile[field] = normalized
                    declined = [f for f in declined if f != field]
                    if field == pending:
                        pending, answered = None, True

        # 3. Still no usable answer: re-ask, at most MAX_ATTEMPTS_PER_FIELD
        #    times, then skip the field instead of looping.
        if in_discovery and pending and not answered and not question_turn and not recommend_now:
            attempts[pending] = attempts.get(pending, 0) + 1
            if attempts[pending] >= discovery.MAX_ATTEMPTS_PER_FIELD:
                declined.append(pending)
                pending = None
            else:
                unclear = True

        if question_turn:
            if names_a_course(text):
                active_topic = text  # later follow-ups are searched in its context
            next_step, answer_mode = "retrieve", MODE_QUESTION
        elif (
            not in_discovery
            and not active_topic
            and classify_course_intent(text, course_context=True) == COURSE_QUESTION
        ):
            # "How much does it cost?" with no course discussed yet, or a
            # question naming a course without the word "course" ("Who
            # teaches Building LLM Applications?"). The evidence decides:
            # one matching course is answered; several are listed and the
            # learner is asked which one (grounding rule 6).
            next_step, answer_mode = "retrieve", MODE_QUESTION
        elif intent == RECOMMENDATION or in_discovery or recommend_now:
            in_discovery = True
            missing = discovery.missing_fields(profile, declined)
            if missing and not recommend_now:
                pending = missing[0]
                next_step, answer_mode = "ask_question", None
            else:
                next_step, answer_mode = "retrieve", MODE_RECOMMENDATION
        elif looks_like_question(text) and not is_small_talk(text):
            # Any other question ("How experienced is Karim Nabil?") is
            # checked against the knowledge base rather than guessed from
            # keywords: relevant evidence -> an answer; none -> the
            # honest "couldn't find" message.
            next_step, answer_mode = "retrieve", MODE_QUESTION
        elif _bare_course_reply(text, state, catalogue):
            # "python foundations" on its own -- a course name, or the
            # reply to "which one do you mean?" in a course conversation.
            next_step, answer_mode = "retrieve", MODE_QUESTION
        else:
            next_step, answer_mode = "general", None

        updates = {
            "profile": profile,
            "declined_fields": declined,
            "field_attempts": attempts,
            "pending_field": pending,
            "mode": DISCOVERY if in_discovery else state.get("mode"),
            "active_topic": active_topic,
            "intent": intent,
            "next_step": next_step,
            "answer_mode": answer_mode,
            "clarify": unclear,
            # Reached with a consent question open only when the learner
            # moved on (route_after_language): leave consent unanswered.
            "consent_pending": False,
            # A "Which course?" question is answered on the very next turn
            # or not at all: an unrelated reply drops it.
            "course_choice": None,
        }

        if next_step == "retrieve" and answer_mode == MODE_QUESTION and is_catalogue_listing(text):
            # "What courses do you offer?" / "ما الدورات المتاحة في تحليل البيانات؟"
            updates.update({"answer_mode": MODE_CATALOGUE, "course_focus": []})
        elif next_step == "retrieve" and answer_mode == MODE_QUESTION:
            updates.update(_course_updates(text, state, catalogue))

        return updates

    return understand_node


def route_after_understand(state: RexAssistantState) -> str:
    return {
        "ask_question": "ask_question",
        "retrieve": "retrieve",
        "general": "general",
        "clarify_course": "clarify_course",
    }[state["next_step"]]


def route_after_retrieve(state: RexAssistantState) -> str:
    return "clarify_course" if state.get("next_step") == "clarify_course" else "generate"


def make_clarify_course_node():
    def clarify_course_node(state: RexAssistantState) -> dict[str, Any]:
        """A fixed bilingual "Which course do you mean?"; the LLM is not called."""
        language = state.get("response_language") or "en"
        options = course_context.clean_courses(state.get("course_options"))
        return {"response": course_context.clarification_message(options, language), "rag_result": None}

    return clarify_course_node


def make_ask_question_node():
    def ask_question_node(state: RexAssistantState) -> dict[str, Any]:
        language = state.get("response_language") or "en"
        field = state["pending_field"]
        parts = []

        if not state.get("discovery_intro_shown"):
            parts.append(discovery.DISCOVERY_INTRO["ar" if language == "ar" else "en"])
        if state.get("clarify"):
            parts.append(discovery.CLARIFY_PREFIX["ar" if language == "ar" else "en"]
                         + discovery.question_for(field, language))
        else:
            parts.append(discovery.question_for(field, language))

        return {"response": "\n\n".join(parts), "discovery_intro_shown": True}

    return ask_question_node


def make_general_node():
    def general_node(state: RexAssistantState) -> dict[str, Any]:
        language = state.get("response_language") or "en"
        return {"response": _t("greeting", language), "rag_result": None}

    return general_node


_RECORD_FACTS = (("department", "Department"), ("level", "Level"), ("total_hours", "Duration"),
                 ("delivery_mode", "Delivery mode"), ("fee_text", "Fee"), ("entry_requirements", "Entry requirements"),
                 ("prerequisites_text", "Prerequisite courses"), ("target_audience", "Intended audience"),
                 ("summary", "Summary"))


def _record_passage(record: dict[str, Any]) -> dict[str, Any]:
    """A verified course record (built by the RAG service from the indexed documents) as one evidence passage."""
    lines = [f"Catalogue record of {record['course_id']} {record['course_name']} (from the indexed catalogue)"]
    for key, label in _RECORD_FACTS:
        value = record.get(key)
        if key == "total_hours" and value:
            value = f"{value} hours"
        if key == "fee_text" and value is None and record.get("fee") is not None:
            value = f"{record['fee']:g} {record.get('fee_currency') or ''}".strip()
        lines.append(f"{label}: {value if value not in (None, '', []) else 'not stated in the catalogue'}")
    return {"text": "\n".join(lines), "course_id": record["course_id"], "course_name": record["course_name"],
            "source": (record.get("source_files") or [None])[0], "page": (record.get("pages") or [None])[0],
            "section": "Catalogue record", "section_path": None, "record_id": None, "chunk_id": None,
            "relevance_score": None, "distance": None}


def _with_course_records(evidence: dict[str, Any], courses: list[dict[str, Any]],
                         catalogue: "CourseCatalogue") -> dict[str, Any]:
    """
    The selected course's verified catalogue record, first among its
    passages: in a catalogue cut into small passages the model could miss
    a fact the retrieved passage does state ("45 guided learning hours" as
    a heading; live-index trace). The record comes from the RAG tool
    get_course_records and states only what the documents state.
    """
    if evidence.get("status") != "ok":
        return evidence
    wanted = {course_context.course_key(c) for c in courses}
    records = [r for r in catalogue.records()[0] if course_context.course_key(r) in wanted]
    if not records:
        return evidence
    chunks = [_record_passage(r) for r in records] + list(evidence.get("chunks") or [])
    return {**evidence, "chunks": [{**chunk, "reference": n} for n, chunk in enumerate(chunks, start=1)]}


# Passages retrieved for a recommendation (a question gets the default 5).
RECOMMENDATION_TOP_K = 12


def _catalogue_listing(mcp_client: RagMcpClient, text: str) -> dict[str, Any]:
    """The get_course_records MCP tool for a listing request; a failure becomes an error result."""
    getter = getattr(mcp_client, "get_course_records", None)
    try:
        result = getter(query=text) if callable(getter) else None
    except Exception as exc:
        logger.warning("Catalogue listing tool call failed", exc_info=True)
        return {"status": "error", "records": [], "error": f"The catalogue listing is unavailable ({type(exc).__name__})."}
    if not isinstance(result, dict) or "status" not in result:
        return {"status": "error", "records": [], "error": "Unexpected tool response."}
    return result


_PROFILE_QUERY = {
    "en": ("Courses suitable for a learner with ", "; ",
           {"career_goal": "career goal", "interests": "interests", "background": "background",
            "skill_level": "skill level", "training_preference": "training preference"}),
    "ar": ("دورات مناسبة لمتعلم ", "، ",
           {"career_goal": "هدفه المهني", "interests": "اهتماماته", "background": "خلفيته",
            "skill_level": "مستواه", "training_preference": "يفضل التدريب"}),
}
# Stored codes as words of the query's language.
_PROFILE_WORDS = {"ar": {"beginner": "مبتدئ", "intermediate": "متوسط", "advanced": "متقدم",
                         "online": "أونلاين", "offline": "حضوري", "hybrid": "مدمج"}}


def _recommendation_query(profile: dict[str, Any]) -> Optional[str]:
    """
    The search text for a recommendation, in ONE language: Arabic when the
    learner's answers are Arabic, so the RAG service translates a whole
    sentence (English labels around Arabic values came out garbled:
    ":: Data analysis;: data analysis;: beginner;..." in a live trace).
    "No preference" is no search term.
    """
    language = "ar" if any(contains_arabic(str(v)) for v in profile.values() if v) else "en"
    opening, separator, labels = _PROFILE_QUERY[language]
    parts = []
    for field in ("career_goal", "interests", "background", "skill_level", "training_preference"):
        value = profile.get(field)
        if not value or value == "no_preference":
            continue
        parts.append(f"{labels[field]}: {_PROFILE_WORDS.get(language, {}).get(value, value)}")
    return (opening + separator.join(parts)) if parts else None


def make_retrieve_node(mcp_client: RagMcpClient, catalogue: Optional[CourseCatalogue] = None):
    catalogue = catalogue or CourseCatalogue(None)

    def retrieve_node(state: RexAssistantState) -> dict[str, Any]:
        text = state["user_input"]

        if state["answer_mode"] == MODE_RECOMMENDATION:
            retrieval_query = _recommendation_query(state.get("profile") or {})
            # More passages than for a question: candidates come only from
            # the courses whose passages this similarity search retrieves.
            evidence = _search(mcp_client, question=text, retrieval_query=retrieval_query,
                               top_k=RECOMMENDATION_TOP_K)
            # Only these verified catalogue records may be recommended.
            records, records_status = catalogue.records()
            return {"evidence": {**evidence, "course_records": records, "course_records_status": records_status}}

        if state["answer_mode"] == MODE_CATALOGUE:
            # "What courses do you offer?" asks for the whole catalogue, which a
            # top-k passage search cannot give: the RAG tool lists every
            # verified course, narrowed by the request's topic or level.
            return {"evidence": {"status": "ok", "question": text, "chunks": [],
                                 "catalogue": _catalogue_listing(mcp_client, text)}}

        courses = catalogue.courses()

        # The course is already known: its evidence only.
        scope = course_context.clean_courses(state.get("course_scope"))
        if scope:
            question = state.get("course_question") or text
            evidence = _evidence_for_courses(mcp_client, question, scope, courses)
            return {"evidence": _with_course_records(evidence, scope, catalogue)}

        if (
            not courses
            and state.get("active_topic")
            and state["active_topic"] != text
            and not names_a_course(text)
        ):
            # No course list to identify courses with: search a follow-up
            # without a course name in the context of the earlier question.
            retrieval_query = f"{text} (about: {state['active_topic']})"
        else:
            retrieval_query = None

        evidence = _search(mcp_client, question=text, retrieval_query=retrieval_query)
        if not state.get("course_post_match"):
            return {"evidence": evidence}

        # Decide the course now: from the course list or, without one, the
        # courses in the evidence; an Arabic question by its English query.
        pool = courses or course_context.clean_courses(evidence.get("courses"))
        reference = course_context.resolve_reference(text, pool)
        translated = evidence.get("retrieval_query")
        if (
            reference.kind == NONE
            and retrieval_query is None
            and contains_arabic(text)
            and isinstance(translated, str)
            and translated != text
        ):
            reference = course_context.resolve_reference(translated, pool)

        if reference.kind == AMBIGUOUS:
            return {"evidence": evidence, **_clarify(text, reference.courses)}
        if reference.kind == NAMED:
            scoped = _evidence_for_courses(mcp_client, text, reference.courses, courses,
                                           evidence=evidence if retrieval_query is None else None)
            return {
                "evidence": _with_course_records(scoped, reference.courses, catalogue),
                "course_scope": reference.courses,
                "course_focus": reference.courses,
            }

        updates: dict[str, Any] = {"evidence": evidence}
        if names_a_course(text) and course_context.content_words(text):
            # A course REX does not know ("the Kubernetes course"): no
            # earlier course applies to the next follow-up either.
            updates["course_focus"] = []
        return updates

    return retrieve_node


def _invoke_text(llm, messages: list) -> str:
    """The model's reply text, or "" when the call fails."""
    try:
        reply = llm.invoke(messages)
        return str(getattr(reply, "content", "") or "").strip()
    except Exception:
        logger.exception("Answer generation failed")
        return ""


def generate_answer(
    llm,
    evidence: dict[str, Any],
    language: str,
    mode: str = MODE_QUESTION,
    profile: Optional[dict[str, Any]] = None,
    declined: Optional[list[str]] = None,
    history: Optional[list[dict[str, str]]] = None,
    selected_courses: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    REX writes its answer from the retrieved evidence. selected_courses:
    the course(s) a direct question is about, when REX identified them
    (the evidence is then already limited to them).

    A recommendation is built from the verified catalogue records only
    (grounded_recommendation.py, via _recommend_from_records): the model
    picks record ids, and every course name and fact comes from a record.

    A direct question:
    1. No usable evidence (empty index, no results, retrieval error):
       a fixed bilingual message, and the LLM is not called.
    2. Number the retrieved chunks so the answer can cite [1], [2]...
    3. Build the prompt: grounding rules, reply language, recent turns,
       the course-question task, the evidence and the original question.
    4. Call the chat model.

    Returns {"answer", "sources", "status"}; status is "answered" or the
    reason no answer was generated ("no_results", "empty_index",
    "unavailable", "no_usable_evidence", "generation_failed",
    "no_verified_courses", "no_suitable_course").
    """
    if mode == MODE_CATALOGUE:
        return _catalogue_answer(evidence.get("catalogue") or {}, language)
    if mode == MODE_RECOMMENDATION:
        # Only verified catalogue records may be recommended (retrieve_node
        # attaches them); without them nothing is recommended.
        return _recommend_from_records(llm, evidence, language, profile, declined)

    fallback = fallback_key(evidence)
    if fallback:
        return {"answer": message(fallback, language), "sources": [], "status": fallback}

    context, sources = evidence_context(evidence.get("chunks") or [])
    if not context:
        return {"answer": message("no_usable_evidence", language), "sources": [],
                "status": "no_usable_evidence"}

    messages = build_messages(
        evidence.get("question", ""), context, language, history,
        task=course_question_task(selected_courses),
        courses=evidence.get("courses") or [],
    )

    answer = _invoke_text(llm, messages)

    # The local model sometimes drifts into Chinese mid-answer: ask once
    # more with an explicit reminder, then remove any that remains.
    if FOREIGN_SCRIPT.search(answer):
        retry = _invoke_text(llm, [*messages, HumanMessage(content=LANGUAGE_REMINDER[language])])
        answer = retry if retry and not FOREIGN_SCRIPT.search(retry) else answer
        answer = re.sub(r"[ \t]{2,}", " ", FOREIGN_SCRIPT.sub(" ", answer)).strip()

    if not answer:
        return {"answer": message("generation_failed", language), "sources": [],
                "status": "generation_failed"}

    if is_absence_answer(answer):
        # "The context does not contain X. [1], [2]": nothing supports an
        # absence, so the labels go and no sources are listed.
        return {"answer": strip_citations(answer), "sources": [], "status": "answered"}

    # Only the sources the answer relies on (cited, or naming their course).
    return {"answer": answer, "sources": cited_sources(answer, sources), "status": "answered"}


def _catalogue_answer(listing: dict[str, Any], language: str) -> dict[str, Any]:
    """
    The catalogue written from the records the RAG tool returned (no model
    call: every course and fact is copied from its record). Its sources
    are the catalogue files the records come from.
    """
    status = listing.get("status")
    if status in ("empty_index", "translation_failed"):
        return {"answer": message(status, language), "sources": [], "status": status, "courses": []}
    if status != "ok":
        return {"answer": message("unavailable", language), "sources": [], "status": "unavailable", "courses": []}
    answer, listed = grounded_recommendation.render_catalogue(listing, language)
    files: list[str] = []
    for record in listing.get("records") or []:
        files.extend(f for f in record.get("source_files") or [] if f not in files)
    sources = [{"reference": None, "course_id": None, "course_name": None, "source": f, "page": None,
                "section": None, "section_path": None, "record_id": None, "chunk_id": None} for f in files]
    return {"answer": answer, "sources": sources, "status": "answered" if listed else "no_results", "courses": listed}


_RECOMMENDER_SYSTEM = (
    "You are REX Assistant, the course advisor of the academy. You recommend only courses from the "
    "catalogue list you are given, by their ids, and never invent, rename or combine courses."
)


def _recommend_from_records(llm, evidence: dict[str, Any], language: str,
                            profile: Optional[dict[str, Any]], declined: Optional[list[str]]) -> dict[str, Any]:
    """
    A recommendation built from the verified catalogue records only (see
    grounded_recommendation.py). With no records because the knowledge
    base is empty or unreachable, the usual fixed message is given.
    """
    records = evidence.get("course_records") or []
    fallback = fallback_key(evidence)
    if not records and evidence.get("course_records_status", "unavailable") == "unavailable" and fallback != "empty_index":
        fallback = "unavailable"
    if fallback in ("empty_index", "unavailable"):
        # The knowledge base is empty or unreachable: nothing can be verified.
        return {"answer": message(fallback, language), "sources": [], "status": fallback}
    _, sources = evidence_context(evidence.get("chunks") or [])
    system = f"{_RECOMMENDER_SYSTEM}\n\n{language_rule(language)}"
    written = grounded_recommendation.recommend(llm, _invoke_text, records, evidence, sources, language,
                                                profile, declined, system)
    if FOREIGN_SCRIPT.search(written["answer"]):
        written["answer"] = re.sub(r"[ \t]{2,}", " ", FOREIGN_SCRIPT.sub(" ", written["answer"])).strip()
    return written


def make_generate_node(model: str):
    def generate_node(state: RexAssistantState) -> dict[str, Any]:
        language = state.get("response_language") or "en"
        evidence = state["evidence"]
        mode = state["answer_mode"]
        profile = state.get("profile") or {}
        declined = state.get("declined_fields") or []

        written = generate_answer(
            _get_llm(model), evidence, language,
            mode=mode, profile=profile, declined=declined,
            history=state.get("history"),
            selected_courses=(course_context.clean_courses(state.get("course_scope"))
                              if mode == MODE_QUESTION else None),
        )

        response = written["answer"]
        updates: dict[str, Any] = {
            "rag_result": {
                "answer": written["answer"],
                "sources": written["sources"],
                "status": written["status"],
                "error": evidence.get("error"),
                "translation": evidence.get("translation"),
            },
        }
        if mode == MODE_CATALOGUE and written.get("courses"):
            # "Tell me more about the first one" refers to the listed courses.
            updates["course_focus"] = course_context.clean_courses(written["courses"])

        if mode == MODE_RECOMMENDATION and written["status"] == "answered":
            missing = discovery.missing_fields(profile, [])
            if missing:
                labels = ", ".join(discovery.field_label(f, language) for f in missing)
                response += "\n\n" + _t("incomplete", language).format(fields=labels)

            updates["mode"] = None
            updates["pending_field"] = None
            # A follow-up ("How long is it?") is about the recommended
            # course; with several recommended, REX asks which one.
            updates["course_focus"] = course_context.courses_of_sources(written["sources"])

            if (
                not state.get("lead_captured")
                and state.get("consent") is None
                and any(profile.get(f) for f in discovery.REQUIRED_FIELDS)
            ):
                response += "\n\n" + _t("consent", language)
                updates["consent_pending"] = True

        elif (
            state.get("mode") == DISCOVERY
            and state.get("pending_field")
            and written["status"] == "answered"
        ):
            # A direct question in the middle of discovery: answer it,
            # then show where the questionnaire will resume.
            response += "\n\n" + _t("resume", language) + discovery.question_for(
                state["pending_field"], language
            )

        updates["response"] = response
        return updates

    return generate_node


def make_handle_consent_node(lead_store: LeadStore):
    def handle_consent_node(state: RexAssistantState) -> dict[str, Any]:
        language = state.get("response_language") or "en"
        decision = _detect_consent(state["user_input"])

        if decision is True:
            # The only place learner data is persisted, and only the
            # profile fields (no conversation text).
            profile = {
                field: value
                for field, value in (state.get("profile") or {}).items()
                if field in discovery.ALL_FIELDS and value
            }
            try:
                lead_store.save_lead({"profile": profile})
            except Exception:
                return {
                    "consent": None,
                    "consent_pending": False,
                    "lead_captured": False,
                    "response": _t("save_failed", language),
                }
            return {
                "consent": True,
                "consent_pending": False,
                "lead_captured": True,
                "response": _t("saved", language),
            }

        if decision is False:
            return {
                "consent": False,
                "consent_pending": False,
                "lead_captured": False,
                "response": _t("not_saved", language),
            }

        # Ambiguous reply: ask again instead of guessing consent.
        return {"consent_pending": True, "response": _t("confirm", language)}

    return handle_consent_node


def build_graph(
    lead_store: LeadStore,
    model: str = REX_DEFAULT_MODEL,
    mcp_client: Optional[RagMcpClient] = None,
):
    """
    Build and compile Agent 1's LangGraph workflow.

    Returns (compiled_graph, mcp_client); call mcp_client.close() on
    shutdown. mcp_client defaults to a real client that starts
    agents/rag_mcp_server.py on first use.
    """
    mcp_client = mcp_client or RagMcpClient()
    catalogue = CourseCatalogue(mcp_client)

    graph = StateGraph(RexAssistantState)

    graph.add_node("detect_language", make_detect_language_node())
    graph.add_node("understand", make_understand_node(model, catalogue))
    graph.add_node("ask_question", make_ask_question_node())
    graph.add_node("general", make_general_node())
    graph.add_node("clarify_course", make_clarify_course_node())
    graph.add_node("retrieve", make_retrieve_node(mcp_client, catalogue))
    graph.add_node("generate", make_generate_node(model))
    graph.add_node("handle_consent", make_handle_consent_node(lead_store))

    graph.set_entry_point("detect_language")
    graph.add_conditional_edges(
        "detect_language",
        route_after_language,
        {"understand": "understand", "handle_consent": "handle_consent"},
    )
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {"ask_question": "ask_question", "retrieve": "retrieve", "general": "general",
         "clarify_course": "clarify_course"},
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "clarify_course": "clarify_course"},
    )
    graph.add_edge("ask_question", END)
    graph.add_edge("general", END)
    graph.add_edge("clarify_course", END)
    graph.add_edge("generate", END)
    graph.add_edge("handle_consent", END)

    return graph.compile(), mcp_client
