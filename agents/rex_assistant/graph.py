"""
LangGraph workflow for Agent 1 (REX Assistant).

    detect_language -> handle_consent                       (consent reply)
                    -> understand -> ask_question            (discovery)
                                  -> general                 (greeting/help)
                                  -> retrieve -> generate    (both answer modes)

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
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents.config import REX_DEFAULT_MODEL
from agents.course_intent import (
    CATALOGUE_LISTING_HINT,
    COURSE_QUESTION,
    GENERAL,
    RECOMMENDATION,
    classify_course_intent,
    is_catalogue_listing,
    is_small_talk,
    looks_like_question,
    names_a_course,
)
from agents.rex_assistant import discovery
from agents.rex_assistant.lead_store import LeadStore
from agents.rex_assistant.mcp_client import RagMcpClient
from agents.rex_assistant.prompts import (
    FOREIGN_SCRIPT,
    LANGUAGE_REMINDER,
    MODE_QUESTION,
    MODE_RECOMMENDATION,
    build_messages,
    evidence_context,
    fallback_key,
    message,
)
from agents.rex_assistant.recommendation import check_recommendation, recommendation_task
from agents.rex_assistant.state import RexAssistantState
from agents.text_utils import normalize_for_matching
from knowledge_base.language import (
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


def make_understand_node(model: str):
    def understand_node(state: RexAssistantState) -> dict[str, Any]:
        text = state["user_input"]
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
        intent = classify_course_intent(text, course_context=bool(active_topic))
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

        # 2. Free-form messages can state several facts at once ("I'm an
        #    accountant, beginner, I want to become a data analyst").
        if intent == RECOMMENDATION or (in_discovery and not question_turn):
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
        else:
            next_step, answer_mode = "general", None

        return {
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
        }

    return understand_node


def route_after_understand(state: RexAssistantState) -> str:
    return {
        "ask_question": "ask_question",
        "retrieve": "retrieve",
        "general": "general",
    }[state["next_step"]]


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


def _recommendation_query(profile: dict[str, Any]) -> Optional[str]:
    parts = [
        f"{label}: {profile[field]}"
        for field, label in (
            ("career_goal", "career goal"),
            ("interests", "interests"),
            ("background", "background"),
            ("skill_level", "skill level"),
            ("training_preference", "training preference"),
        )
        if profile.get(field)
    ]
    return ("Courses suitable for a learner with " + "; ".join(parts)) if parts else None


def make_retrieve_node(mcp_client: RagMcpClient):
    def retrieve_node(state: RexAssistantState) -> dict[str, Any]:
        text = state["user_input"]

        if state["answer_mode"] == MODE_RECOMMENDATION:
            retrieval_query = _recommendation_query(state.get("profile") or {})
        elif (
            state.get("active_topic")
            and state["active_topic"] != text
            and not names_a_course(text)
        ):
            # Follow-up without a course name: search it in the context
            # of the course discussed before.
            retrieval_query = f"{text} (about: {state['active_topic']})"
        elif is_catalogue_listing(text):
            retrieval_query = f"{text}. {CATALOGUE_LISTING_HINT}"
        else:
            retrieval_query = None

        try:
            # The only route to the knowledge base: the MCP tool. It
            # returns evidence and source metadata, never an answer.
            evidence = mcp_client.search_knowledge_base(
                question=text, retrieval_query=retrieval_query
            )
        except Exception as exc:
            evidence = {"status": "error", "chunks": [], "error": f"{type(exc).__name__}: {exc}"}

        if not isinstance(evidence, dict) or "status" not in evidence:
            evidence = {"status": "error", "chunks": [], "error": "Unexpected tool response."}

        return {"evidence": evidence}

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
) -> dict[str, Any]:
    """
    REX writes its answer from the retrieved evidence.

    1. No usable evidence (empty index, no results, retrieval error):
       a fixed bilingual message, and the LLM is not called.
    2. Number the retrieved chunks so the answer can cite [1], [2]...
    3. Build the prompt: grounding rules, reply language, recent turns,
       the recommendation task (profile + course layout) when
       recommending, the evidence, and the learner's original question.
    4. Call the chat model.
    5. Recommendations only: apply REX's course checks to the answer.

    Returns {"answer", "sources", "status"}; status is "answered" or the
    reason no answer was generated ("no_results", "empty_index",
    "unavailable", "no_usable_evidence", "generation_failed").
    """
    fallback = fallback_key(evidence)
    if fallback:
        return {"answer": message(fallback, language), "sources": [], "status": fallback}

    context, sources = evidence_context(evidence.get("chunks") or [])
    if not context:
        return {"answer": message("no_usable_evidence", language), "sources": [],
                "status": "no_usable_evidence"}

    recommending = mode == MODE_RECOMMENDATION
    messages = build_messages(
        evidence.get("question", ""), context, language, history,
        task=recommendation_task(profile, declined) if recommending else None,
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

    if recommending:
        answer = check_recommendation(answer, context, profile, language)

    return {"answer": answer, "sources": sources, "status": "answered"}


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

        if mode == MODE_RECOMMENDATION and written["status"] == "answered":
            missing = discovery.missing_fields(profile, [])
            if missing:
                labels = ", ".join(discovery.field_label(f, language) for f in missing)
                response += "\n\n" + _t("incomplete", language).format(fields=labels)

            updates["mode"] = None
            updates["pending_field"] = None

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

    graph = StateGraph(RexAssistantState)

    graph.add_node("detect_language", make_detect_language_node())
    graph.add_node("understand", make_understand_node(model))
    graph.add_node("ask_question", make_ask_question_node())
    graph.add_node("general", make_general_node())
    graph.add_node("retrieve", make_retrieve_node(mcp_client))
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
        {"ask_question": "ask_question", "retrieve": "retrieve", "general": "general"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("ask_question", END)
    graph.add_edge("general", END)
    graph.add_edge("generate", END)
    graph.add_edge("handle_consent", END)

    return graph.compile(), mcp_client
