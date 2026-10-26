"""
LangGraph workflow for Agent 1 (REX Assistant).

Reaches the existing RAG knowledge base exclusively through the MCP
tool in agents/rag_mcp_server.py (via RagMcpClient), never by importing
knowledge_base.answer_generator or other RAG internals directly here.
The MCP server itself still wraps the validated RAGAnswerGenerator, so
no RAG logic is duplicated -- this module only adds what Agent 1's own
requirements need: profile-aware follow-up questions, course
recommendation/Q&A via the MCP tool, and consent-gated lead capture.

This agent does not route between the five REX AI agents and does not
decide which agent handles a request -- that is the central
orchestrator's responsibility, implemented separately by a teammate.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from knowledge_base.language import detect_language, resolve_response_language
from agents.rex_assistant.lead_store import LeadStore
from agents.rex_assistant.mcp_client import RagMcpClient
from agents.rex_assistant.state import (
    PROFILE_FIELDS_IN_PRIORITY_ORDER,
    RexAssistantState,
)


_KNOWLEDGE_QUESTION_KEYWORDS = (
    "course", "courses", "programme", "program", "fee", "fees", "price",
    "cost", "duration", "prerequisite", "instructor", "schedule",
    "format", "outcome", "content", "curriculum", "audience",
    "دورة", "دورات", "برنامج", "رسوم", "سعر", "مدة", "متطلبات",
    "مدرب", "جدول", "محتوى", "مناهج",
)

_CONSENT_YES = (
    "yes", "sure", "ok", "okay", "agree", "fine",
    "نعم", "موافق", "تمام", "حسنا", "اوافق", "أوافق",
)
_CONSENT_NO = (
    "no", "nope", "don't", "do not", "disagree",
    "لا", "ارفض", "أرفض", "لا أوافق",
)


@lru_cache(maxsize=4)
def _get_llm(model: str) -> ChatOllama:
    """Reuse one ChatOllama client per model name across graph calls."""
    return ChatOllama(model=model, temperature=0)


def _looks_like_knowledge_question(text: str) -> bool:
    normalized = text.casefold()
    return any(keyword in normalized for keyword in _KNOWLEDGE_QUESTION_KEYWORDS)


def _missing_profile_fields(profile: dict[str, Any]) -> list[str]:
    return [
        field
        for field in PROFILE_FIELDS_IN_PRIORITY_ORDER
        if not profile.get(field)
    ]


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
    normalized = text.casefold().strip()

    if any(word in normalized for word in _CONSENT_NO):
        return False
    if any(word in normalized for word in _CONSENT_YES):
        return True

    return None


def make_detect_language_node():
    def detect_language_node(state: RexAssistantState) -> dict[str, Any]:
        detected = detect_language(state["user_input"])
        response_language = resolve_response_language(
            detected,
            state.get("preferred_language"),
        )

        return {
            "detected_language": detected,
            "response_language": response_language,
        }

    return detect_language_node


def make_update_profile_node(model: str):
    def update_profile_node(state: RexAssistantState) -> dict[str, Any]:
        profile = dict(state.get("profile", {}))
        llm = _get_llm(model)

        instruction = (
            "Extract any of the following fields the user just mentioned: "
            "background, career_goal, skill_level, interests, budget, "
            "available_time. Respond with ONLY a JSON object using these "
            "exact keys. Omit a key entirely if it was not mentioned in "
            "this message. Do not guess values that were not stated.\n\n"
            f"User message: {state['user_input']}"
        )

        try:
            reply = llm.invoke([
                SystemMessage(content=(
                    "You extract structured facts from a single user "
                    "message. Reply with raw JSON only, no commentary."
                )),
                HumanMessage(content=instruction),
            ])
            updates = _parse_json_object(reply.content)
        except Exception:
            # A classification failure should not block the conversation;
            # simply proceed without updating the profile this turn.
            updates = {}

        for field in (
            "background", "career_goal", "skill_level",
            "interests", "budget", "available_time",
        ):
            value = updates.get(field)
            if isinstance(value, str) and value.strip():
                profile[field] = value.strip()

        return {"profile": profile}

    return update_profile_node


def route_after_language(state: RexAssistantState) -> str:
    """
    Send a reply to a pending consent question straight to handle_consent,
    without running profile extraction on it first -- otherwise a plain
    "yes"/"no" gets misread as a profile fact (e.g. an "interest").
    """
    if state.get("consent_pending"):
        return "handle_consent"

    return "update_profile"


def route_after_profile(state: RexAssistantState) -> str:
    profile = state.get("profile", {})
    is_knowledge_question = _looks_like_knowledge_question(state["user_input"])
    profile_sufficient = not _missing_profile_fields(profile)

    if is_knowledge_question or profile_sufficient:
        return "rag_answer"

    return "ask_followup"


def make_ask_followup_node(model: str):
    def ask_followup_node(state: RexAssistantState) -> dict[str, Any]:
        missing = _missing_profile_fields(state.get("profile", {}))
        response_language = state.get("response_language") or "en"
        llm = _get_llm(model)

        language_instruction = (
            "Write the question in Modern Standard Arabic."
            if response_language == "ar"
            else "Write the question in English."
        )

        prompt = (
            "You are REX Assistant, a friendly course-recommendation "
            "chatbot. Ask the user ONE short, natural follow-up question "
            f"to learn about: {missing[0]}. "
            "Do not ask about anything else, and do not repeat "
            "information already known. "
            f"{language_instruction}"
        )

        try:
            reply = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=state["user_input"]),
            ])
            question = reply.content.strip()
        except Exception as exc:
            question = (
                "عذرًا، حدثت مشكلة في الاتصال بالمساعد. حاول مرة أخرى."
                if response_language == "ar"
                else (
                    "Sorry, I couldn't reach the assistant model right "
                    "now. Please try again."
                )
            )

        return {"response": question}

    return ask_followup_node


def make_rag_answer_node(mcp_client: RagMcpClient):
    def rag_answer_node(state: RexAssistantState) -> dict[str, Any]:
        profile = state.get("profile", {})
        response_language = state.get("response_language")

        retrieval_query = None

        if not _looks_like_knowledge_question(state["user_input"]):
            # Profile is sufficient and this isn't a direct factual
            # question: build a recommendation-oriented retrieval query
            # from the gathered profile instead of the raw small-talk
            # message, so retrieval has something concrete to search for.
            parts = [
                value
                for value in (
                    profile.get("career_goal"),
                    profile.get("background"),
                    profile.get("skill_level"),
                    profile.get("interests"),
                )
                if value
            ]

            if parts:
                retrieval_query = (
                    "Suitable course recommendation for a learner with: "
                    + "; ".join(parts)
                )

        try:
            # Reach the knowledge base only through the MCP tool -- the
            # RAG internals (retriever, answer generator, translator)
            # are never imported or called directly from this agent.
            rag_result = mcp_client.ask_knowledge_base(
                question=state["user_input"],
                retrieval_query=retrieval_query,
                preferred_language=state.get("preferred_language"),
            )
        except Exception as exc:
            error_message = (
                "عذرًا، تعذّر الوصول إلى قاعدة المعرفة حاليًا."
                if response_language == "ar"
                else (
                    "Sorry, the knowledge base is temporarily unavailable."
                )
            )
            return {
                "rag_result": {
                    "answer": error_message,
                    "sources": [],
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "response": error_message,
            }

        response_text = rag_result["answer"]

        consent_pending = state.get("consent_pending", False)

        if (
            not state.get("lead_captured")
            and not state.get("consent_pending")
            and state.get("consent") is None
            and not _missing_profile_fields(profile)
        ):
            consent_prompt = (
                "\n\nهل توافق على أن نحتفظ بمعلوماتك الأساسية "
                "(الخلفية التعليمية والهدف المهني) لمتابعتك لاحقًا؟ "
                "(نعم / لا)"
                if response_language == "ar"
                else (
                    "\n\nWould you like us to save your basic details "
                    "(background and career goal) so we can follow up "
                    "with you later? (yes / no)"
                )
            )
            response_text += consent_prompt
            consent_pending = True

        return {
            "rag_result": rag_result,
            "response": response_text,
            "consent_pending": consent_pending,
        }

    return rag_answer_node


def make_handle_consent_node(lead_store: LeadStore):
    def handle_consent_node(state: RexAssistantState) -> dict[str, Any]:
        response_language = state.get("response_language") or "en"
        decision = _detect_consent(state["user_input"])

        if decision is True:
            lead_store.save_lead({"profile": state.get("profile", {})})

            response = (
                "تم حفظ بياناتك بنجاح، وسنتواصل معك قريبًا. شكرًا لك!"
                if response_language == "ar"
                else "Thanks! I've saved your details for follow-up."
            )
            return {
                "consent": True,
                "consent_pending": False,
                "lead_captured": True,
                "response": response,
            }

        if decision is False:
            response = (
                "تم تجاهل الطلب، ولن يتم حفظ أي معلومات شخصية."
                if response_language == "ar"
                else (
                    "No problem, I won't save any personal information."
                )
            )
            return {
                "consent": False,
                "consent_pending": False,
                "lead_captured": False,
                "response": response,
            }

        # Ambiguous reply: ask again instead of guessing consent.
        response = (
            "عذرًا، هل توافق على حفظ بياناتك للمتابعة؟ (نعم / لا)"
            if response_language == "ar"
            else "Sorry, could you confirm with yes or no?"
        )
        return {"consent_pending": True, "response": response}

    return handle_consent_node


def build_graph(
    lead_store: LeadStore,
    model: str = "qwen2.5:3b",
    mcp_client: Optional[RagMcpClient] = None,
):
    """
    Build and compile Agent 1's LangGraph workflow.

    Args:
        lead_store: Where consented leads are saved (see lead_store.py).
        model: Ollama chat model used for profile extraction and
            follow-up questions. Knowledge-base answers continue to use
            whatever model the RAG MCP server is configured with.
        mcp_client: Optional pre-built RagMcpClient, mainly for tests
            that want to inject a mock. Defaults to a real client that
            launches agents/rag_mcp_server.py as a subprocess on first
            use and keeps that connection open across turns.

    Returns:
        (compiled_graph, mcp_client) -- the caller should keep the
        mcp_client reference to call .close() on shutdown.
    """
    mcp_client = mcp_client or RagMcpClient()

    graph = StateGraph(RexAssistantState)

    graph.add_node("detect_language", make_detect_language_node())
    graph.add_node("update_profile", make_update_profile_node(model))
    graph.add_node("ask_followup", make_ask_followup_node(model))
    graph.add_node("rag_answer", make_rag_answer_node(mcp_client))
    graph.add_node("handle_consent", make_handle_consent_node(lead_store))

    graph.set_entry_point("detect_language")

    graph.add_conditional_edges(
        "detect_language",
        route_after_language,
        {
            "update_profile": "update_profile",
            "handle_consent": "handle_consent",
        },
    )

    graph.add_conditional_edges(
        "update_profile",
        route_after_profile,
        {
            "ask_followup": "ask_followup",
            "rag_answer": "rag_answer",
        },
    )

    graph.add_edge("ask_followup", END)
    graph.add_edge("rag_answer", END)
    graph.add_edge("handle_consent", END)

    return graph.compile(), mcp_client
