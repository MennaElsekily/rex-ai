"""
Agent 3 -- Curriculum Research & Accreditation Alignment Agent.

Entry point for the orchestrator: ``CurriculumAgent.process_request``
runs the LangGraph workflow (graph.py) with
- tools: the Curriculum MCP server through CurriculumMcpClient (started on
  the first tool call);
- the model: LangChain ChatOllama (llm.py, CURRICULUM_LLM_MODEL);
- conversation memory: the shared ConversationStore under
  "curriculum_research" (active curriculum, an open question, the last
  suggestions), so follow-ups need not repeat anything.

The agent drafts and revises; it can never approve (AdminService does,
with an administrator's token).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.conversation_store import ConversationStore
from agents.curriculum_research import curriculum_store, curriculum_tools, understanding as rules
from agents.curriculum_research.llm import curriculum_model, get_llm
from agents.curriculum_research.request_parser import detect_refinements
from agents.curriculum_research.revisions import detect_changes
from agents.curriculum_research.state import ConversationMemory
from knowledge_base.language import detect_conversation_language

logger = logging.getLogger(__name__)

AGENT_NAME = "curriculum_research"


def default_curriculum_tools() -> Any:
    """
    The Curriculum MCP server, on the same database and documents folder
    the admin dashboard reads. tests/conftest.py replaces this with the
    in-process toolbox on temporary storage.
    """
    from agents.curriculum_research.mcp_client import CurriculumMcpClient
    from agents.config import get_setting

    env = {curriculum_tools.DB_SETTING: get_setting(curriculum_tools.DB_SETTING, "")
           or str(curriculum_store.DEFAULT_DB_PATH),
           curriculum_tools.DOCUMENTS_SETTING: get_setting(curriculum_tools.DOCUMENTS_SETTING, "")
           or str(curriculum_tools.DEFAULT_DOCUMENTS_DIR)}
    return CurriculumMcpClient(env=env)


def default_llm(model: Optional[str] = None) -> Any:
    """The LangChain chat model (tests/conftest.py replaces this)."""
    return get_llm(model or curriculum_model())


class CurriculumAgent:
    AGENT_NAME = AGENT_NAME

    def __init__(self, conversation_store: Optional[ConversationStore] = None, tools: Any = None,
                 llm: Any = None, model: Optional[str] = None) -> None:
        self.conversation_store = conversation_store
        self._tools = tools
        self._llm = llm
        self.model = model
        self._graph = None

    # Built on first use, so registering the agent starts no server and
    # loads no model.
    @property
    def tools(self) -> Any:
        if self._tools is None:
            self._tools = default_curriculum_tools()
        return self._tools

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = default_llm(self.model)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            from agents.curriculum_research.graph import build_graph
            self._graph = build_graph(self.tools, self.llm)
        return self._graph

    # ---------------------------------------------------------------- entry

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id", "unknown")
        text = " ".join(str(request.get("user_input") or "").split())
        context = request.get("context") or {}
        memory = self._load(context)
        language = detect_conversation_language(text) or memory.get("language") or "en"
        if not text:
            return self._envelope(request_id, "error", None, "The curriculum request cannot be empty."
                                  if language == "en" else "لا يمكن أن يكون طلب المنهج فارغًا.")
        try:
            state = self.graph.invoke({"request_text": text, "request_id": request_id, "memory": memory})
        except Exception as exc:  # tool server down, unexpected failure: an answer, never a crash
            logger.exception("Curriculum workflow failed")
            message = (f"The curriculum request could not be completed ({type(exc).__name__}: {exc})."
                       if language == "en" else f"تعذر إكمال طلب المنهج ({type(exc).__name__}).")
            return self._envelope(request_id, "error", None, message)
        if state.get("memory_out") is not None:
            self._save(context, state["memory_out"])
        response = state.get("response") or {"kind": "error", "message": "No response."}
        status = state.get("status") or "success"
        return self._envelope(request_id, status, response, response["message"] if status == "error" else None)

    def wants_follow_up(self, request: dict[str, Any]) -> bool:
        """
        Orchestrator continuation check: this conversation is waiting for
        this agent (an open question, a suggestion list, an active draft)
        and the message continues it ("40 hours", "generate suggestion 2",
        "add a module on Git"). Never true without such state.
        """
        memory = self._load(request.get("context") or {})
        text = str(request.get("user_input") or "")
        if not memory or not text.strip():
            return False
        if memory.get("pending"):
            answer = rules.answer_slots(text)
            if any(answer.get(k) for k in ("total_hours", "level", "course_ref")) or rules.accepts_defaults(text):
                return True
            if "topic" in (memory["pending"].get("missing") or []) and answer.get("topic") \
                    and len(text.split()) <= 8:
                return True
        if memory.get("last_suggestions") and rules.suggestion_choice(text):
            return True
        if memory.get("active_curriculum_id"):
            shape = {"language": memory.get("active_language", "en"), "level": memory.get("active_level"),
                     "capstone": None}
            if (detect_changes(text, shape) or detect_refinements(text) or rules.is_approve_attempt(text)
                    or rules.is_alignment_request(text) or rules.is_show_request(text)):
                return True
        return False

    def close(self) -> None:
        if self._tools is not None and hasattr(self._tools, "close"):
            self._tools.close()

    # --------------------------------------------------------------- memory

    @staticmethod
    def _conversation_id(context: dict[str, Any]) -> Optional[str]:
        value = context.get("conversation_id")
        return value if isinstance(value, str) and value.strip() else None

    def _load(self, context: dict[str, Any]) -> ConversationMemory:
        conversation_id = self._conversation_id(context)
        if self.conversation_store is None or conversation_id is None:
            return {}
        try:
            stored = self.conversation_store.get(conversation_id, AGENT_NAME)
        except Exception:
            return {}
        return stored if isinstance(stored, dict) else {}

    def _save(self, context: dict[str, Any], memory: ConversationMemory) -> None:
        conversation_id = self._conversation_id(context)
        if self.conversation_store is None or conversation_id is None:
            return
        try:
            self.conversation_store.set(conversation_id, AGENT_NAME, dict(memory))
        except Exception:
            logger.warning("Could not store the curriculum conversation state", exc_info=True)

    def _envelope(self, request_id: str, status: str, result: Optional[dict[str, Any]],
                  error_message: Optional[str]) -> dict[str, Any]:
        return {"request_id": request_id, "agent_name": AGENT_NAME, "status": status, "result": result,
                "error_message": error_message}
