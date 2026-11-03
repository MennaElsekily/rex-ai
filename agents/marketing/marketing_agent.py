"""
Agent 5 -- AI Marketing Agent.

Entry point for the orchestrator: ``MarketingAgent.process_request`` runs the LangGraph workflow (graph.py) with
- tools: the Marketing MCP server through MarketingMcpClient (started on the first tool call, so registering the agent
  starts nothing);
- the model: LangChain ChatOllama (MARKETING_LLM_MODEL, default the Curriculum Agent's model);
- conversation memory: the shared ConversationStore under "marketing" (only an open question, e.g. "which course?" or
  "who is the audience?").

Two ways in: a chat request ("Write a marketing campaign for CUR-..."), read by understanding.py, and the Marketing
tab, which passes the chosen course, confirmed audience, tone, channels and number of variations in
request["context"]["marketing_request"].

The agent writes pending-review campaigns only, and only from approved courses. Editing, approving and downloading
belong to AdminService with an administrator's token. Nothing is ever published, e-mailed or sent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from agents.config import get_setting
from agents.conversation_store import ConversationStore
from agents.curriculum_research.llm import curriculum_model, get_llm
from agents.marketing import marketing_store, marketing_tools, understanding as rules
from agents.marketing.graph import AGENT_NAME

logger = logging.getLogger(__name__)

MODEL_SETTING = "MARKETING_LLM_MODEL"
# An open question ("which course?", "who is the audience?") is forgotten after this long, so a message sent much later is
# never taken as its answer.
PENDING_TTL_SECONDS = 15 * 60


def marketing_model() -> str:
    return get_setting(MODEL_SETTING, "") or curriculum_model()


def default_marketing_tools() -> Any:
    """
    The Marketing MCP server, on the same databases the admin dashboard reads. tests/conftest.py replaces this with the
    in-process toolbox on temporary storage.
    """
    from agents.curriculum_research import curriculum_store
    from agents.marketing.mcp_client import MarketingMcpClient

    env = {marketing_tools.DB_SETTING: get_setting(marketing_tools.DB_SETTING, "") or str(marketing_store.DEFAULT_DB_PATH),
           "CURRICULUM_DB_PATH": get_setting("CURRICULUM_DB_PATH", "") or str(curriculum_store.DEFAULT_DB_PATH)}
    return MarketingMcpClient(env=env)


def default_llm(model: Optional[str] = None) -> Any:
    """The LangChain chat model (tests/conftest.py replaces this)."""
    return get_llm(model or marketing_model())


class MarketingAgent:
    AGENT_NAME = AGENT_NAME

    def __init__(self, conversation_store: Optional[ConversationStore] = None, tools: Any = None,
                 llm: Any = None, model: Optional[str] = None, clock: Callable[[], float] = time.time) -> None:
        self.conversation_store = conversation_store
        self._clock = clock
        self._tools = tools
        self._llm = llm
        self.model = model
        self._graph = None

    # Built on first use, so registering the agent starts no server and loads no model.
    @property
    def tools(self) -> Any:
        if self._tools is None:
            self._tools = default_marketing_tools()
        return self._tools

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = default_llm(self.model)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            from agents.marketing.graph import build_graph
            self._graph = build_graph(self.tools, self.llm)
        return self._graph

    # ---------------------------------------------------------------- entry

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id", "unknown")
        text = " ".join(str(request.get("user_input") or "").split())
        context = request.get("context") or {}
        ui = "ar" if context.get("language") == "ar" else "en"
        form = context.get("marketing_request")
        if not text:
            return self._envelope(request_id, "error", None, "The marketing request cannot be empty."
                                  if ui == "en" else "لا يمكن أن يكون الطلب التسويقي فارغًا.")
        if form is not None and not isinstance(form, dict):
            return self._envelope(request_id, "error", None, "marketing_request must be an object.")
        memory = self._load(context)
        progress = context.get("progress")
        try:
            state = self.graph.invoke({
                "request_text": text, "request_id": request_id, "ui_language": ui, "form": form or {}, "memory": memory,
                "previous_results": list(context.get("previous_results") or []),
                "progress": progress if callable(progress) else None})
        except Exception as exc:  # tool server down, unexpected failure: an answer, never a crash
            logger.exception("Marketing workflow failed")
            message = (f"The marketing request could not be completed ({type(exc).__name__}: {exc})."
                       if ui == "en" else f"تعذر إكمال الطلب التسويقي ({type(exc).__name__}).")
            return self._envelope(request_id, "error", None, message)
        if state.get("memory_out") is not None:
            out = dict(state["memory_out"])
            pending = out.get("pending")
            if isinstance(pending, dict) and "created_at" not in pending:
                out["pending"] = {**pending, "created_at": self._clock()}               # the question is stamped when it is first asked
            self._save(context, {**memory, **out})
        response = state.get("response") or {"kind": "error", "message": "No response."}
        status = state.get("status") or "error"
        return self._envelope(request_id, status, response, response["message"] if status == "error" else None)

    def wants_follow_up(self, request: dict[str, Any]) -> bool:
        """
        Orchestrator continuation check: this conversation is waiting for this agent's question (which course? who is
        the audience?) and the message answers it.
        """
        pending = self._load(request.get("context") or {}).get("pending")      # (an expired question is not returned)
        text = str(request.get("user_input") or "")
        if not pending or not text.strip():
            return False
        kind = rules.classify_reply(text)[0]
        if pending.get("awaiting") == "audience":
            # Only a real answer, a cancellation, or a reply that needs asking again. An acknowledgement ("thanks"), a question
            # or a request for something else is not claimed, and never becomes the audience.
            return kind in ("audience", "cancel", "ambiguous", "invalid")
        return kind == "cancel" or bool(rules.find_curriculum_id(text)
                                        or rules.choose_from_options(text, pending.get("course_ids") or []))

    def close(self) -> None:
        if self._tools is not None and hasattr(self._tools, "close"):
            self._tools.close()

    # --------------------------------------------------------------- memory

    @staticmethod
    def _conversation_id(context: dict[str, Any]) -> Optional[str]:
        value = context.get("conversation_id")
        return value if isinstance(value, str) and value.strip() else None

    def _load(self, context: dict[str, Any]) -> dict[str, Any]:
        conversation_id = self._conversation_id(context)
        if self.conversation_store is None or conversation_id is None:
            return {}
        try:
            stored = self.conversation_store.get(conversation_id, AGENT_NAME)
        except Exception:
            return {}
        if not isinstance(stored, dict):
            return {}
        if isinstance(stored.get("pending"), dict) and self.pending_expired(stored["pending"]):
            stored = {k: v for k, v in stored.items() if k != "pending"}
        return stored

    def pending_expired(self, pending: dict[str, Any]) -> bool:
        """An open question with no time stamp (an older format) or older than PENDING_TTL_SECONDS is expired."""
        created = pending.get("created_at")
        return not isinstance(created, (int, float)) or self._clock() - created > PENDING_TTL_SECONDS

    def _save(self, context: dict[str, Any], memory: dict[str, Any]) -> None:
        conversation_id = self._conversation_id(context)
        if self.conversation_store is None or conversation_id is None:
            return
        try:
            self.conversation_store.set(conversation_id, AGENT_NAME, dict(memory))
        except Exception:
            logger.warning("Could not store the marketing conversation state", exc_info=True)

    def _envelope(self, request_id: str, status: str, result: Optional[dict[str, Any]],
                  error_message: Optional[str]) -> dict[str, Any]:
        return {"request_id": request_id, "agent_name": AGENT_NAME, "status": status, "result": result,
                "error_message": error_message}
