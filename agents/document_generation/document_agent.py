"""
Agent 4 -- Educational Content & Document Generation Agent.

Entry point for the orchestrator: ``DocumentAgent.process_request`` runs
the LangGraph workflow (graph.py) with
- tools: the Document MCP server through DocumentMcpClient (started on the
  first tool call, so registering the agent starts nothing);
- the model: LangChain ChatOllama (DOCUMENT_LLM_MODEL, default the
  Curriculum Agent's model);
- conversation memory: the shared ConversationStore under
  "document_generation" (only an open question, e.g. "which curriculum?").

Two ways in: a chat request ("Generate a course description for CUR-..."),
read by understanding.py, and the Documents tab form, which passes the
chosen document type and curriculum in request["context"]["document_request"]
so no keyword guessing is needed.

The agent writes drafts only. Approving a document, editing it and
downloading it belong to AdminService with an administrator's token.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.config import get_setting
from agents.conversation_store import ConversationStore
from agents.curriculum_research.llm import curriculum_model, get_llm
from agents.document_generation import document_store, document_tools, understanding as rules
from agents.document_generation.graph import AGENT_NAME

logger = logging.getLogger(__name__)

MODEL_SETTING = "DOCUMENT_LLM_MODEL"


def document_model() -> str:
    return get_setting(MODEL_SETTING, "") or curriculum_model()


def default_document_tools() -> Any:
    """
    The Document MCP server, on the same databases and folders the admin
    dashboard reads. tests/conftest.py replaces this with the in-process
    toolbox on temporary storage.
    """
    from agents.curriculum_research import curriculum_store
    from agents.document_generation.mcp_client import DocumentMcpClient

    env = {document_tools.DB_SETTING: get_setting(document_tools.DB_SETTING, "") or str(document_store.DEFAULT_DB_PATH),
           document_tools.DIR_SETTING: get_setting(document_tools.DIR_SETTING, "") or str(document_tools.DEFAULT_DOCUMENTS_DIR),
           "CURRICULUM_DB_PATH": get_setting("CURRICULUM_DB_PATH", "") or str(curriculum_store.DEFAULT_DB_PATH)}
    return DocumentMcpClient(env=env)


def default_llm(model: Optional[str] = None) -> Any:
    """The LangChain chat model (tests/conftest.py replaces this)."""
    return get_llm(model or document_model())


class DocumentAgent:
    AGENT_NAME = AGENT_NAME

    def __init__(self, conversation_store: Optional[ConversationStore] = None, tools: Any = None,
                 llm: Any = None, model: Optional[str] = None) -> None:
        self.conversation_store = conversation_store
        self._tools = tools
        self._llm = llm
        self.model = model
        self._graph = None

    # Built on first use, so registering the agent starts no server and loads no model.
    @property
    def tools(self) -> Any:
        if self._tools is None:
            self._tools = default_document_tools()
        return self._tools

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = default_llm(self.model)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            from agents.document_generation.graph import build_graph
            self._graph = build_graph(self.tools, self.llm)
        return self._graph

    # ---------------------------------------------------------------- entry

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id", "unknown")
        text = " ".join(str(request.get("user_input") or "").split())
        context = request.get("context") or {}
        ui = "ar" if context.get("language") == "ar" else "en"
        form = context.get("document_request")
        if not text:
            return self._envelope(request_id, "error", None, "The document request cannot be empty."
                                  if ui == "en" else "لا يمكن أن يكون طلب الوثيقة فارغًا.")
        if form is not None and not isinstance(form, dict):
            return self._envelope(request_id, "error", None, "document_request must be an object.")
        memory = self._load(context)
        progress = context.get("progress")
        try:
            state = self.graph.invoke({"request_text": text, "request_id": request_id, "ui_language": ui,
                                       "form": form or {}, "memory": memory,
                                       "progress": progress if callable(progress) else None})
        except Exception as exc:  # tool server down, unexpected failure: an answer, never a crash
            logger.exception("Document workflow failed")
            message = (f"The document request could not be completed ({type(exc).__name__}: {exc})."
                       if ui == "en" else f"تعذر إكمال طلب الوثيقة ({type(exc).__name__}).")
            return self._envelope(request_id, "error", None, message)
        if state.get("memory_out") is not None:
            self._save(context, {**memory, **state["memory_out"]})
        response = state.get("response") or {"kind": "error", "message": "No response."}
        status = state.get("status") or "error"
        return self._envelope(request_id, status, response, response["message"] if status == "error" else None)

    def wants_follow_up(self, request: dict[str, Any]) -> bool:
        """
        Orchestrator continuation check: this conversation is waiting for
        this agent's question (which curriculum? which document?) and the
        message answers it ("2", "CUR-...", "a participant guide").
        """
        memory = self._load(request.get("context") or {})
        text = str(request.get("user_input") or "")
        pending = memory.get("pending")
        if not pending or not text.strip():
            return False
        return bool(rules.find_curriculum_id(text) or rules.choose_from_options(text, pending.get("options") or [])
                    or (not pending.get("doc_type") and rules.find_document_type(text)))

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
        return stored if isinstance(stored, dict) else {}

    def _save(self, context: dict[str, Any], memory: dict[str, Any]) -> None:
        conversation_id = self._conversation_id(context)
        if self.conversation_store is None or conversation_id is None:
            return
        try:
            self.conversation_store.set(conversation_id, AGENT_NAME, dict(memory))
        except Exception:
            logger.warning("Could not store the document conversation state", exc_info=True)

    def _envelope(self, request_id: str, status: str, result: Optional[dict[str, Any]],
                  error_message: Optional[str]) -> dict[str, Any]:
        return {"request_id": request_id, "agent_name": AGENT_NAME, "status": status, "result": result,
                "error_message": error_message}
