"""
Pauses a multi-agent Ask Agents request at the curriculum approval gate,
and automatically resumes it afterwards -- through the SAME Central
Orchestrator call it would have used anyway (agents/orchestrator.py is
never duplicated or bypassed; see admin_service.py's ask()/
approve_curriculum()/request_curriculum_changes()).

Why this exists: "Create a curriculum for data analysis and generate a
course description document" asks for two chained agents, but Document
Generation always refuses to work from a curriculum that is not yet
admin-approved (a deliberate, pre-existing safety gate this project does
not weaken). Without this module, the admin would see the document step
fail and have to re-type the whole request after approving. With it:

1. admin_service.ask() notices a chained plan stopped at "the curriculum
   needs approval first" and saves the still-to-run agents here, keyed
   by the curriculum_id it is waiting on.
2. admin_service.approve_curriculum() checks this store; if something
   was waiting on the curriculum it just approved, it resumes by calling
   the orchestrator again with the ORIGINAL request text and
   context["skip_agents"] set to the agents already done (so curriculum
   generation is not repeated) -- see Orchestrator._plan_node.
3. The result is kept (one per conversation) for agents_tab() to pick
   up and show as a new assistant turn on its next render -- the admin
   never has to ask again.
4. admin_service.request_curriculum_changes() cancels (never silently
   resumes) anything waiting on a rejected curriculum.

SQLite-backed, so a paused workflow survives an application restart,
same as every other store in this project.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.sqlite_utils import DatabaseBusy, connect, write_transaction

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "pending_workflows.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PendingWorkflowStore:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # The default is read here, not as the parameter's default value, so tests (and
        # anything else) that patch DEFAULT_DB_PATH after this module is imported still take
        # effect -- a bare "= DEFAULT_DB_PATH" default binds once, at import time, and would
        # silently ignore any later patch.
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(self.db_path)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pending_workflows (
                    curriculum_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    remaining_agents_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resumed_results (
                    conversation_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._connection.commit()

    # ------------------------------------------------------------- paused

    def save_pending(self, curriculum_id: str, conversation_id: str, request_text: str,
                     context: dict[str, Any], remaining_requests: list[dict[str, Any]]) -> None:
        """
        remaining_requests: one dict per agent still waiting, each with
        at least "agent_name"; a document_generation/marketing entry also
        carries the exact request it had already resolved before it
        refused (document_type/options/...), captured from that agent's
        own "not_approved" result, so resuming can hand it back directly
        instead of re-matching the original free text.
        """
        with write_transaction(self._connection, self._lock, DatabaseBusy):
            self._connection.execute(
                "INSERT INTO pending_workflows (curriculum_id, conversation_id, request_text, "
                "context_json, remaining_agents_json, created_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(curriculum_id) DO UPDATE SET conversation_id = excluded.conversation_id, "
                "request_text = excluded.request_text, context_json = excluded.context_json, "
                "remaining_agents_json = excluded.remaining_agents_json, created_at = excluded.created_at",
                (curriculum_id, conversation_id, request_text, json.dumps(context, ensure_ascii=False),
                 json.dumps(remaining_requests, ensure_ascii=False), _now()),
            )

    def pending_for_curriculum(self, curriculum_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT conversation_id, request_text, context_json, remaining_agents_json "
                "FROM pending_workflows WHERE curriculum_id = ?", (curriculum_id,),
            ).fetchone()
        if row is None:
            return None
        conversation_id, request_text, context_json, remaining_json = row
        return {"curriculum_id": curriculum_id, "conversation_id": conversation_id, "request_text": request_text,
                "context": json.loads(context_json), "remaining_requests": json.loads(remaining_json)}

    def clear_pending(self, curriculum_id: str) -> None:
        with write_transaction(self._connection, self._lock, DatabaseBusy):
            self._connection.execute("DELETE FROM pending_workflows WHERE curriculum_id = ?", (curriculum_id,))

    # ------------------------------------------------------------ resumed

    def save_resumed_result(self, conversation_id: str, result: dict[str, Any]) -> None:
        with write_transaction(self._connection, self._lock, DatabaseBusy):
            self._connection.execute(
                "INSERT INTO resumed_results (conversation_id, result_json, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET result_json = excluded.result_json, "
                "created_at = excluded.created_at",
                (conversation_id, json.dumps(result, ensure_ascii=False), _now()))

    def pop_resumed_result(self, conversation_id: str) -> Optional[dict[str, Any]]:
        """The result of a resumed workflow for this conversation, if one finished since it was last checked."""
        with write_transaction(self._connection, self._lock, DatabaseBusy):
            row = self._connection.execute(
                "SELECT result_json FROM resumed_results WHERE conversation_id = ?", (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute("DELETE FROM resumed_results WHERE conversation_id = ?", (conversation_id,))
        return json.loads(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()
