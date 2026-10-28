"""
Persists per-conversation, per-agent state as JSON blobs in SQLite.

Conversation memory (e.g. REX Assistant's learner profile, consent
status, and language preference) has a different lifetime than the
orchestrator's per-request workflow state (agents/orchestrator.py's
OrchestratorState): it must survive across many separate
handle_request() calls, and across application restarts.

This store is intentionally decoupled from the orchestrator's
StateGraph -- the orchestrator has no knowledge this exists. Only an
agent-specific adapter (see register_agents.py) reads and writes it.

Uses Python's standard-library sqlite3, so no new dependency is
required; the project's own suggested technology stack already lists
SQLite as an acceptable database choice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "conversation_state.sqlite3"


class ConversationStore:
    """SQLite-backed key-value store for (conversation_id, agent_name) state."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False because the Streamlit app and the
        # agent's own background threads (e.g. the MCP client) may call
        # this from different threads; the lock below serializes access
        # since a single sqlite3 connection is not otherwise thread-safe.
        self._connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._lock = threading.Lock()

        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    conversation_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, agent_name)
                )
                """
            )
            self._connection.commit()

    def get(
        self,
        conversation_id: str,
        agent_name: str,
    ) -> Optional[dict[str, Any]]:
        """Return the stored state for this conversation/agent, or None."""
        if not conversation_id or not agent_name:
            raise ValueError(
                "conversation_id and agent_name must be non-empty strings"
            )

        with self._lock:
            cursor = self._connection.execute(
                "SELECT state_json FROM conversation_state "
                "WHERE conversation_id = ? AND agent_name = ?",
                (conversation_id, agent_name),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def set(
        self,
        conversation_id: str,
        agent_name: str,
        state: dict[str, Any],
    ) -> None:
        """Replace the stored state for this conversation/agent."""
        if not conversation_id or not agent_name:
            raise ValueError(
                "conversation_id and agent_name must be non-empty strings"
            )

        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")

        state_json = json.dumps(state, ensure_ascii=False)

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO conversation_state
                    (conversation_id, agent_name, state_json)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, agent_name)
                DO UPDATE SET state_json = excluded.state_json
                """,
                (conversation_id, agent_name, state_json),
            )
            self._connection.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()
