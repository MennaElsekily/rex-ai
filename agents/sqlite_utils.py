"""
SQLite connection setup and write transactions shared by the agents' stores.

Every agent still owns its own database file (curricula, documents, marketing, tenders, conversation state): nothing here
shares a database. This module only makes the way they are opened and written the same, because the Streamlit process and
the MCP server processes open the same files at the same time.

- ``connect``: a busy timeout (a writer waits for another writer instead of failing at once) and foreign-key enforcement.
  The journal mode is left alone: WAL would change the database file itself, add -wal/-shm side files and does not suit
  databases kept in a synced folder, and the default rollback journal is enough for this write volume.
- ``write_transaction``: one serialised write. It takes the process lock, then ``BEGIN IMMEDIATE``, which takes the database's
  write lock BEFORE anything is read, so two processes that both "check, then write" (approve a version, take the next
  version number) cannot both pass the check. It commits on success and rolls back on any exception. A database that stays
  locked is reported as the store's own error ("busy; nothing was changed") and is never retried automatically, so a write
  cannot be applied twice.

No transaction is ever open while a language model, the network or an MCP tool is called: the stores' methods only run
SQL inside it.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Type

BUSY_TIMEOUT_SECONDS = 30.0
BUSY_MESSAGE = "The database is busy with another change. Nothing was changed; try again in a moment."


class DatabaseBusy(RuntimeError):
    """The database stayed locked past the busy timeout. The write was not applied."""


def connect(path: str | Path) -> sqlite3.Connection:
    """A connection other threads may use (the stores serialise access with their own lock) with the project's settings."""
    connection = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_SECONDS, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def is_busy(error: BaseException) -> bool:
    return isinstance(error, sqlite3.OperationalError) and ("locked" in str(error).lower() or "busy" in str(error).lower())


def _rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.rollback()
    except sqlite3.Error:
        pass


@contextmanager
def write_transaction(connection: sqlite3.Connection, lock: threading.Lock,
                      error_class: Type[Exception] = DatabaseBusy) -> Iterator[sqlite3.Connection]:
    """See the module docstring. error_class: what a busy database raises (a store passes its own ValueError subclass)."""
    with lock:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if is_busy(exc):
                raise error_class(BUSY_MESSAGE) from exc
            raise
        try:
            yield connection
            connection.commit()
        except BaseException as exc:
            _rollback(connection)
            if is_busy(exc):
                raise error_class(BUSY_MESSAGE) from exc
            raise
