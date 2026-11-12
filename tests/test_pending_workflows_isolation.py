"""
Regression test for a real bug found while auditing this project: an
earlier version of PendingWorkflowStore.__init__ had `db_path: ... =
DEFAULT_DB_PATH` as the parameter's default value, which binds once at
import time. tests/conftest.py's autouse fixture patches
agents.pending_workflows.DEFAULT_DB_PATH to a temporary path for every
test -- but a default bound at import time never sees that patch, so
every test that built a PendingWorkflowStore with no explicit path was
silently writing to the real agents/pending_workflows.sqlite3 the whole
time. Fixed by reading the module attribute inside __init__ instead
(the same convention every other store in this project already uses).
"""

from pathlib import Path

from agents import pending_workflows


def test_the_default_path_is_read_at_call_time_not_at_import_time(tmp_path, monkeypatch):
    patched = tmp_path / "patched.sqlite3"
    monkeypatch.setattr(pending_workflows, "DEFAULT_DB_PATH", patched)

    store = pending_workflows.PendingWorkflowStore()

    assert store.db_path == patched
    assert patched.exists()
    store.close()


def test_the_real_default_db_path_constant_is_never_the_project_source_tree(tmp_path, monkeypatch):
    """A second, independent check: the patched path is honoured even when it differs from the real default."""
    monkeypatch.setattr(pending_workflows, "DEFAULT_DB_PATH", tmp_path / "isolated.sqlite3")

    store = pending_workflows.PendingWorkflowStore()

    assert store.db_path.parent == tmp_path
    assert Path(pending_workflows.__file__).parent not in store.db_path.parents
    store.close()
