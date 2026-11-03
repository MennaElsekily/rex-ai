"""
How the agents' SQLite stores are opened and written: foreign keys on, a busy timeout, the journal mode left alone, writes that
take the write lock before they read, rollback on failure, clean shutdown, compatibility with databases made before these
changes, and concurrent use by threads and by separate processes (each really run here). Each store keeps its own database.
"""

import multiprocessing
import sqlite3
import threading
import time

import pytest

from agents import sqlite_utils
from agents.conversation_store import ConversationStore
from agents.curriculum_research.curriculum_store import CurriculumStore, CurriculumStoreError
from agents.document_generation.document_store import DocumentStore
from agents.marketing.marketing_store import MarketingStore, MarketingStoreError
from agents.rfp_research.application_profile import ApplicationProfile
from agents.rfp_research.opportunity_store import OpportunityStore
from tests import sqlite_workers
from tests.curriculum_fixtures import valid_curriculum

ADMIN = "admin-session-test0001"
VALID = {"valid": True, "errors": []}
CAMPAIGN = {"campaign_type": "marketing_campaign", "title": "Course", "curriculum_id": "CUR-0123456789ab", "curriculum_version": 1, "language": "en"}


def curriculum(store, valid=True):
    return store.save_version(valid_curriculum(), {"valid": valid, "errors": []}, [])["curriculum_id"]


def make_stores(tmp_path):
    return {
        "conversation": ConversationStore(tmp_path / "c.sqlite3"), "curricula": CurriculumStore(tmp_path / "cur.sqlite3"),
        "documents": DocumentStore(tmp_path / "d.sqlite3"), "marketing": MarketingStore(tmp_path / "m.sqlite3"),
        "tenders": OpportunityStore(db_path=tmp_path / "t.sqlite3"), "profile": ApplicationProfile(tmp_path / "p.sqlite3")}


# ---- connection settings -------------------------------------------------------------------------------------------------------------------

def test_every_store_opens_its_database_with_the_shared_settings(tmp_path):
    for name, store in make_stores(tmp_path).items():
        connection = store._connection
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1, name
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == int(sqlite_utils.BUSY_TIMEOUT_SECONDS * 1000), name
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete", name        # the journal mode is not changed (no WAL)


def test_each_agent_keeps_its_own_database_file(tmp_path):
    stores = make_stores(tmp_path)
    paths = {name: store.db_path for name, store in stores.items()}
    assert len(set(paths.values())) == len(paths) == 6
    for name, store in stores.items():
        tables = {r[0] for r in store._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        others = set().union(*({r[0] for r in s._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                               for n, s in stores.items() if n not in (name, "tenders", "profile")))
        if name not in ("tenders", "profile"):
            assert not (tables - {"sqlite_sequence"}) & others, name                            # no table is shared between agents' databases


def test_foreign_keys_are_enforced_where_a_table_declares_them(tmp_path):
    connection = sqlite_utils.connect(tmp_path / "fk.sqlite3")
    connection.executescript("CREATE TABLE parent (id INTEGER PRIMARY KEY); CREATE TABLE child (parent_id INTEGER REFERENCES parent(id));")
    connection.execute("INSERT INTO parent VALUES (1)")
    connection.execute("INSERT INTO child VALUES (1)")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO child VALUES (99)")
    connection.close()


# ---- transactions ------------------------------------------------------------------------------------------------------------------------------

def test_a_failed_write_is_rolled_back_completely(tmp_path):
    connection, lock = sqlite_utils.connect(tmp_path / "t.sqlite3"), threading.Lock()
    connection.execute("CREATE TABLE t (v TEXT)")
    connection.commit()
    with pytest.raises(ValueError):
        with sqlite_utils.write_transaction(connection, lock):
            connection.execute("INSERT INTO t VALUES ('half done')")
            raise ValueError("fails after the first statement")
    assert connection.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    with sqlite_utils.write_transaction(connection, lock):                                  # and the lock and connection are usable again
        connection.execute("INSERT INTO t VALUES ('fine')")
    assert connection.execute("SELECT v FROM t").fetchall() == [("fine",)]
    connection.close()


def test_a_store_write_that_fails_midway_changes_nothing(tmp_path, monkeypatch):
    store = CurriculumStore(tmp_path / "cur.sqlite3")
    cid = curriculum(store)
    monkeypatch.setattr(store, "_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("audit failed")))

    with pytest.raises(RuntimeError):
        store.approve(cid, 1, ADMIN)

    monkeypatch.undo()
    assert store.get(cid)["status"] == "pending_review" and store.get(cid)["approved_version"] is None   # not half-approved
    assert [r["event"] for r in store.audit_trail(cid)] == ["version_saved"]
    store.approve(cid, 1, ADMIN)                                                                       # and it can still be approved
    assert store.get(cid)["approved_version"] == 1


def test_a_failed_new_version_leaves_no_trace(tmp_path, monkeypatch):
    store = MarketingStore(tmp_path / "m.sqlite3")
    first = store.save_version(CAMPAIGN, VALID)["campaign_id"]
    monkeypatch.setattr(store, "_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        store.save_version({**CAMPAIGN, "marker": "v2"}, VALID, campaign_id=first)
    monkeypatch.undo()

    assert store.get(first)["current_version"] == 1 and [v["version"] for v in store.versions(first)] == [1]
    assert store.save_version({**CAMPAIGN, "marker": "v2"}, VALID, campaign_id=first)["version"] == 2


# ---- busy timeout and locking errors -------------------------------------------------------------------------------------------------------------

def test_a_locked_database_is_reported_as_busy_and_the_write_is_not_applied_or_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_utils, "BUSY_TIMEOUT_SECONDS", 0.3)
    path = tmp_path / "cur.sqlite3"
    store = CurriculumStore(path)
    cid = curriculum(store)
    blocker = sqlite3.connect(str(path), timeout=0.1)
    blocker.execute("BEGIN IMMEDIATE")                                                              # another process holds the write lock

    started = time.monotonic()
    with pytest.raises(CurriculumStoreError, match="busy"):
        store.approve(cid, 1, ADMIN)
    waited = time.monotonic() - started
    assert 0.2 <= waited < 5                                                                        # it waited the timeout, once, then gave up

    blocker.rollback()
    blocker.close()
    assert store.get(cid)["status"] == "pending_review"                                              # nothing was changed
    assert [r["event"] for r in store.audit_trail(cid)] == ["version_saved"]
    store.approve(cid, 1, ADMIN)                                                                     # applied once, when the caller tries again
    assert [r["event"] for r in store.audit_trail(cid)] == ["version_saved", "review"]


def test_the_other_stores_report_a_busy_database_in_their_own_error_type(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_utils, "BUSY_TIMEOUT_SECONDS", 0.2)
    market, state = MarketingStore(tmp_path / "m.sqlite3"), ConversationStore(tmp_path / "c.sqlite3")
    for path in (market.db_path, state.db_path):
        blocker = sqlite3.connect(str(path), timeout=0.1)
        blocker.execute("BEGIN IMMEDIATE")
        if path == market.db_path:
            with pytest.raises(MarketingStoreError, match="busy"):
                market.save_version(CAMPAIGN, VALID)
        else:
            with pytest.raises(sqlite_utils.DatabaseBusy):
                state.set("c", "agent", {"a": 1})
        blocker.rollback()
        blocker.close()
    assert market.list() == [] and state.get("c", "agent") is None


# ---- clean shutdown ------------------------------------------------------------------------------------------------------------------------------

def test_closing_a_store_twice_is_safe_and_releases_the_file(tmp_path):
    for name, store in make_stores(tmp_path).items():
        path = store.db_path
        store.close()
        store.close()                                                                               # a second close is harmless
        path.rename(path.with_suffix(".moved"))                                                     # (the file is not held open: Windows would refuse)
        assert path.with_suffix(".moved").exists(), name


def test_using_a_closed_store_fails_clearly(tmp_path):
    store = MarketingStore(tmp_path / "m.sqlite3")
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.counts()


# ---- databases made before these changes ---------------------------------------------------------------------------------------------------------------

def test_an_existing_curriculum_database_keeps_every_record_and_still_works(tmp_path):
    path = tmp_path / "old.sqlite3"
    first = CurriculumStore(path)
    approved, pending = curriculum(first), curriculum(first)
    first.approve(approved, 1, ADMIN, "kept")
    first.close()

    reopened = CurriculumStore(path)

    assert reopened.get(approved)["approved_version"] == 1 and reopened.get(pending)["status"] == "pending_review"
    assert [r["event"] for r in reopened.audit_trail(approved)] == ["version_saved", "review"]
    assert reopened.list_approved() and reopened.count_approved() == 1
    reopened.approve(pending, 1, ADMIN)
    assert reopened.count_approved() == 2


def test_a_document_database_without_the_later_columns_is_migrated_in_place(tmp_path):
    path = tmp_path / "documents.sqlite3"
    raw = sqlite3.connect(str(path))
    raw.executescript("""
        CREATE TABLE documents (document_id TEXT PRIMARY KEY, curriculum_id TEXT NOT NULL, curriculum_version INTEGER NOT NULL,
            document_type TEXT NOT NULL, language TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL,
            current_version INTEGER NOT NULL, approved_version INTEGER, approved_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE document_versions (document_id TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
            curriculum_version INTEGER NOT NULL, content_json TEXT NOT NULL, validation_json TEXT NOT NULL, change_summary TEXT,
            filename TEXT, sha256 TEXT, pages INTEGER, size_bytes INTEGER, created_at TEXT NOT NULL, created_by TEXT NOT NULL,
            PRIMARY KEY (document_id, version));
        CREATE TABLE document_audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL, version INTEGER,
            event TEXT NOT NULL, previous_status TEXT, new_status TEXT, actor TEXT NOT NULL, notes TEXT, at TEXT NOT NULL);
        INSERT INTO documents VALUES ('GDOC-0123456789ab', 'CUR-0123456789ab', 1, 'course_description', 'en', 'Old title', 'pending_review', 1, NULL, NULL, 't', 't');
        INSERT INTO document_versions (document_id, version, status, curriculum_version, content_json, validation_json, created_at, created_by)
            VALUES ('GDOC-0123456789ab', 1, 'pending_review', 1, '{"title": "Old title"}', '{"valid": true}', 't', 'system:document_agent');
    """)
    raw.commit()
    raw.close()

    store = DocumentStore(path)
    DocumentStore(path).close()                                                                      # opening again changes nothing

    record = store.get("GDOC-0123456789ab")
    assert record["title"] == "Old title" and record["version_status"] == "pending_review" and record["approved_filename"] is None
    columns = {r[1] for r in store._connection.execute("PRAGMA table_info(document_versions)")}
    assert {"approved_filename", "approved_sha256", "approved_pages", "approved_size_bytes"} <= columns


def test_an_existing_marketing_database_is_reopened_unchanged(tmp_path):
    path = tmp_path / "m.sqlite3"
    first = MarketingStore(path)
    cid = first.save_version(CAMPAIGN, VALID)["campaign_id"]
    first.approve(cid, 1, ADMIN)
    first.close()
    again = MarketingStore(path)
    assert again.get_approved(cid)["version"] == 1 and again.counts() == {"draft": 0, "pending_review": 0, "approved": 1}


# ---- threads --------------------------------------------------------------------------------------------------------------------------------------

def run_threads(target, count):
    barrier, outcomes = threading.Barrier(count), []

    def runner(n):
        barrier.wait(timeout=30)
        try:
            outcomes.append(("ok", target(n)))
        except Exception as exc:
            outcomes.append(("error", f"{type(exc).__name__}: {exc}"))

    threads = [threading.Thread(target=runner, args=(n,)) for n in range(count)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    return outcomes


def test_many_threads_can_read_at_the_same_time(tmp_path):
    store = CurriculumStore(tmp_path / "cur.sqlite3")
    cid = curriculum(store)
    outcomes = run_threads(lambda n: [store.get(cid)["title"] for _ in range(50)], 8)
    assert [o[0] for o in outcomes] == ["ok"] * 8 and all(len(set(o[1])) == 1 for o in outcomes)


def test_writes_from_several_connections_to_one_file_are_all_applied_once(tmp_path):
    path = tmp_path / "cur.sqlite3"
    cid = curriculum(CurriculumStore(path))
    stores = [CurriculumStore(path) for _ in range(6)]                                              # six connections, as six processes would have

    outcomes = run_threads(lambda n: stores[n].save_version(valid_curriculum(), VALID, [], curriculum_id=cid)["version"], 6)

    assert [o[0] for o in outcomes] == ["ok"] * 6, outcomes
    assert sorted(o[1] for o in outcomes) == [2, 3, 4, 5, 6, 7]                                      # every version number used exactly once
    check = CurriculumStore(path)
    assert check.get(cid)["current_version"] == 7 and [v["version"] for v in check.versions(cid)] == list(range(1, 8))
    assert len([r for r in check.audit_trail(cid) if r["event"] == "version_saved"]) == 7


def test_conversation_state_survives_concurrent_writers(tmp_path):
    path = tmp_path / "c.sqlite3"
    stores = [ConversationStore(path) for _ in range(5)]
    outcomes = run_threads(lambda n: [stores[n].set("conv", f"agent{n}", {"n": i}) for i in range(20)] and n, 5)
    assert [o[0] for o in outcomes] == ["ok"] * 5
    assert [ConversationStore(path).get("conv", f"agent{n}") for n in range(5)] == [{"n": 19}] * 5


def test_two_concurrent_approvals_of_the_same_marketing_version_apply_once(tmp_path):
    path = tmp_path / "m.sqlite3"
    cid = MarketingStore(path).save_version(CAMPAIGN, VALID)["campaign_id"]
    stores = [MarketingStore(path) for _ in range(4)]
    outcomes = run_threads(lambda n: stores[n].approve(cid, 1, ADMIN, f"reviewer {n}")["version_status"], 4)

    assert sorted(o[0] for o in outcomes) == ["error", "error", "error", "ok"], outcomes
    assert all("pending review" in o[1] for o in outcomes if o[0] == "error")
    assert len([r for r in MarketingStore(path).audit_trail(cid) if r["event"] == "review"]) == 1


# ---- separate processes (really started) -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def mp():
    return multiprocessing.get_context("spawn")


def run_processes(mp, target, args_list):
    barrier, results = mp.Barrier(len(args_list)), mp.Queue()
    processes = [mp.Process(target=target, args=(*args, barrier, results)) for args in args_list]
    [p.start() for p in processes]
    outcomes = [results.get(timeout=120) for _ in processes]
    [p.join(60) for p in processes]
    assert all(p.exitcode == 0 for p in processes)
    return outcomes


def test_two_processes_approving_the_same_curriculum_apply_it_once(tmp_path, mp):
    path = tmp_path / "cur.sqlite3"
    store = CurriculumStore(path)
    cid = curriculum(store)
    store.close()

    outcomes = run_processes(mp, sqlite_workers.approve_curriculum, [(str(path), cid, 1)] * 2)

    assert sorted(o[0] for o in outcomes) == ["error", "ok"], outcomes
    assert "pending review" in next(o[1] for o in outcomes if o[0] == "error")
    check = CurriculumStore(path)
    assert check.get(cid)["approved_version"] == 1 and check.get(cid)["status"] == "approved"
    assert [r["event"] for r in check.audit_trail(cid)] == ["version_saved", "review"]               # one approval, one audit row


def test_processes_saving_versions_get_unique_consecutive_numbers(tmp_path, mp):
    path = tmp_path / "cur.sqlite3"
    store = CurriculumStore(path)
    cid = curriculum(store)
    store.close()

    outcomes = run_processes(mp, sqlite_workers.save_curriculum_version, [(str(path), cid)] * 4)

    assert [o[0] for o in outcomes] == ["ok"] * 4, outcomes
    assert sorted(o[1] for o in outcomes) == [2, 3, 4, 5]
    check = CurriculumStore(path)
    assert [v["version"] for v in check.versions(cid)] == [1, 2, 3, 4, 5] and check.get(cid)["current_version"] == 5


def test_two_processes_approving_the_same_campaign_apply_it_once(tmp_path, mp):
    path = tmp_path / "m.sqlite3"
    store = MarketingStore(path)
    cid = store.save_version(CAMPAIGN, VALID)["campaign_id"]
    store.close()
    outcomes = run_processes(mp, sqlite_workers.approve_campaign, [(str(path), cid, 1)] * 2)
    assert sorted(o[0] for o in outcomes) == ["error", "ok"], outcomes
    assert len([r for r in MarketingStore(path).audit_trail(cid) if r["event"] == "review"]) == 1


def test_processes_writing_conversation_state_do_not_lose_or_break_each_others_writes(tmp_path, mp):
    path = tmp_path / "c.sqlite3"
    ConversationStore(path).close()
    outcomes = run_processes(mp, sqlite_workers.set_state, [(str(path), "conv", n) for n in range(3)])
    assert sorted(o[0] for o in outcomes) == ["ok"] * 3, outcomes
    check = ConversationStore(path)
    assert [check.get("conv", f"agent{n}") for n in range(3)] == [{"n": 19}] * 3
