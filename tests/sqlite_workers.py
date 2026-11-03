"""
Worker functions for the multi-process SQLite tests (tests/test_sqlite_concurrency.py). They live in an importable module
because a spawned process (the only kind Windows has) must be able to import its target.
"""

from __future__ import annotations

ADMIN = "admin-session-worker01"


def approve_curriculum(db_path: str, curriculum_id: str, version: int, barrier, results) -> None:
    from agents.curriculum_research.curriculum_store import CurriculumStore

    store = CurriculumStore(db_path)
    barrier.wait(timeout=60)
    try:
        store.approve(curriculum_id, version, ADMIN, "from a process")
        results.put(("ok", ""))
    except Exception as exc:  # the loser of the race reports why
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        store.close()


def save_curriculum_version(db_path: str, curriculum_id: str, barrier, results) -> None:
    from agents.curriculum_research.curriculum_store import CurriculumStore
    from tests.curriculum_fixtures import valid_curriculum

    store = CurriculumStore(db_path)
    barrier.wait(timeout=60)
    try:
        saved = store.save_version(valid_curriculum(), {"valid": True, "errors": []}, [], curriculum_id=curriculum_id)
        results.put(("ok", saved["version"]))
    except Exception as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        store.close()


def approve_campaign(db_path: str, campaign_id: str, version: int, barrier, results) -> None:
    from agents.marketing.marketing_store import MarketingStore

    store = MarketingStore(db_path)
    barrier.wait(timeout=60)
    try:
        store.approve(campaign_id, version, ADMIN)
        results.put(("ok", ""))
    except Exception as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        store.close()


def set_state(db_path: str, conversation_id: str, number: int, barrier, results) -> None:
    from agents.conversation_store import ConversationStore

    store = ConversationStore(db_path)
    barrier.wait(timeout=60)
    try:
        for i in range(20):
            store.set(conversation_id, f"agent{number}", {"n": i})
        results.put(("ok", number))
    except Exception as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        store.close()
