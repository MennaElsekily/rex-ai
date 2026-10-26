"""
Tests for agents.conversation_store.ConversationStore.

Each test uses its own temporary SQLite file (pytest's tmp_path) so
tests never share or pollute each other's data, and never touch the
default conversation_state.sqlite3 used by the real application.
"""

import pytest

from agents.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(db_path=tmp_path / "test_conversations.sqlite3")


def test_get_returns_none_for_unknown_conversation(store):
    assert store.get("unknown", "rex_assistant") is None


def test_set_then_get_roundtrip(store):
    store.set("conv-1", "rex_assistant", {"background": "cs"})

    assert store.get("conv-1", "rex_assistant") == {"background": "cs"}


def test_update_overwrites_previous_state(store):
    store.set("conv-1", "rex_assistant", {"a": 1})
    store.set("conv-1", "rex_assistant", {"a": 2})

    assert store.get("conv-1", "rex_assistant") == {"a": 2}


def test_different_conversations_are_isolated(store):
    store.set("conv-1", "rex_assistant", {"a": 1})
    store.set("conv-2", "rex_assistant", {"a": 2})

    assert store.get("conv-1", "rex_assistant") == {"a": 1}
    assert store.get("conv-2", "rex_assistant") == {"a": 2}


def test_different_agents_in_same_conversation_are_isolated(store):
    store.set("conv-1", "rex_assistant", {"a": 1})
    store.set("conv-1", "curriculum_research", {"b": 2})

    assert store.get("conv-1", "rex_assistant") == {"a": 1}
    assert store.get("conv-1", "curriculum_research") == {"b": 2}


def test_persists_across_store_instances_on_same_db_file(tmp_path):
    db_path = tmp_path / "shared.sqlite3"

    store1 = ConversationStore(db_path=db_path)
    store1.set("conv-1", "rex_assistant", {"background": "cs"})
    store1.close()

    store2 = ConversationStore(db_path=db_path)

    assert store2.get("conv-1", "rex_assistant") == {"background": "cs"}


def test_rejects_empty_conversation_id(store):
    with pytest.raises(ValueError):
        store.get("", "rex_assistant")

    with pytest.raises(ValueError):
        store.set("", "rex_assistant", {})


def test_rejects_non_dict_state(store):
    with pytest.raises(TypeError):
        store.set("conv-1", "rex_assistant", "not a dict")


def test_supports_arabic_text_in_state(store):
    state = {"background": "علوم الحاسوب", "career_goal": "مهندس تعلم آلي"}
    store.set("conv-1", "rex_assistant", state)

    assert store.get("conv-1", "rex_assistant") == state
