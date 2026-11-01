"""
VectorStore.reload_if_changed (knowledge_base/vector_store.py): chunks that
ANOTHER process ingests while the store is open become searchable without
a restart. Generalization check T5. Every test uses a Chroma folder in a
pytest temporary directory -- never the knowledge-base index.
"""

import subprocess
import sys

import pytest

from knowledge_base.vector_store import VectorStore

VECTOR = [1.0, 0.0, 0.0]


def _chunk(chunk_id, text, source="docs/a.md", **fields):
    return {"text": text, "chunk_id": chunk_id, "source": source, **fields}


def _write_from_another_process(path, chunk_id, text):
    """What an ingestion run does: open the same index in its own process and upsert."""
    code = (
        "import sys\n"
        "from knowledge_base.vector_store import VectorStore\n"
        f"VectorStore(persist_directory={str(path)!r}).add_chunks(\n"
        f"    [{{'text': {text!r}, 'chunk_id': {chunk_id}, 'source': 'docs/new.md', 'field_course_id': 'TST-901'}}],\n"
        f"    [{VECTOR!r}])\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(_project_root()))


def _project_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    store = VectorStore(persist_directory=tmp_path / "index")
    store.add_chunks([_chunk(1, "An existing course passage.")], [[0.0, 1.0, 0.0]])
    return store


def test_a_chunk_ingested_by_another_process_is_searchable_without_a_restart(store, tmp_path):
    assert [m["text"] for m in store.search(VECTOR, top_k=5)] == ["An existing course passage."]

    _write_from_another_process(tmp_path / "index", 7, "A newly ingested course passage.")

    texts = [m["text"] for m in store.search(VECTOR, top_k=5)]
    assert "A newly ingested course passage." in texts


def test_own_writes_do_not_reopen_the_index(store):
    store.add_chunks([_chunk(2, "Another passage.")], [[0.5, 0.5, 0.0]])
    assert store.reload_if_changed() is False

    store.delete_source("docs/a.md")
    assert store.reload_if_changed() is False


def test_unchanged_index_is_not_reopened(store):
    store.search(VECTOR, top_k=1)
    store.collection.get(limit=1)
    assert store.reload_if_changed() is False


def test_reopening_never_creates_a_collection(store, tmp_path):
    import chromadb

    other = chromadb.PersistentClient(path=str(tmp_path / "index"))
    other.delete_collection(store.collection_name)          # e.g. removed by an admin tool
    store._index_stamp = (0, 0)                              # force a reopen

    with pytest.raises(Exception):
        store.reload_if_changed()
    assert store.collection_name not in [c.name for c in other.list_collections()]
