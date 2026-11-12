"""
Project-level safeguards (acceptance, 2026-11-11):

- the knowledge base is built only from the authoritative documents in
  knowledge_base/documents/ and can be recreated from them;
- the active index holds no data of the obsolete NileBridge demonstration
  catalogue and no document that is not in the documents folder;
- generated ChromaDB/ingestion files are not tracked by Git;
- tests cannot modify the live index, user records or curriculum records,
  and cannot send a real email.

Embeddings are faked where a test builds an index (no model needed).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from knowledge_base import chunker, pipeline
from knowledge_base.vector_store import DEFAULT_STORE_PATH, VectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "knowledge_base" / "documents"
LIVE_INDEX = Path(DEFAULT_STORE_PATH) / "chroma.sqlite3"
GENERATED = ("knowledge_base/vector_db", "knowledge_base/metadata/chunk_metadata.json",
             "knowledge_base/metadata/ingestion_manifest.json")
# Identifiers of the obsolete demonstration catalogue (removed 2026-11-11).
OBSOLETE = re.compile(rb"(?i)nilebridge|Expanded_Catalogue|DA-XL-101|DA-SQL-201|MK-ANA-301|MK-STR-101|AI-FND-101|"
                      rb"AI-ML-201|AI-RAG-301|FS-WEB-101|FS-API-301|FL-DART-101|FL-STATE-201")


def _fake_embed(texts, *args, **kwargs):
    return [[byte / 255 for byte in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


def _ingest(tmp_path, name):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "embed_texts", _fake_embed)
        mp.setattr(chunker, "embed_texts", _fake_embed)
        store = VectorStore(persist_directory=tmp_path / name / "db")
        summary = pipeline.ingest_directory(DOCUMENTS, vector_store=store, metadata_path=tmp_path / name / "meta.json",
                                            manifest_path=tmp_path / name / "manifest.json")
    return store, summary


# --------------------------------------------------------------------
# 1, 3: only current source documents; the index can be recreated
# --------------------------------------------------------------------

def test_only_the_authoritative_documents_are_ingested(tmp_path):
    store, summary = _ingest(tmp_path, "a")
    on_disk = sorted(p.name for p in DOCUMENTS.iterdir() if p.is_file())
    metadatas = store.collection.get(include=["metadatas"])["metadatas"]

    assert summary["files_failed"] == [] and summary["files_processed"] == len(on_disk) == 10
    assert sorted({m["filename"] for m in metadatas}) == on_disk
    assert all(str(m["source"]).replace("\\", "/").startswith("knowledge_base/documents/") for m in metadatas)
    assert not OBSOLETE.search("".join(store.collection.get(include=["documents"])["documents"]).encode())


def test_the_index_can_be_recreated_from_the_source_documents(tmp_path):
    first, summary_a = _ingest(tmp_path, "a")
    second, summary_b = _ingest(tmp_path, "b")

    assert summary_a["chunks_stored"] == summary_b["chunks_stored"] > 0
    assert sorted(first.collection.get()["ids"]) == sorted(second.collection.get()["ids"])
    assert sorted(first.collection.get(include=["documents"])["documents"]) == \
        sorted(second.collection.get(include=["documents"])["documents"])


# --------------------------------------------------------------------
# 2: the active index (built locally; read-only check)
# --------------------------------------------------------------------

@pytest.mark.skipif(not LIVE_INDEX.exists(), reason="No local index built yet (python -m knowledge_base.pipeline).")
def test_the_active_index_holds_only_current_documents_and_no_obsolete_catalogue():
    # Opened read-only with sqlite (not Chroma): this test cannot change the index.
    connection = sqlite3.connect(f"file:{LIVE_INDEX}?mode=ro", uri=True)
    files = {v for (v,) in connection.execute("select string_value from embedding_metadata where key='filename'")}
    text = b"".join((v or "").encode() for (v,) in connection.execute(
        "select string_value from embedding_metadata where key='chroma:document'"))
    connection.close()

    assert files and files <= {p.name for p in DOCUMENTS.iterdir() if p.is_file()}
    assert not OBSOLETE.search(text)


# --------------------------------------------------------------------
# 12: generated files are not tracked by Git
# --------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="Not a git checkout.")
def test_generated_knowledge_base_files_are_not_tracked_and_are_ignored():
    tracked = subprocess.run(["git", "ls-files", "--", *GENERATED], cwd=ROOT, capture_output=True, text=True, check=True)
    ignored = [subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT).returncode
               for path in ("knowledge_base/vector_db/chroma.sqlite3", "knowledge_base/metadata/chunk_metadata.json",
                            "knowledge_base/metadata/ingestion_manifest.json")]
    documents = subprocess.run(["git", "ls-files", "--", "knowledge_base/documents"], cwd=ROOT,
                               capture_output=True, text=True, check=True)

    assert tracked.stdout.strip() == ""                      # nothing generated is tracked
    assert ignored == [0, 0, 0]                              # and it stays out of commits
    assert len(documents.stdout.split()) == 10               # the source documents stay tracked


# --------------------------------------------------------------------
# 10, 11, 13: tests cannot touch live data or send email
# --------------------------------------------------------------------

def test_stores_opened_without_a_path_are_temporary_during_tests(tmp_path):
    from agents import conversation_store, pending_workflows
    from agents.curriculum_research import curriculum_store, curriculum_tools
    from agents.document_generation import document_store
    from agents.marketing import marketing_store
    from agents.rex_assistant.lead_store import JSONLLeadStore
    from agents.rfp_research import opportunity_store

    real = ROOT.resolve()
    paths = {
        "index": Path(VectorStore().persist_directory),
        "conversations": Path(conversation_store.ConversationStore().db_path),
        "paused workflows": Path(pending_workflows.DEFAULT_DB_PATH),
        "curricula": Path(curriculum_store.DEFAULT_DB_PATH),
        "curriculum PDFs": Path(curriculum_tools.DEFAULT_DOCUMENTS_DIR),
        "documents": Path(document_store.DEFAULT_DB_PATH),
        "marketing": Path(marketing_store.DEFAULT_DB_PATH),
        "tenders": Path(opportunity_store.DEFAULT_DB_PATH),
        "leads": JSONLLeadStore().path,
    }
    for name, path in paths.items():
        resolved = path.resolve()
        assert real not in resolved.parents or tmp_path.parent in resolved.parents, f"{name} points at {path}"


def test_no_test_can_send_a_real_email():
    from agents.rfp_research import email_provider

    with pytest.raises(email_provider.EmailProviderError, match="RESEND_API_KEY is not configured"):
        email_provider.send_via_resend("someone@example.com", "academy@example.com", "Subject", "Body")
