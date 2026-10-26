
import json
from pathlib import Path
from typing import Any

from knowledge_base.file_loader import discover_files
from knowledge_base.parsers import parse_document
from knowledge_base.cleaner import (
    clean_text,
    remove_repeated_headers_footers,
)
from knowledge_base.chunker import chunk_document
from knowledge_base.metadata_extractor import (
    extract_metadata,
    save_metadata_json,
)
from knowledge_base.embedding_service import embed_texts
from knowledge_base.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "knowledge_base" / "documents"
DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "knowledge_base" / "metadata" / "chunk_metadata.json"
)
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "knowledge_base" / "metadata" / "ingestion_manifest.json"
)


def _normalize_source(source: str | Path) -> str:
    """Return a consistent absolute document path."""
    return str(Path(source).resolve())


def _relative_source(source: str | Path) -> str:
    """Return a project-relative path when possible."""
    resolved = Path(source).resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path, default: Any) -> Any:
    """Load JSON and return a default value if reading fails."""
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_previous_sources(manifest_path: Path) -> set[str]:
    """Load previously tracked document paths."""
    manifest = _load_json(manifest_path, {})

    if not isinstance(manifest, dict):
        return set()

    sources = manifest.get("sources", [])

    if not isinstance(sources, list):
        return set()

    return {
        _normalize_source(source)
        for source in sources
        if isinstance(source, str) and source.strip()
    }


def _load_previous_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load previously saved metadata records."""
    records = _load_json(metadata_path, [])

    if not isinstance(records, list):
        return []

    return [
        record
        for record in records
        if isinstance(record, dict)
    ]


def _get_record_source(record: dict[str, Any]) -> str | None:
    """Get the source path recorded in a metadata entry."""
    source = record.get("data_source") or record.get("document_id")

    if not isinstance(source, str) or not source.strip():
        return None

    return _normalize_source(source)


def _preserve_failed_metadata(
    previous_metadata: list[dict[str, Any]],
    failed_sources: set[str],
) -> list[dict[str, Any]]:
    """Keep previous metadata for documents that failed processing."""
    return [
        record
        for record in previous_metadata
        if _get_record_source(record) in failed_sources
    ]


def _process_document(
    file_path: str | Path,
    max_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the existing document-processing components in sequence."""
    source = _normalize_source(file_path)

    document = parse_document(file_path)
    document["source"] = source

    cleaned_pages = [
        {
            **page,
            "text": clean_text(page["text"]),
        }
        for page in document["pages"]
    ]

    document["pages"] = remove_repeated_headers_footers(cleaned_pages)

    chunks = chunk_document(document, max_chars=max_chars)

    metadata_records = []

    for chunk in chunks:
        chunk["source"] = source

        metadata = extract_metadata(chunk)
        chunk["extracted_metadata"] = metadata
        metadata_records.append(metadata)

    return chunks, metadata_records


def _delete_source_vectors(
    vector_store: VectorStore,
    source: str,
) -> int:
    """Delete vectors matching absolute and legacy relative source paths."""
    deleted = vector_store.delete_source(source)

    relative_source = _relative_source(source)

    if relative_source != source:
        deleted += vector_store.delete_source(relative_source)

    return deleted


def ingest_directory(
    directory: str | Path = DEFAULT_DOCUMENTS_DIR,
    vector_store: VectorStore | None = None,
    max_chars: int = 2500,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Coordinate document ingestion using the existing knowledge-base modules.

    Components handle their own processing logic. This function coordinates
    processing, embedding, vector replacement, and ingestion bookkeeping.
    """
    directory = Path(directory).resolve()
    metadata_path = Path(metadata_path).resolve()
    manifest_path = Path(manifest_path).resolve()

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files = discover_files(directory)

    if vector_store is None:
        vector_store = VectorStore()

    current_sources = {
        _normalize_source(file_path)
        for file_path in files
    }

    previous_sources = _load_previous_sources(manifest_path)
    previous_metadata = _load_previous_metadata(metadata_path)

    summary: dict[str, Any] = {
        "files_discovered": len(files),
        "files_processed": 0,
        "files_failed": [],
        "chunks_created": 0,
        "chunks_stored": 0,
        "chunks_deleted": 0,
        "metadata_json": str(metadata_path),
    }

    all_chunks: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []

    failed_sources: set[str] = set()
    successful_sources: set[str] = set()

    # Process documents through the dedicated components.
    for file_path in files:
        source = _normalize_source(file_path)

        try:
            chunks, metadata_records = _process_document(
                file_path,
                max_chars,
            )

            all_chunks.extend(chunks)
            all_metadata.extend(metadata_records)

            successful_sources.add(source)

            summary["files_processed"] += 1
            summary["chunks_created"] += len(chunks)

        except Exception as exc:
            failed_sources.add(source)

            summary["files_failed"].append({
                "source": source,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Generate embeddings before replacing existing vectors.
    embeddings = []

    if all_chunks:
        embeddings = embed_texts([
            chunk["text"]
            for chunk in all_chunks
        ])

    # Replace vectors only for documents processed successfully.
    for source in successful_sources:
        summary["chunks_deleted"] += _delete_source_vectors(
            vector_store,
            source,
        )

    if all_chunks:
        summary["chunks_stored"] = vector_store.add_chunks(
            all_chunks,
            embeddings,
        )

    # Remove vectors for documents deleted from the input directory.
    removed_sources = (
        previous_sources - current_sources - failed_sources
    )

    for source in removed_sources:
        summary["chunks_deleted"] += _delete_source_vectors(
            vector_store,
            source,
        )

    # Preserve metadata belonging to documents that failed this run.
    preserved_metadata = _preserve_failed_metadata(
        previous_metadata,
        failed_sources,
    )

    combined_metadata = preserved_metadata + all_metadata

    save_metadata_json(combined_metadata, metadata_path)

    # Keep failed sources in the manifest so they can be retried later.
    manifest_sources = (
        previous_sources - removed_sources
    ) | successful_sources | failed_sources

    manifest_path.write_text(
        json.dumps(
            {"sources": sorted(manifest_sources)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary
