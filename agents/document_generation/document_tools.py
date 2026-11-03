"""
Agent 4's tools, as one validated toolbox.

agents/document_mcp_server.py exposes the tool methods below over MCP and
the agent reaches them through agents/document_generation/mcp_client.py.
Tests use the same toolbox in-process. Every tool method checks its input,
never raises for ordinary failures, and returns a JSON-ready dict with a
"status" ("ok", "invalid_request", "not_found", "not_approved" or "error")
and a "message" on failure.

Boundaries:
- curricula are read through the Curriculum Agent's own CurriculumStore;
  only a version whose review status is "approved" is ever used, and
  nothing is written to that store (no second approval logic);
- the save tool validates the document itself against the approved
  curriculum (validation.validate_content) and can only store "draft" or
  "pending_review" -- approval exists only in AdminService;
- no tool takes a file path: documents are addressed by validated ids and
  PDF names are built from them.

Two methods are deliberately NOT tools, because they need an administrator
identity or return file contents: save_admin_edit and document_bytes
(called by AdminService after its session check).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from agents.config import get_setting
from agents.curriculum_research.curriculum_store import (
    APPROVED as CURRICULUM_APPROVED,
    CurriculumStore,
    CurriculumStoreError,
)
from agents.document_generation import templates
from agents.document_generation.document_store import (
    AGENT_ACTOR,
    DocumentStore,
    DocumentStoreError,
    STATUSES,
    check_document_id,
)
from agents.document_generation.validation import issue, validate_content

logger = logging.getLogger(__name__)

DB_SETTING = "DOCUMENT_DB_PATH"
DIR_SETTING = "DOCUMENT_OUTPUT_DIR"
DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
MAX_DOCUMENT_BYTES = 400_000
_CURRICULUM_PDF_NAME = re.compile(r"CUR-[0-9a-f]{12}_v\d+_(ar|en)\.pdf")
_PDF_NAME = re.compile(r"GDOC-[0-9a-f]{12}_v\d+_(ar|en)(_approved)?\.pdf")


def _invalid(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "invalid_request", "message": message, **extra}


class DocumentToolbox:
    def __init__(self, store: Optional[DocumentStore] = None, curricula: Optional[CurriculumStore] = None,
                 documents_dir: Optional[Path] = None) -> None:
        self.store = store or DocumentStore(get_setting(DB_SETTING, "") or None)
        self.curricula = curricula or CurriculumStore(get_setting("CURRICULUM_DB_PATH", "") or None)
        self.documents_dir = Path(documents_dir or get_setting(DIR_SETTING, "") or DEFAULT_DOCUMENTS_DIR)

    def _safely(self, name: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return call()
        except (DocumentStoreError, CurriculumStoreError, ValueError) as exc:
            return _invalid(str(exc))
        except Exception as exc:  # never let an internal error escape as a crash
            logger.exception("Document tool %s failed", name)
            return {"status": "error", "message": f"The {name} tool failed ({type(exc).__name__})."}

    # ----------------------------------------------------------- curricula

    def _approved_record(self, curriculum_id: Any, version: Optional[int] = None) -> dict[str, Any]:
        """
        The approved version of a curriculum (or the given version, if it is
        approved). Raises ValueError with the reason otherwise. The review
        status comes from the Curriculum Agent's store, which only
        AdminService can change.
        """
        head = self.curricula.get(curriculum_id)
        if head is None:
            raise LookupError(f"No curriculum {curriculum_id}.")
        approved_version = head.get("approved_version")
        if approved_version is None:
            raise PermissionError(
                f"Curriculum {curriculum_id} is {head['status']}: it needs to be approved by an administrator "
                "(Curricula tab) before documents can be generated from it.")
        record = self.curricula.get(curriculum_id, version or approved_version)
        if record is None or record.get("version_status") != CURRICULUM_APPROVED:
            raise PermissionError(f"Version {version} of curriculum {curriculum_id} is not an approved version.")
        record["current_version"] = head["current_version"]
        return record

    def ensure_curriculum_approved(self, curriculum_id: str, version: int) -> None:
        """For AdminService before it approves a document: the curriculum version must still be an approved one."""
        try:
            self._approved_record(curriculum_id, version)
        except (LookupError, PermissionError) as exc:
            raise ValueError(str(exc)) from exc

    def list_document_types(self) -> dict:
        return {"status": "ok",
                "supported": [{"document_type": t, **tpl.label} for t, tpl in templates.TEMPLATES.items()],
                "unsupported": [{"document_type": t, "en": v["en"], "ar": v["ar"], "reason": v["reason"]}
                                for t, v in templates.UNSUPPORTED.items()],
                "extra_fields": dict(templates.EXTRA_FIELDS)}

    def list_approved_curricula(self, limit=50, offset=0) -> dict:
        """One page of the curricula that have an approved version, newest first, with the total."""
        def run():
            limit_, offset_ = max(1, min(int(limit), 200)), max(0, int(offset))
            rows = self.curricula.list_approved(limit_, offset_)
            total = self.curricula.count_approved()
            keep = ("curriculum_id", "title", "language", "level", "total_hours", "approved_version",
                    "current_version", "status", "target_audience")
            return {"status": "ok", "curricula": [{k: r.get(k) for k in keep} for r in rows], "total": total,
                    "offset": offset_, "has_more": offset_ + len(rows) < total}
        return self._safely("list_approved_curricula", run)

    def get_approved_curriculum(self, curriculum_id, version=None) -> dict:
        def run():
            try:
                record = self._approved_record(curriculum_id, int(version) if version is not None else None)
            except LookupError as exc:
                return {"status": "not_found", "message": str(exc)}
            except PermissionError as exc:
                return {"status": "not_approved", "message": str(exc)}
            return {"status": "ok", "curriculum": record,
                    "newer_unapproved_version": record["current_version"] > record["version"]}
        return self._safely("get_approved_curriculum", run)

    # ----------------------------------------------------------- documents

    def save_document_draft(self, content, document_id=None, change_summary=None) -> dict:
        """Save a new document or a new version; the tool validates it itself."""
        return self._save(content, document_id, change_summary, AGENT_ACTOR)

    def save_admin_edit(self, content, document_id, admin: str, change_summary=None) -> dict:
        """An administrator's edited version (not an MCP tool: it records who edited)."""
        if not isinstance(admin, str) or not admin.strip() or admin.startswith("system:"):
            return _invalid("An administrator identity is required.")
        return self._save(content, document_id, change_summary or "Edited by an administrator", admin)

    def _save(self, content, document_id, change_summary, actor) -> dict:
        def run():
            import json
            if not isinstance(content, dict) or len(json.dumps(content, ensure_ascii=False)) > MAX_DOCUMENT_BYTES:
                return _invalid("content must be an object of reasonable size.")
            if content.get("document_type") not in templates.TEMPLATES:
                return _invalid("Unknown or unsupported document type.")
            try:
                record = self._approved_record(content.get("curriculum_id"), content.get("curriculum_version"))
            except LookupError as exc:
                return {"status": "not_found", "message": str(exc)}
            except PermissionError as exc:
                return {"status": "not_approved", "message": str(exc)}
            validation = validate_content(content, record)
            saved = self.store.save_version(content, validation, document_id=document_id,
                                            change_summary=_clip(change_summary), created_by=actor)
            return {"status": "ok", "document_id": saved["document_id"], "version": saved["version"],
                    "review_status": saved["status"], "validation": validation}
        return self._safely("save_document_draft", run)

    def get_document(self, document_id, version=None) -> dict:
        def run():
            record = self.store.get(document_id, int(version) if version is not None else None)
            if record is None:
                return {"status": "not_found", "message": f"No document {document_id} (version {version})."}
            return {"status": "ok", "document": record, "versions": self.store.versions(document_id)}
        return self._safely("get_document", run)

    def list_documents(self, status=None, curriculum_id=None, limit=50) -> dict:
        def run():
            if status is not None and status not in STATUSES:
                return _invalid(f"status must be one of {', '.join(STATUSES)}.")
            return {"status": "ok", "documents": self.store.list(status, curriculum_id or None,
                                                                 max(1, min(int(limit), 200)))}
        return self._safely("list_documents", run)

    def render_document_pdf(self, document_id, version=None) -> dict:
        """Render one version as a PDF once; a version that has a PDF keeps it."""
        def run():
            from agents.document_generation.pdf import render_document_pdf
            record = self.store.get(document_id, int(version) if version is not None else None)
            if record is None:
                return {"status": "not_found", "message": f"No document {document_id}."}
            if record["filename"]:
                return {"status": "ok", "document": record}
            data, pages = render_document_pdf(record)
            filename = f"{document_id}_v{record['version']}_{record['language']}.pdf"
            if not _PDF_NAME.fullmatch(filename):
                return _invalid("Unsafe file name.")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            path = self.documents_dir / filename
            # Checked again just before writing: another process may have recorded this version's PDF meanwhile.
            current = self.store.get(document_id, record["version"])
            if current["filename"]:
                return {"status": "ok", "document": current}
            # The store has no PDF for this version, so a file with this name is a leftover (for example from a
            # crash between writing the file and recording it): replace it instead of failing for ever.
            path.unlink(missing_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
            try:
                updated = self.store.attach_pdf(document_id, record["version"], filename, data, pages)
            except Exception:
                self._discard_unrecorded(path, document_id, record["version"], "filename")
                raise
            return {"status": "ok", "document": updated}
        return self._safely("render_document_pdf", run)

    def approve_document(self, document_id: str, version: int, admin: str, notes: Optional[str] = None) -> dict:
        """
        Approve a version (not an MCP tool: AdminService calls it after its
        session check, with the administrator's identity).

        The approved PDF is made and written first; the store then records
        the approval and that file in one transaction. If anything fails
        before that, nothing is approved and the file is removed. The
        curriculum version it was built from must still be an approved one.
        """
        from agents.document_generation.pdf import render_document_pdf

        record = self.store.check_approvable(document_id, int(version), admin)
        self.ensure_curriculum_approved(record["curriculum_id"], record["curriculum_version"])
        approved_at = datetime.now(timezone.utc).isoformat()
        data, pages = render_document_pdf(record, approval={"approved_at": approved_at})
        filename = f"{document_id}_v{record['version']}_{record['language']}_approved.pdf"
        if not _PDF_NAME.fullmatch(filename):
            raise ValueError("Unsafe file name.")
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        path = self.documents_dir / filename
        # Checked again just before writing: if the version was approved meanwhile, its file must not be touched.
        self.store.check_approvable(document_id, int(version), admin)
        # A file left by an earlier failed attempt belongs to no approval (the store has none recorded): replace it.
        path.unlink(missing_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        try:
            return self.store.approve(document_id, int(version), admin, notes,
                                      approved_pdf=(filename, data, pages), approved_at=approved_at)
        except Exception:
            self._discard_unrecorded(path, document_id, int(version), "approved_filename")
            raise

    def _discard_unrecorded(self, path: Path, document_id: str, version: int, column: str) -> None:
        """
        Remove a file this call wrote after recording it failed, but only while the store still has no record of
        a file for it: if another process recorded the file meanwhile, it is that record's file now.
        """
        current = self.store.get(document_id, version)
        if current is None or not current[column]:
            path.unlink(missing_ok=True)

    def approved_curriculum_pdf(self, curriculum_id: str) -> dict:
        """
        The Curriculum Agent's own PDF of an approved curriculum, for download (not an MCP tool; AdminService calls it
        after its session check). It is not a generated document: nothing is stored or recorded, the curriculum store
        is only read, and Agent 3's renderer (agents/curriculum_research/pdf.py) is the only code that draws it.

        Only the curriculum's approved version is used; a curriculum that is draft, pending review or has changes
        requested is refused. The PDF is in the curriculum's own language, and its cover says the version is approved.
        Raises ValueError (with the reason) when it cannot be produced.
        """
        from agents.curriculum_research.pdf import render_pdf

        try:
            record = self._approved_record(curriculum_id)
        except (LookupError, PermissionError) as exc:
            raise ValueError(str(exc)) from exc
        language = "ar" if record["content"].get("language") == "ar" else "en"
        filename = f"{record['curriculum_id']}_v{record['version']}_{language}.pdf"
        if not _CURRICULUM_PDF_NAME.fullmatch(filename):
            raise ValueError("Unsafe file name.")
        data, pages = render_pdf(record)
        return {"filename": filename, "data": data, "pages": pages, "version": record["version"], "language": language}

    def document_bytes(self, document_id: str, version: Optional[int] = None) -> Optional[bytes]:
        """
        For AdminService (UI download) only; not an MCP tool. An approved
        version gives its approved PDF; any other version its review PDF.
        A version approved before approved PDFs existed has none, and falls
        back to its review PDF (read-only: nothing is created or changed).
        """
        check_document_id(document_id)
        record = self.store.get(document_id, version)
        if record is None:
            return None
        name = record["approved_filename"] or record["filename"] if record["version_status"] == "approved" \
            else record["filename"]
        if not name or not _PDF_NAME.fullmatch(name):
            return None
        path = self.documents_dir / name
        return path.read_bytes() if path.is_file() else None

    def close(self) -> None:
        pass


def _clip(text: Any, limit: int = 300) -> Optional[str]:
    cleaned = " ".join(str(text or "").split())[:limit]
    return cleaned or None


def toolbox_from_settings() -> DocumentToolbox:
    """The toolbox the MCP server uses: settings from the environment / .env."""
    return DocumentToolbox()


__all__ = ["DocumentToolbox", "issue", "toolbox_from_settings"]
