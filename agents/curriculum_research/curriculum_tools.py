"""
The Curriculum Agent's tools, as one validated toolbox.

agents/curriculum_mcp_server.py exposes these methods as MCP tools and
the agent reaches them through agents/curriculum_research/mcp_client.py.
Tests use the same toolbox in-process. Every method checks its input,
never raises for ordinary failures, and returns a JSON-ready dict with a
"status" ("ok", "invalid_request", "not_found", "not_configured",
"awaiting_credentials" or "error") and a "message" on failure.

Boundaries:
- no tool takes a file path: curricula, versions and documents are
  addressed by validated ids;
- the save tool validates the curriculum itself (it never trusts a
  validation result passed in) and can only store "draft" or
  "pending_review" -- approval exists only in AdminService;
- research is limited to the approved domains (research.py);
- the academy catalogue is read from the existing index, read-only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from agents.config import get_setting
from agents.curriculum_research import frameworks, research
from agents.curriculum_research.academy_courses import AcademyCatalogue
from agents.curriculum_research.curriculum_store import (
    CurriculumStore,
    CurriculumStoreError,
    STATUSES,
)
from agents.curriculum_research.validation import validate_curriculum

logger = logging.getLogger(__name__)

DB_SETTING = "CURRICULUM_DB_PATH"
DOCUMENTS_SETTING = "CURRICULUM_DOCUMENTS_DIR"
INDEX_SETTING = "CURRICULUM_INDEX_DIR"
DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
MAX_CURRICULUM_BYTES = 200_000
MAX_TEXT = 300
_PUBLIC = lambda record: {k: v for k, v in record.items() if not k.startswith("_")}  # noqa: E731


def _invalid(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "invalid_request", "message": message, **extra}


def _text(value: Any, limit: int = MAX_TEXT) -> Optional[str]:
    text = " ".join(str(value or "").split())[:limit]
    return text or None


def _texts(values: Any, limit: int = 40) -> list[str]:
    return [t for t in (_text(v) for v in (values or [])[:limit]) if t] if isinstance(values, list) else []


def _check_curriculum(curriculum: Any) -> Optional[str]:
    if not isinstance(curriculum, dict):
        return "curriculum must be an object."
    if len(json.dumps(curriculum, ensure_ascii=False)) > MAX_CURRICULUM_BYTES:
        return "The curriculum is too large."
    hours = curriculum.get("total_hours")
    if not isinstance(hours, int) or isinstance(hours, bool) or not 1 <= hours <= 1000:
        return "total_hours must be a whole number between 1 and 1000."
    return None


class CurriculumToolbox:
    def __init__(
        self,
        store: Optional[CurriculumStore] = None,
        catalogue: Optional[AcademyCatalogue] = None,
        frameworks_dir: Optional[Path] = None,
        documents_dir: Optional[Path] = None,
        research_domains: Optional[list[str]] = None,
        search: Optional[Callable[..., Any]] = None,
        translate: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.store = store or CurriculumStore(get_setting(DB_SETTING, "") or None)
        self.catalogue = catalogue or AcademyCatalogue(translate=translate)
        self.frameworks_dir = Path(frameworks_dir) if frameworks_dir else frameworks.frameworks_dir()
        self.documents_dir = Path(documents_dir or get_setting(DOCUMENTS_SETTING, "") or DEFAULT_DOCUMENTS_DIR)
        self.research_domains = research_domains
        self._search = search
        self._translate = translate

    def _safely(self, name: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return call()
        except (CurriculumStoreError, ValueError) as exc:
            return _invalid(str(exc))
        except Exception as exc:  # never let an internal error escape as a crash
            logger.exception("Curriculum tool %s failed", name)
            return {"status": "error", "message": f"The {name} tool failed ({type(exc).__name__})."}

    # ----------------------------------------------------- academy courses

    def list_academy_courses(self, department=None, level=None, audience_contains=None, limit=100) -> dict:
        def run():
            courses = self.catalogue.courses(_text(department, 80), _text(level, 20),
                                             _text(audience_contains, 80), max(1, min(int(limit), 200)))
            return {"status": "ok", "courses": [_PUBLIC(c) for c in courses],
                    "data_status": sorted({c.get("data_status") for c in courses if c.get("data_status")})}
        return self._safely("list_academy_courses", run)

    def get_course_details(self, course) -> dict:
        def run():
            reference = _text(course, 120)
            if not reference:
                return _invalid("A course id or name is required.")
            record, how = self.catalogue.course(reference)
            if record is None:
                return {"status": "not_found", "message": f"No course matching {reference!r} is in the catalogue."}
            public = _PUBLIC(record)
            public["sections"] = record["_lists"]
            public["facts"] = record["_facts"]
            return {"status": "ok", "course": public, "matched_by": how}
        return self._safely("get_course_details", run)

    def check_course_overlap(self, title, topics=None, outcomes=None) -> dict:
        def run():
            name = _text(title, 200)
            if not name:
                return _invalid("A title is required.")
            return {"status": "ok", **self.catalogue.overlap(name, _texts(topics), _texts(outcomes))}
        return self._safely("check_course_overlap", run)

    def catalogue_gap_report(self, department=None) -> dict:
        return self._safely("catalogue_gap_report",
                            lambda: {"status": "ok", **self.catalogue.gap_report(_text(department, 80))})

    # ------------------------------------------------------------ research

    def search_curriculum_sources(self, query, max_results=5) -> dict:
        return self._safely("search_curriculum_sources", lambda: research.search_sources(
            _text(query) or "", int(max_results), self.research_domains, self._search))

    # ---------------------------------------------------------- frameworks

    def list_frameworks(self) -> dict:
        return self._safely("list_frameworks", lambda: {
            "status": "ok", "frameworks": frameworks.list_frameworks(self.frameworks_dir)})

    def get_framework_requirements(self, framework_id) -> dict:
        return self._safely("get_framework_requirements",
                            lambda: frameworks.load_framework(framework_id, self.frameworks_dir))

    def compare_with_framework(self, curriculum, framework_id) -> dict:
        def run():
            problem = _check_curriculum(curriculum)
            if problem:
                return _invalid(problem)
            loaded = frameworks.load_framework(framework_id, self.frameworks_dir)
            if loaded["status"] != "ok":
                return loaded
            translate = self.catalogue.to_english if curriculum.get("language") == "ar" else None
            framework = loaded["framework"]
            return {"status": "ok", "framework": {k: framework.get(k) for k in (
                        "framework_id", "name", "publisher", "version", "source_url", "retrieved_on")},
                    "pre_mapping": frameworks.pre_mapping(curriculum, framework, translate)}
        return self._safely("compare_with_framework", run)

    # ---------------------------------------------------------- validation

    def _framework_for(self, framework_id) -> Optional[dict]:
        if not framework_id:
            return None
        loaded = frameworks.load_framework(framework_id, self.frameworks_dir)
        return loaded.get("framework") if loaded["status"] == "ok" else None

    def validate_curriculum(self, curriculum, requested_hours, framework_id=None) -> dict:
        def run():
            problem = _check_curriculum(curriculum)
            if problem:
                return _invalid(problem)
            if not isinstance(requested_hours, int) or isinstance(requested_hours, bool) or requested_hours < 1:
                return _invalid("requested_hours must be a whole number above zero.")
            return {"status": "ok", **validate_curriculum(curriculum, requested_hours,
                                                          self._framework_for(framework_id))}
        return self._safely("validate_curriculum", run)

    # ------------------------------------------------------------- storage

    def save_curriculum_draft(self, curriculum, sources=None, curriculum_id=None, change_summary=None,
                              alignment=None, framework_id=None, topic=None, origin="request",
                              source_request=None) -> dict:
        def run():
            problem = _check_curriculum(curriculum)
            if problem:
                return _invalid(problem)
            if origin not in ("request", "suggestion"):
                return _invalid("origin must be request or suggestion.")
            # Validated here, never taken from the caller.
            validation = validate_curriculum(curriculum, curriculum["total_hours"],
                                             self._framework_for(framework_id))
            saved = self.store.save_version(
                curriculum, validation, [s for s in (sources or [])[:50] if isinstance(s, dict)],
                curriculum_id=curriculum_id, change_summary=_text(change_summary, 500),
                alignment=alignment if isinstance(alignment, dict) else None,
                framework_id=_text(framework_id, 40), topic=_text(topic, 120), origin=origin,
                source_request=_text(source_request, 1000))
            # "status" is the tool's result; the review status has its own key.
            return {"status": "ok", "curriculum_id": saved["curriculum_id"], "version": saved["version"],
                    "review_status": saved["status"], "validation": validation}
        return self._safely("save_curriculum_draft", run)

    def get_curriculum(self, curriculum_id, version=None) -> dict:
        def run():
            record = self.store.get(curriculum_id, int(version) if version is not None else None)
            if record is None:
                return {"status": "not_found", "message": f"No curriculum {curriculum_id} (version {version})."}
            return {"status": "ok", "curriculum": record, "versions": self.store.versions(curriculum_id),
                    "documents": self.store.documents(curriculum_id)}
        return self._safely("get_curriculum", run)

    def list_curricula(self, status=None, limit=50) -> dict:
        def run():
            if status is not None and status not in STATUSES:
                return _invalid(f"status must be one of {', '.join(STATUSES)}.")
            return {"status": "ok", "curricula": self.store.list(status, max(1, min(int(limit), 200)))}
        return self._safely("list_curricula", run)

    # ---------------------------------------------------------------- PDF

    def render_curriculum_pdf(self, curriculum_id, version=None, language=None) -> dict:
        def run():
            from agents.curriculum_research.pdf import render_pdf
            record = self.store.get(curriculum_id, int(version) if version is not None else None)
            if record is None:
                return {"status": "not_found", "message": f"No curriculum {curriculum_id}."}
            language_ = language or record["content"].get("language", "en")
            if language_ not in ("ar", "en"):
                return _invalid("language must be ar or en.")
            if language_ != record["content"].get("language"):
                return _invalid("The PDF is rendered in the curriculum's own language; "
                                "revise the curriculum to change its language.")
            data, pages = render_pdf(record)
            filename = f"{curriculum_id}_v{record['version']}_{language_}.pdf"
            if not re.fullmatch(r"CUR-[0-9a-f]{12}_v\d+_(ar|en)\.pdf", filename):
                return _invalid("Unsafe file name.")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            (self.documents_dir / filename).write_bytes(data)
            document = self.store.add_document(curriculum_id, record["version"], language_, filename, data, pages)
            return {"status": "ok", "document": document}
        return self._safely("render_curriculum_pdf", run)

    def document_bytes(self, document_id: str) -> Optional[bytes]:
        """For AdminService (UI download) only; not an MCP tool."""
        document = self.store.document(document_id)
        if document is None:
            return None
        path = self.documents_dir / document["filename"]
        return path.read_bytes() if path.is_file() else None

    def close(self) -> None:
        pass


def toolbox_from_settings() -> CurriculumToolbox:
    """The toolbox the MCP server uses: settings from the environment / .env."""
    index_dir = get_setting(INDEX_SETTING, "")
    catalogue = None
    if index_dir:
        from knowledge_base.vector_store import VectorStore
        catalogue = AcademyCatalogue(vector_store=VectorStore(persist_directory=index_dir))
    return CurriculumToolbox(catalogue=catalogue)
