"""
Agent 5's tools, as one validated toolbox.

agents/marketing_mcp_server.py exposes the tool methods below over MCP and the agent reaches them through
agents/marketing/mcp_client.py. Tests use the same toolbox in-process. Every tool method checks its input, never
raises for ordinary failures, and returns a JSON-ready dict with a "status" ("ok", "invalid_request", "not_found",
"not_approved" or "error") and a "message" on failure.

Boundaries:
- courses are read through the Curriculum Agent's own CurriculumStore public methods (get, list_approved,
  count_approved), never its tables; only a version whose review status is "approved" is ever used, and nothing is
  written to that store. The approval rule below is this agent's own (tests/test_marketing_tools.py checks that it
  agrees with Agent 4's for every status a curriculum can be in);
- the save tool validates the campaign itself against the approved course (validation.validate_content) and can only
  store "draft" or "pending_review": approval exists only in AdminService;
- nothing here publishes, e-mails or sends anything.

Three methods are deliberately NOT tools, because they need an administrator identity or return file contents:
save_admin_edit, approve_campaign and approved_export (called by AdminService after its session check).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from agents.config import get_setting
from agents.curriculum_research.curriculum_store import (
    APPROVED as CURRICULUM_APPROVED,
    CurriculumStore,
    CurriculumStoreError,
)
from agents.marketing import composer
from agents.marketing.marketing_store import AGENT_ACTOR, MarketingStore, MarketingStoreError
from agents.marketing.validation import validate_content

logger = logging.getLogger(__name__)

DB_SETTING = "MARKETING_DB_PATH"
MAX_CAMPAIGN_BYTES = 200_000
EXPORT_FORMATS = ("markdown", "json")


def _invalid(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "invalid_request", "message": message, **extra}


class MarketingToolbox:
    def __init__(self, store: Optional[MarketingStore] = None, curricula: Optional[CurriculumStore] = None) -> None:
        self.store = store or MarketingStore(get_setting(DB_SETTING, "") or None)
        self.curricula = curricula or CurriculumStore(get_setting("CURRICULUM_DB_PATH", "") or None)

    def _safely(self, name: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return call()
        except (MarketingStoreError, CurriculumStoreError, ValueError) as exc:
            return _invalid(str(exc))
        except Exception as exc:  # never let an internal error escape as a crash
            logger.exception("Marketing tool %s failed", name)
            return {"status": "error", "message": f"The {name} tool failed ({type(exc).__name__})."}

    # ------------------------------------------------------------ courses

    def _approved_record(self, curriculum_id: Any, version: Optional[int] = None) -> dict[str, Any]:
        """
        The approved version of a curriculum (or the given version, if it is approved). Raises LookupError (unknown)
        or PermissionError (not approved) with the reason. Unapproved drafts, pending versions and versions with
        changes requested are never returned, and a newer unapproved revision never replaces the approved one.
        """
        head = self.curricula.get(curriculum_id)
        if head is None:
            raise LookupError(f"No course {curriculum_id}.")
        approved_version = head.get("approved_version")
        if approved_version is None:
            raise PermissionError(
                f"Course {curriculum_id} is {head['status']}: an administrator must approve it (Curricula tab) "
                "before marketing content can be created from it.")
        record = self.curricula.get(curriculum_id, version or approved_version)
        if record is None or record.get("version_status") != CURRICULUM_APPROVED:
            raise PermissionError(f"Version {version} of course {curriculum_id} is not an approved version.")
        record["current_version"] = head["current_version"]
        return record

    def ensure_curriculum_approved(self, curriculum_id: str, version: int) -> None:
        """For AdminService before it approves a campaign: the course version must still be an approved one."""
        try:
            self._approved_record(curriculum_id, version)
        except (LookupError, PermissionError) as exc:
            raise ValueError(str(exc)) from exc

    def list_approved_courses(self, limit=50, offset=0) -> dict:
        """One page of the courses that have an approved version, newest first, with the total."""
        def run():
            limit_, offset_ = max(1, min(int(limit), 200)), max(0, int(offset))
            rows = self.curricula.list_approved(limit_, offset_)
            total = self.curricula.count_approved()
            keep = ("curriculum_id", "title", "language", "level", "total_hours", "approved_version", "current_version",
                    "status", "target_audience")
            return {"status": "ok", "courses": [{k: r.get(k) for k in keep} for r in rows], "total": total,
                    "offset": offset_, "has_more": offset_ + len(rows) < total}
        return self._safely("list_approved_courses", run)

    def get_approved_course(self, curriculum_id, version=None) -> dict:
        def run():
            try:
                record = self._approved_record(curriculum_id, int(version) if version is not None else None)
            except LookupError as exc:
                return {"status": "not_found", "message": str(exc)}
            except PermissionError as exc:
                return {"status": "not_approved", "message": str(exc)}
            return {"status": "ok", "course": record, "newer_unapproved_version": record["current_version"] > record["version"]}
        return self._safely("get_approved_course", run)

    # ---------------------------------------------------------- campaigns

    def save_campaign_draft(self, content, campaign_id=None, change_summary=None) -> dict:
        """Save a new campaign or a new version; the tool validates it itself."""
        return self._save(content, campaign_id, change_summary, AGENT_ACTOR)

    def save_admin_edit(self, content, campaign_id, admin: str, change_summary=None) -> dict:
        """An administrator's edited version (not an MCP tool: it records who edited)."""
        if not isinstance(admin, str) or not admin.strip() or admin.startswith("system:"):
            return _invalid("An administrator identity is required.")
        return self._save(content, campaign_id, change_summary or "Edited by an administrator", admin)

    def _save(self, content, campaign_id, change_summary, actor) -> dict:
        def run():
            if not isinstance(content, dict) or len(json.dumps(content, ensure_ascii=False)) > MAX_CAMPAIGN_BYTES:
                return _invalid("content must be an object of reasonable size.")
            try:
                record = self._approved_record(content.get("curriculum_id"), content.get("curriculum_version"))
            except LookupError as exc:
                return {"status": "not_found", "message": str(exc)}
            except PermissionError as exc:
                return {"status": "not_approved", "message": str(exc)}
            validation = validate_content(content, record)
            saved = self.store.save_version(content, validation, campaign_id=campaign_id,
                                            change_summary=_clip(change_summary), created_by=actor)
            return {"status": "ok", "campaign_id": saved["campaign_id"], "version": saved["version"],
                    "review_status": saved["status"], "validation": validation}
        return self._safely("save_campaign_draft", run)

    def get_campaign(self, campaign_id, version=None) -> dict:
        def run():
            record = self.store.get(campaign_id, int(version) if version is not None else None)
            if record is None:
                return {"status": "not_found", "message": f"No campaign {campaign_id}."}
            return {"status": "ok", "campaign": record, "versions": self.store.versions(campaign_id)}
        return self._safely("get_campaign", run)

    def list_campaigns(self, status=None, curriculum_id=None, limit=50) -> dict:
        def run():
            return {"status": "ok", "campaigns": self.store.list(status, curriculum_id, max(1, min(int(limit), 200)))}
        return self._safely("list_campaigns", run)

    def approve_campaign(self, campaign_id: str, version: int, admin: str, notes: Optional[str] = None) -> dict:
        """Admin only (AdminService, after its session check). The course version must still be an approved one."""
        record = self.store.check_approvable(campaign_id, version, admin)
        self.ensure_curriculum_approved(record["curriculum_id"], record["curriculum_version"])
        return self.store.approve(campaign_id, version, admin, notes)

    def approved_export(self, campaign_id: str, fmt: str = "markdown") -> dict[str, Any]:
        """
        {"filename", "data", "mime", "version"} of the APPROVED version of a campaign. Raises ValueError if there is
        none: draft and pending-review content is never exported as if it were ready.
        """
        if fmt not in EXPORT_FORMATS:
            raise ValueError("The download format must be markdown or json.")
        record = self.store.get_approved(campaign_id)
        if record is None:
            raise ValueError("This campaign has no approved version. Approve it first: only approved content can be downloaded.")
        content, version = record["content"], record["version"]
        stem = f"{campaign_id}_v{version}_{record['language']}"
        if fmt == "markdown":
            text = composer.to_markdown(content, version=version, approved_at=record["approved_at"])
            return {"filename": f"{stem}.md", "data": text.encode("utf-8"), "mime": "text/markdown", "version": version}
        text = composer.to_json(content, campaign_id=campaign_id, version=version, approved_at=record["approved_at"])
        return {"filename": f"{stem}.json", "data": text.encode("utf-8"), "mime": "application/json", "version": version}

    def close(self) -> None:
        pass


def _clip(text: Any, limit: int = 300) -> Optional[str]:
    cleaned = " ".join(str(text or "").split())[:limit]
    return cleaned or None


def toolbox_from_settings() -> MarketingToolbox:
    """The toolbox the MCP server uses: settings from the environment / .env."""
    return MarketingToolbox()
