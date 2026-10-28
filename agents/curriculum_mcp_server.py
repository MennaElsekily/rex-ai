"""
MCP server for Agent 3 (Curriculum Research & Accreditation Alignment).

Each tool calls the matching method of
agents/curriculum_research/curriculum_tools.CurriculumToolbox, where the
logic and input checks live. There is deliberately no approve or
request-changes tool: an MCP server cannot tell who is calling, so
review decisions stay behind the admin session in agents/admin_service.py.

Settings (environment or .env, all optional): CURRICULUM_DB_PATH,
CURRICULUM_DOCUMENTS_DIR, CURRICULUM_FRAMEWORKS_DIR, CURRICULUM_INDEX_DIR,
CURRICULUM_RESEARCH_DOMAINS, TAVILY_API_KEY (read here, never returned).

Run (normally started by CurriculumMcpClient):
    python -m agents.curriculum_mcp_server
"""

import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from agents.curriculum_research.curriculum_tools import CurriculumToolbox, toolbox_from_settings

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)

mcp = MCPServer("REX AI Curriculum Research")

_toolbox: Optional[CurriculumToolbox] = None


def _tools() -> CurriculumToolbox:
    global _toolbox
    if _toolbox is None:
        _toolbox = toolbox_from_settings()
    return _toolbox


@mcp.tool()
def list_academy_courses(department: Optional[str] = None, level: Optional[str] = None,
                         audience_contains: Optional[str] = None, limit: int = 100) -> dict:
    """
    The academy's courses from the existing knowledge-base index, with
    their metadata (department, level, target audience, duration, total
    hours, industry, accreditation, language, update date, data status,
    source files) and the fields no source states ("missing"). Read-only.
    """
    return _tools().list_academy_courses(department, level, audience_contains, limit)


@mcp.tool()
def get_course_details(course: str) -> dict:
    """
    One academy course by id or name (any language): its metadata, its
    sections as written (outcomes, topics, skills...) and how it was
    matched ("id", "name" or "search"). Read-only.
    """
    return _tools().get_course_details(course)


@mcp.tool()
def check_course_overlap(title: str, topics: Optional[list[str]] = None,
                         outcomes: Optional[list[str]] = None) -> dict:
    """
    How much a proposed course overlaps existing courses (name similarity,
    shared terms) and whether it looks like a duplicate. Read-only.
    """
    return _tools().check_course_overlap(title, topics, outcomes)


@mcp.tool()
def catalogue_gap_report(department: Optional[str] = None) -> dict:
    """
    Confirmed observations from the catalogue: levels each department
    lacks, missing prerequisite courses, single-course departments,
    near-duplicate courses, and each department's topics and skills.
    Gap ids are "G1"... Read-only.
    """
    return _tools().catalogue_gap_report(department)


@mcp.tool()
def search_curriculum_sources(query: str, max_results: int = 5) -> dict:
    """
    Search approved research domains only (CURRICULUM_RESEARCH_DOMAINS).
    Returns sources with ids, title, URL, publisher, date, excerpt and
    retrieval time; "not_configured" when no domain is approved.
    """
    return _tools().search_curriculum_sources(query, max_results)


@mcp.tool()
def list_frameworks() -> dict:
    """Approved framework documents available for alignment (may be none)."""
    return _tools().list_frameworks()


@mcp.tool()
def get_framework_requirements(framework_id: str) -> dict:
    """An approved framework's provenance and requirements, or not_found."""
    return _tools().get_framework_requirements(framework_id)


@mcp.tool()
def validate_curriculum(curriculum: dict[str, Any], requested_hours: int,
                        framework_id: Optional[str] = None) -> dict:
    """
    Deterministic checks: sections, types, module hours and their total,
    measurable outcomes, outcome links, sources, alignment evidence, no
    accreditation claims, language. Returns valid, errors, warnings.
    """
    return _tools().validate_curriculum(curriculum, requested_hours, framework_id)


@mcp.tool()
def compare_with_framework(curriculum: dict[str, Any], framework_id: str) -> dict:
    """
    A deterministic first pass of alignment: for each requirement of an
    approved framework, the curriculum elements sharing terms with it.
    """
    return _tools().compare_with_framework(curriculum, framework_id)


@mcp.tool()
def save_curriculum_draft(curriculum: dict[str, Any], sources: Optional[list[dict[str, Any]]] = None,
                          curriculum_id: Optional[str] = None, change_summary: Optional[str] = None,
                          alignment: Optional[dict[str, Any]] = None, framework_id: Optional[str] = None,
                          topic: Optional[str] = None, origin: str = "request",
                          source_request: Optional[str] = None) -> dict:
    """
    Save a new curriculum or a new version of one. The server validates it
    itself; the status is pending_review when valid, otherwise draft.
    Cannot approve.
    """
    return _tools().save_curriculum_draft(curriculum, sources, curriculum_id, change_summary, alignment,
                                          framework_id, topic, origin, source_request)


@mcp.tool()
def get_curriculum(curriculum_id: str, version: Optional[int] = None) -> dict:
    """A curriculum (current or given version) with its version history and documents."""
    return _tools().get_curriculum(curriculum_id, version)


@mcp.tool()
def list_curricula(status: Optional[str] = None, limit: int = 50) -> dict:
    """Saved curricula, optionally by review status."""
    return _tools().list_curricula(status, limit)


@mcp.tool()
def render_curriculum_pdf(curriculum_id: str, version: Optional[int] = None,
                          language: Optional[str] = None) -> dict:
    """Render a curriculum version as a PDF (English or Arabic) and record the document."""
    return _tools().render_curriculum_pdf(curriculum_id, version, language)


if __name__ == "__main__":
    mcp.run()
