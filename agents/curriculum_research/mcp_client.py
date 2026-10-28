"""
Persistent MCP client for Agent 3 to reach its tools through
agents/curriculum_mcp_server.py. The connection handling is shared with
the other agents (agents/mcp_tool_client.py); this class only names the
server and gives each tool a typed method with the same signature as
CurriculumToolbox, so tests can use the toolbox in-process instead.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.mcp_tool_client import McpToolClient


class CurriculumMcpClient(McpToolClient):
    def __init__(self, server_module: str = "agents.curriculum_mcp_server", startup_timeout: float = 180.0,
                 env: Optional[dict[str, str]] = None) -> None:
        super().__init__(server_module, startup_timeout=startup_timeout, server_name="curriculum", env=env)

    def list_academy_courses(self, department=None, level=None, audience_contains=None, limit=100):
        return self.call_tool("list_academy_courses", {"department": department, "level": level,
                                                        "audience_contains": audience_contains, "limit": limit})

    def get_course_details(self, course):
        return self.call_tool("get_course_details", {"course": course})

    def check_course_overlap(self, title, topics=None, outcomes=None):
        return self.call_tool("check_course_overlap", {"title": title, "topics": topics, "outcomes": outcomes})

    def catalogue_gap_report(self, department=None):
        return self.call_tool("catalogue_gap_report", {"department": department})

    def search_curriculum_sources(self, query, max_results=5):
        return self.call_tool("search_curriculum_sources", {"query": query, "max_results": max_results})

    def list_frameworks(self):
        return self.call_tool("list_frameworks", {})

    def get_framework_requirements(self, framework_id):
        return self.call_tool("get_framework_requirements", {"framework_id": framework_id})

    def validate_curriculum(self, curriculum, requested_hours, framework_id=None):
        return self.call_tool("validate_curriculum", {"curriculum": curriculum, "requested_hours": requested_hours,
                                                       "framework_id": framework_id})

    def compare_with_framework(self, curriculum, framework_id):
        return self.call_tool("compare_with_framework", {"curriculum": curriculum, "framework_id": framework_id})

    def save_curriculum_draft(self, curriculum, sources=None, curriculum_id=None, change_summary=None,
                              alignment=None, framework_id=None, topic=None, origin="request",
                              source_request=None) -> dict[str, Any]:
        return self.call_tool("save_curriculum_draft", {
            "curriculum": curriculum, "sources": sources, "curriculum_id": curriculum_id,
            "change_summary": change_summary, "alignment": alignment, "framework_id": framework_id,
            "topic": topic, "origin": origin, "source_request": source_request})

    def get_curriculum(self, curriculum_id, version=None):
        return self.call_tool("get_curriculum", {"curriculum_id": curriculum_id, "version": version})

    def list_curricula(self, status=None, limit=50):
        return self.call_tool("list_curricula", {"status": status, "limit": limit})

    def render_curriculum_pdf(self, curriculum_id, version=None, language=None):
        return self.call_tool("render_curriculum_pdf", {"curriculum_id": curriculum_id, "version": version,
                                                         "language": language})
