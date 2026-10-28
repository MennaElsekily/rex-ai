"""
State for Agent 3's LangGraph workflow (graph.py).

``CurriculumState`` lives for one request. What must survive between
messages of a conversation is kept in the shared ConversationStore under
"curriculum_research" as a ``ConversationMemory`` (curriculum_agent.py).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class Slots(TypedDict, total=False):
    topic: Optional[str]
    course_ref: Optional[str]          # a catalogue course id or name named in the request
    total_hours: Optional[int]
    level: Optional[str]
    target_audience: Optional[str]
    capstone: Optional[bool]           # False when the admin asked for no capstone
    origin: str                        # "request" or "suggestion"
    framework_id: Optional[str]        # an approved framework to align with
    defaults_accepted: bool


class ConversationMemory(TypedDict, total=False):
    language: str                      # the conversation's reply language
    active_curriculum_id: Optional[str]
    active_version: Optional[int]
    pending: Optional[dict[str, Any]]  # {"slots", "missing", "request", "output_language"} while a question is open
    last_suggestions: list[dict[str, Any]]


class CurriculumState(TypedDict, total=False):
    # Input
    request_text: str
    request_id: str
    memory: ConversationMemory

    # Understanding
    language: str                      # reply language
    output_language: str               # curriculum language
    intent: str                        # generate | revise | suggest | align | show | approve_attempt | help | unsupported
    slots: Slots
    missing: list[str]
    revision: dict[str, Any]           # {"changes": [...], "instruction": str}
    regenerate: bool                   # the revision needs the model to rewrite the curriculum
    notes: list[str]                   # facts for the reply (e.g. "hours taken from the catalogue")

    # Evidence
    academy_evidence: list[dict[str, Any]]
    related_courses: list[dict[str, Any]]
    data_status: list[str]
    sources: list[dict[str, Any]]
    research_status: str

    # Curriculum
    base: Optional[dict[str, Any]]     # the stored version a revision starts from
    draft: Optional[dict[str, Any]]
    validation: Optional[dict[str, Any]]
    repair_attempts: int
    repairs: list[str]
    change_summary: Optional[str]
    saved: Optional[dict[str, Any]]
    document: Optional[dict[str, Any]]
    document_status: Optional[str]

    # Suggestions (Workflow B)
    gap_report: Optional[dict[str, Any]]
    suggestions: list[dict[str, Any]]

    # Output
    status: str                        # "success" | "error"
    response: dict[str, Any]
    memory_out: ConversationMemory
