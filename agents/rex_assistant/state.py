"""
Conversation state for Agent 1 (REX Assistant).

Kept intentionally small: only the fields Agent 1's own requirements
need (learner profile, language preference, consent/lead status, and
the last RAG result). No database or persistence system is introduced
here -- the orchestrator (built separately) is expected to own session
storage and pass this dict back in on each turn.
"""

from typing import Any, Optional, TypedDict


class LearnerProfile(TypedDict, total=False):
    background: Optional[str]
    career_goal: Optional[str]
    skill_level: Optional[str]
    interests: Optional[str]
    budget: Optional[str]
    available_time: Optional[str]


PROFILE_FIELDS_IN_PRIORITY_ORDER = (
    "background",
    "career_goal",
    "skill_level",
)


class RexAssistantState(TypedDict, total=False):
    # Set by the caller on every turn.
    user_input: str

    # Carried across turns by the caller (see interface.py).
    preferred_language: Optional[str]
    profile: LearnerProfile
    consent: Optional[bool]
    consent_pending: bool
    lead_captured: bool
    history: list[dict[str, str]]

    # Derived during this turn's graph run.
    detected_language: str
    response_language: Optional[str]
    rag_result: Optional[dict[str, Any]]
    response: str
