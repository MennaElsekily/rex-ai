"""
Conversation state for Agent 1 (REX Assistant).

The caller keeps this dict between turns (see interface.py). The User
interface keeps it in memory for the session: nothing here is written to
disk unless the learner consents to saving a lead (lead_store.py).
"""

from typing import Any, Optional, TypedDict


class LearnerProfile(TypedDict, total=False):
    background: Optional[str]
    career_goal: Optional[str]
    skill_level: Optional[str]
    training_preference: Optional[str]
    budget: Optional[str]
    available_time: Optional[str]
    interests: Optional[str]


class RexAssistantState(TypedDict, total=False):
    # Set by the caller on every turn.
    user_input: str

    # Carried across turns by the caller (see interface.py).
    # preferred_language: only an EXPLICIT choice ("answer in English").
    # conversation_language: the language the user writes in; kept when
    # a message is too short to tell ("ok", "5000").
    preferred_language: Optional[str]
    conversation_language: Optional[str]
    profile: LearnerProfile
    consent: Optional[bool]
    consent_pending: bool
    lead_captured: bool
    history: list[dict[str, str]]

    # Personalized-discovery progress (agents/rex_assistant/discovery.py).
    # mode is "discovery" while questions are being asked; a direct course
    # question in between leaves it, and pending_field, untouched.
    mode: Optional[str]
    pending_field: Optional[str]
    field_attempts: dict[str, int]
    declined_fields: list[str]
    discovery_intro_shown: bool

    # The last direct course question, so a follow-up such as "how long
    # does it take?" can be searched in its context.
    active_topic: Optional[str]

    # Derived during this turn's graph run.
    detected_language: Optional[str]
    response_language: Optional[str]
    intent: str
    next_step: str
    answer_mode: str
    clarify: bool
    evidence: Optional[dict[str, Any]]
    rag_result: Optional[dict[str, Any]]
    response: str
