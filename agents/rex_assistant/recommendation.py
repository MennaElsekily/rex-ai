"""
The learner profile as REX shows it to the model when recommending.

Recommendations themselves are built from verified catalogue records in
grounded_recommendation.py. The free-text checks that used to live here
(dropping course names not found in the evidence, fixing codes, marking
levels and fees in the model's own text) were removed: a course name is
never taken from the model's text any more, so there is nothing to check.
"""

from __future__ import annotations

from typing import Any

PROFILE_FIELD_LABELS = (
    ("background", "Background"),
    ("career_goal", "Career goal"),
    ("skill_level", "Skill level"),
    ("training_preference", "Training preference (online/offline)"),
    ("budget", "Budget"),
    ("available_time", "Available time"),
    ("interests", "Interests"),
)
_OPTIONAL_FIELDS = ("interests",)


def profile_block(profile: dict[str, Any], declined: list[str]) -> str:
    lines = []
    for field, label in PROFILE_FIELD_LABELS:
        value = profile.get(field)
        if value:
            lines.append(f"- {label}: {value}")
        elif field in declined:
            lines.append(f"- {label}: not provided (the learner skipped it)")
        elif field not in _OPTIONAL_FIELDS:
            lines.append(f"- {label}: not provided")
    return "LEARNER PROFILE:\n" + "\n".join(lines)
