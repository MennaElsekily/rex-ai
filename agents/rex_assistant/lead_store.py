"""
Lead-capture storage for Agent 1 (REX Assistant).

The project brief (Section 3, Agent 1, requirement 8) requires capturing
basic lead information for follow-up, with the user's consent. No
production storage destination (CRM, database, spreadsheet) has been
specified for this yet, so lead-capture *behavior* (implemented and
tested in graph.py) is kept separate from its storage *destination*
here. Swapping JSONLLeadStore for a real backend later requires no
change to the conversation logic -- only a different LeadStore
implementation passed into RexAssistantAgent.

Do not treat JSONLLeadStore as the intended production destination: it
exists only so lead capture can be exercised and tested now.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LeadStore(ABC):
    """Storage interface for captured leads."""

    @abstractmethod
    def save_lead(self, lead: dict[str, Any]) -> bool:
        """Persist one lead record. Return True on success."""


class JSONLLeadStore(LeadStore):
    """
    Placeholder lead storage: appends consented leads to a local JSONL
    file. NOT a production integration -- no CRM, database, or
    spreadsheet destination has been defined by the project yet.
    """

    def __init__(
        self,
        path: str | Path = "agents/rex_assistant/leads.jsonl",
    ) -> None:
        self.path = Path(path)

    def save_lead(self, lead: dict[str, Any]) -> bool:
        if not isinstance(lead, dict):
            raise TypeError("lead must be a dictionary")

        record = {
            **lead,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return True
