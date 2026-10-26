
import json
from pathlib import Path
from typing import Any


class CurriculumData:
    """Loads, validates, and searches curriculum example data."""

    def __init__(self, data_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.data_path = (
            Path(data_path)
            if data_path is not None
            else project_root / "data" / "curriculum_examples.json"
        )

        self.records = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        """Load and validate curriculum records from JSON."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Curriculum data file not found: {self.data_path}"
            )

        with self.data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get(
                "curricula",
                data.get("examples", []),
            )
        else:
            raise ValueError(
                "Curriculum JSON must contain a list or an object."
            )

        if not isinstance(records, list):
            raise ValueError(
                "Expected curriculum records to be stored in a list."
            )

        if not all(isinstance(record, dict) for record in records):
            raise ValueError(
                "Every curriculum record must be a JSON object."
            )

        return records

    def get_all(self) -> list[dict[str, Any]]:
        """Return all loaded curriculum examples."""

        # Copy the list so callers cannot modify the stored list itself.
        return self.records.copy()

    @staticmethod
    def _to_searchable_text(value: Any) -> str:
        """Convert nested JSON content into searchable text."""

        if isinstance(value, dict):
            return " ".join(
                CurriculumData._to_searchable_text(item)
                for item in value.values()
            )

        if isinstance(value, list):
            return " ".join(
                CurriculumData._to_searchable_text(item)
                for item in value
            )

        if value is None:
            return ""

        return str(value)

    def search(self, keyword: str) -> list[dict[str, Any]]:
        """Search curriculum examples across their content."""

        if not isinstance(keyword, str):
            raise ValueError("Search keyword must be a string.")

        keyword = keyword.strip().casefold()

        if not keyword:
            return self.get_all()

        matches = []

        for record in self.records:
            searchable_text = self._to_searchable_text(record).casefold()

            if keyword in searchable_text:
                matches.append(record)

        return matches
