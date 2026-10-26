
import json
from pathlib import Path
from typing import Any


class CurriculumData:
    """Loads and provides access to curriculum example data."""

    def __init__(self, data_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.data_path = (
            Path(data_path)
            if data_path is not None
            else project_root / "data" / "curriculum_examples.json"
        )

        self.records = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        """Load curriculum records from the JSON file."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Curriculum data file not found: {self.data_path}"
            )

        with self.data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("curricula", data.get("examples", []))
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

        return self.records.copy()

    
    def search(self, keyword: str) -> list[dict[str, Any]]:
        """Search relevant curriculum fields by keyword."""

        keyword = keyword.strip().casefold()

        if not keyword:
            return self.get_all()

        matches = []

        for record in self.records:
            title = str(record.get("title", "")).casefold()
            audience = str(record.get("target_audience", "")).casefold()

            outcomes = " ".join(
                str(item)
                for item in record.get("learning_outcomes", [])
            ).casefold()

            modules = " ".join(
                str(module.get("title", ""))
                for module in record.get("modules", [])
                if isinstance(module, dict)
            ).casefold()

            searchable_text = f"{title} {audience} {outcomes} {modules}"

            if keyword in searchable_text:
                matches.append(record)

        return matches
