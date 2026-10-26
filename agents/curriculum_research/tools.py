
from typing import Any

from agents.curriculum_research.data import CurriculumData


class CurriculumTools:
    """Tools for searching and inspecting curriculum examples."""

    def __init__(self, data: CurriculumData | None = None):
        self.data = data or CurriculumData()

    def search_examples(self, keyword: str) -> dict[str, Any]:
        """Search curriculum examples by keyword."""

        try:
            matches = self.data.search(keyword)

            return {
                "status": "success",
                "query": keyword,
                "count": len(matches),
                "results": matches,
                "notice": (
                    "These are synthetic curriculum examples, "
                    "not verified accreditation standards."
                ),
            }

        except Exception as exc:
            return {
                "status": "error",
                "query": keyword,
                "count": 0,
                "results": [],
                "error_message": str(exc),
            }

    def get_curriculum_details(
        self, example_id: str
    ) -> dict[str, Any]:
        """Return one curriculum example by its ID."""

        for record in self.data.get_all():
            if record.get("example_id") == example_id:
                return {
                    "status": "success",
                    "result": record,
                    "notice": (
                        "This is an illustrative example, "
                        "not an accredited curriculum."
                    ),
                }

        return {
            "status": "error",
            "result": None,
            "error_message": (
                f"No curriculum example found with ID: {example_id}"
            ),
        }
