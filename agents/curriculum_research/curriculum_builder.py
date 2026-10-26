
from typing import Any


class CurriculumBuilder:
    """Build structured training curricula from user requirements."""

    REQUIRED_SECTIONS = [
        "course_objectives",
        "target_audience",
        "prerequisites",
        "learning_outcomes",
        "modules",
        "training_hours",
        "practical_exercises",
        "assessments",
        "capstone_project",
    ]

    def build_curriculum(
        self,
        topic: str,
        target_audience: str = "Beginners",
        training_hours: int = 30,
        prerequisites: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a structured curriculum draft."""

        topic = topic.strip()

        if not topic:
            raise ValueError("The curriculum topic cannot be empty.")

        if training_hours <= 0:
            raise ValueError("Training hours must be greater than zero.")

        prerequisites = prerequisites or ["Basic computer skills"]

        # Divide the requested hours across four modules.
        base_hours = training_hours // 4
        remainder = training_hours % 4
        module_hours = [
            base_hours + (1 if i < remainder else 0)
            for i in range(4)
        ]

        modules = [
            {
                "module_number": 1,
                "title": f"Introduction to {topic}",
                "hours": module_hours[0],
                "topics": [
                    f"Fundamental concepts of {topic}",
                    "Key terminology and applications",
                ],
                "practical_exercises": [
                    f"Identify real-world applications of {topic}"
                ],
            },
            {
                "module_number": 2,
                "title": f"Core Skills in {topic}",
                "hours": module_hours[1],
                "topics": [
                    f"Essential methods and techniques in {topic}",
                    "Tools and workflows",
                ],
                "practical_exercises": [
                    f"Complete a guided exercise using {topic}"
                ],
            },
            {
                "module_number": 3,
                "title": f"Applied {topic}",
                "hours": module_hours[2],
                "topics": [
                    "Practical problem-solving",
                    "Common challenges and good practices",
                ],
                "practical_exercises": [
                    f"Solve a practical problem related to {topic}"
                ],
            },
            {
                "module_number": 4,
                "title": f"{topic} Project",
                "hours": module_hours[3],
                "topics": [
                    "Project planning",
                    "Implementation and presentation",
                ],
                "practical_exercises": [
                    f"Develop a small project demonstrating {topic}"
                ],
            },
        ]

        return {
            "title": f"{topic} Training Curriculum",
            "status": "draft",
            "course_objectives": [
                f"Understand the fundamental concepts of {topic}",
                f"Apply relevant techniques to practical problems",
                f"Demonstrate learning through a final project",
            ],
            "target_audience": target_audience,
            "prerequisites": prerequisites,
            "learning_outcomes": [
                f"Explain the main concepts of {topic}",
                f"Apply appropriate methods to a practical task",
                f"Evaluate the results of work related to {topic}",
                f"Present a completed project related to {topic}",
            ],
            "modules": modules,
            "training_hours": training_hours,
            "practical_exercises": [
                exercise
                for module in modules
                for exercise in module["practical_exercises"]
            ],
            "assessments": [
                {
                    "type": "Module exercises",
                    "description": "Assess practical skills throughout the course",
                },
                {
                    "type": "Final assessment",
                    "description": "Evaluate understanding of the course concepts",
                },
            ],
            "capstone_project": {
                "title": f"{topic} Capstone Project",
                "description": (
                    f"Complete and present a practical project demonstrating "
                    f"the skills learned in {topic}."
                ),
            },
            "notice": (
                "This is an automatically generated draft. "
                "Review the content and verify it against credible sources "
                "and applicable standards before approval."
            ),
        }
