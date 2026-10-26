
import json
from typing import Any, TypedDict

from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------
# 1. Curriculum output schemas
# ---------------------------------------------------------

class CurriculumModule(BaseModel):
    module_number: int = Field(ge=1, le=4)
    title: str
    hours: int = Field(ge=0)
    topics: list[str]
    practical_exercises: list[str]


class Assessment(BaseModel):
    type: str
    description: str


class CapstoneProject(BaseModel):
    title: str
    description: str



class CurriculumDraft(BaseModel):
    title: str
    course_objectives: list[str]
    target_audience: str
    prerequisites: list[str]
    learning_outcomes: list[str]
    modules: list[CurriculumModule]

    # Small models may omit these fields.
    practical_exercises: list[str] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    capstone_project: CapstoneProject = Field(
        default_factory=lambda: CapstoneProject(
            title="Final Practical Project",
            description=(
                "Apply the skills learned in the course "
                "to complete a practical project."
            ),
        )
    )



# ---------------------------------------------------------
# 2. LangGraph state
# ---------------------------------------------------------

class CurriculumBuilderState(TypedDict, total=False):
    topic: str
    target_audience: str
    training_hours: int
    prerequisites: list[str]
    curriculum: dict[str, Any]
    error: str


# ---------------------------------------------------------
# 3. Curriculum builder
# ---------------------------------------------------------

class CurriculumBuilder:
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

    def __init__(self):
        # LangChain model connection to the local Ollama model.
        self.model = ChatOllama(
            model="llama3.2:1b",
            temperature=0.2,
            format="json",
        )

        # Parse and validate the model's curriculum response.
        self.parser = PydanticOutputParser(
            pydantic_object=CurriculumDraft
        )

        # Prompt used by the LangChain generation chain.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an educational curriculum designer.

Create a useful, detailed curriculum based on the user's
requirements.

Requirements:
- Return exactly four modules.
- Include measurable course objectives and learning outcomes.
- Include relevant topics and practical exercises.
- Include assessments and a capstone project.
- Adapt the content to the target audience and prerequisites.
- Keep the curriculum relevant to the requested topic.
- Do not invent accreditation, certification, or official approval.
- Write all curriculum content in English.
- The application will assign and validate module hours.

Return ONLY one valid JSON object containing:
- title: string
- course_objectives: array of strings
- target_audience: string
- prerequisites: array of strings
- learning_outcomes: array of strings
- modules: array of exactly four objects, each containing:
  module_number, title, hours, topics, practical_exercises
- practical_exercises: array of strings
- assessments: array of objects containing type and description
- capstone_project: object containing title and description

Return actual curriculum data, not a JSON schema.
Do not include Markdown fences or explanations.
"""
                ),
                (
                    "human",
                    """
Topic: {topic}
Target audience: {target_audience}
Total training hours: {training_hours}
Prerequisites: {prerequisites}

Generate the curriculum draft as JSON.
"""
                ),
            ]
        )

        # LangChain: prompt -> Ollama -> JSON text.
        self.generation_chain = self.prompt | self.model

        # LangGraph: define the curriculum generation workflow.
        graph = StateGraph(CurriculumBuilderState)

        graph.add_node(
            "generate_curriculum",
            self._generate_node,
        )

        graph.add_edge(START, "generate_curriculum")
        graph.add_edge("generate_curriculum", END)

        self.graph = graph.compile()

    # -----------------------------------------------------
    # 4. Public method
    # -----------------------------------------------------

    def build_curriculum(
        self,
        topic: str,
        target_audience: str = "Beginners",
        training_hours: int = 30,
        prerequisites: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate and validate a curriculum."""

        # Validate the input topic.
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic cannot be empty.")

        # Reject booleans because bool is a subclass of int.
        if (
            not isinstance(training_hours, int)
            or isinstance(training_hours, bool)
            or training_hours <= 0
        ):
            raise ValueError(
                "Training hours must be a positive integer."
            )

        if (
            not isinstance(target_audience, str)
            or not target_audience.strip()
        ):
            raise ValueError("Target audience cannot be empty.")

        if prerequisites is None:
            prerequisites = ["Basic computer skills"]

        if not isinstance(prerequisites, list):
            raise ValueError("Prerequisites must be a list.")

        cleaned_prerequisites = []

        for item in prerequisites:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "Each prerequisite must be a non-empty string."
                )
            cleaned_prerequisites.append(item.strip())

        initial_state: CurriculumBuilderState = {
            "topic": topic.strip(),
            "target_audience": target_audience.strip(),
            "training_hours": training_hours,
            "prerequisites": cleaned_prerequisites,
        }

        # Execute the LangGraph workflow.
        result = self.graph.invoke(initial_state)

        if "error" in result:
            raise RuntimeError(result["error"])

        if "curriculum" not in result:
            raise RuntimeError(
                "Curriculum generation returned no curriculum."
            )

        return result["curriculum"]

    # -----------------------------------------------------
    # 5. LangGraph generation node
    # -----------------------------------------------------

    def _generate_node(
        self,
        state: CurriculumBuilderState,
    ) -> dict[str, Any]:
        """Generate, parse, and validate the curriculum."""

        response = self.generation_chain.invoke(
            {
                "topic": state["topic"],
                "target_audience": state["target_audience"],
                "training_hours": state["training_hours"],
                "prerequisites": state["prerequisites"],
            }
        )

        # ChatOllama returns an AI message.
        response_text = response.content

        # Handle unexpected non-string output.
        if not isinstance(response_text, str):
            raise ValueError(
                "The language model returned an invalid response."
            )

        # Remove Markdown fences if the model adds them anyway.
        response_text = response_text.strip()

        if response_text.startswith("```"):
            lines = response_text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response_text = "\n".join(lines).strip()

        # Parse actual JSON data, not the JSON schema.
        try:
            raw_data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ollama did not return valid JSON. "
                "Try running the request again."
            ) from exc

        # Validate JSON against the expected Pydantic schema.
        try:
            draft = CurriculumDraft.model_validate(raw_data)
        except ValidationError as exc:
            raise ValueError(
                "The generated curriculum does not match "
                f"the required structure:\n{exc}"
            ) from exc

        curriculum = draft.model_dump()

        # Exactly four modules are required.
        if len(curriculum["modules"]) != 4:
            raise ValueError(
                "The generated curriculum must contain "
                "exactly four modules."
            )

        # Assign module hours so the total exactly matches
        # the requested training hours.
        total_hours = state["training_hours"]
        base_hours, remainder = divmod(total_hours, 4)

        for index, module in enumerate(curriculum["modules"]):
            module["module_number"] = index + 1

            # Distribute remaining hours among the first modules.
            module["hours"] = (
                base_hours + (1 if index < remainder else 0)
            )

        # Ensure application inputs are authoritative.
        curriculum["title"] = curriculum["title"].strip()
        curriculum["target_audience"] = state["target_audience"]
        curriculum["prerequisites"] = state["prerequisites"]
        curriculum["training_hours"] = total_hours
        curriculum["status"] = "draft"

        # Combine module exercises and curriculum-level exercises.
        all_exercises = list(curriculum["practical_exercises"])

        for module in curriculum["modules"]:
            all_exercises.extend(module["practical_exercises"])

        # Remove duplicate exercises while preserving order.
        curriculum["practical_exercises"] = list(
            dict.fromkeys(all_exercises)
        )

        # Validate the final module-hour allocation.
        allocated_hours = sum(
            module["hours"] for module in curriculum["modules"]
        )

        if allocated_hours != total_hours:
            raise ValueError(
                "Module hours do not equal the requested "
                "training hours."
            )

        # Validate all required sections.
        missing_sections = [
            section
            for section in self.REQUIRED_SECTIONS
            if section not in curriculum
        ]

        if missing_sections:
            raise ValueError(
                "Missing required curriculum sections: "
                + ", ".join(missing_sections)
            )

        curriculum["notice"] = (
            "This curriculum was AI-generated and requires "
            "review by a qualified educational professional. "
            "Accreditation, compliance, and official approval "
            "have not been verified."
        )

        return {"curriculum": curriculum}
