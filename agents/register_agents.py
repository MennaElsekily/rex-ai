
from agents.orchestrator import Orchestrator
from agents.curriculum_research import CurriculumResearchAgent


def create_orchestrator() -> Orchestrator:
    """Create an orchestrator and register implemented agents."""

    orchestrator = Orchestrator()

    curriculum_agent = CurriculumResearchAgent()

    orchestrator.register_agent(
        "curriculum_research",
        curriculum_agent.process_request,
    )

    return orchestrator
