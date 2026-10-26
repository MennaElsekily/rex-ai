
import json

from agents.curriculum_research.agent import CurriculumResearchAgent


class CurriculumResearchServer:
    """Local entry point for testing the curriculum research agent."""

    def __init__(self):
        self.agent = CurriculumResearchAgent()

    def handle_request(self, request: dict) -> dict:
        """Pass a request to the agent and return its response."""
        return self.agent.process_request(request)


def main():
    server = CurriculumResearchServer()

    request = {
        "request_id": "curriculum-test-001",
        "user_input": "Find a curriculum about data analysis",
        "language": "en",
        "context": {},
    }

    response = server.handle_request(request)
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
