
import os
import re
from typing import Any, Callable
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


AgentHandler = Callable[[dict[str, Any]], dict[str, Any]]


class OrchestratorState(TypedDict, total=False):
    request: dict[str, Any]
    plan: list[str]
    results: list[dict[str, Any]]
    current_index: int
    response: dict[str, Any]


class Orchestrator:
    """Route requests and coordinate specialist agents using LangChain
    and LangGraph.
    """

    AGENT_NAMES = {
        "rex_assistant",
        "rfp_research",
        "curriculum_research",
        "document_generation",
        "marketing",
    }

    def __init__(self, model_name: str | None = None) -> None:
        self.agents: dict[str, AgentHandler] = {}

        model_name = model_name or os.getenv(
            "OLLAMA_MODEL", "llama3.2:1b"
        )

        self.routing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a request router for an educational "
                    "multi-agent system.\n"
                    "Choose only from these agent names:\n"
                    "- rex_assistant: course recommendations\n"
                    "- rfp_research: educational tenders, RFPs, RFQs\n"
                    "- curriculum_research: curriculum creation, "
                    "curriculum research, learning outcomes, "
                    "accreditation alignment\n"
                    "- document_generation: educational documents, "
                    "course proposals, guides, PDFs, question banks\n"
                    "- marketing: educational marketing, campaigns, "
                    "promotional and social media content\n\n"
                    "Return only the agent name. For a request that "
                    "clearly requires BOTH curriculum creation/research "
                    "and marketing, return exactly:\n"
                    "curriculum_research,marketing\n"
                    "For an unclear or unrelated request, return exactly:\n"
                    "none\n"
                    "Never invent agent names. Do not include explanations.",
                ),
                ("human", "{user_input}"),
            ]
        )

        self.router = self.routing_prompt | ChatOllama(
            model=model_name,
            temperature=0,
        )

        graph = StateGraph(OrchestratorState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._next_step,
            {"execute": "execute", "finish": END},
        )

        self.graph = graph.compile()

    @staticmethod
    def detect_language(text: str) -> str:
        """Return 'ar' for Arabic text and 'en' otherwise."""
        if any("\u0600" <= char <= "\u06FF" for char in text):
            return "ar"
        return "en"

    def register_agent(
        self, name: str, handler: AgentHandler
    ) -> None:
        """Register an implemented specialist agent."""
        if name not in self.AGENT_NAMES:
            raise ValueError(f"Unknown specialist agent: {name}")

        self.agents[name] = handler

    def handle_request(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Classify, route, execute, and combine specialist responses."""
        request_id = request_id or str(uuid4())
        user_input = user_input.strip()

        if not user_input:
            return self._error_response(
                request_id, "Please provide a request."
            )

        language = self.detect_language(user_input)

        request_context = dict(context or {})
        request_context["language"] = language

        request = {
            "request_id": request_id,
            "agent_name": "orchestrator",
            "user_input": user_input,
            "context": request_context,
            "language": language,
        }

        try:
            state = self.graph.invoke(
                {
                    "request": request,
                    "plan": [],
                    "results": [],
                    "current_index": 0,
                }
            )
            return state["response"]
        except Exception as error:
            return self._error_response(
                request_id, f"Orchestrator error: {error}"
            )

    def _plan_node(
        self, state: OrchestratorState
    ) -> dict[str, Any]:
        """Use the LangChain router, with deterministic fallback."""
        user_input = state["request"]["user_input"]

        try:
            plan = self._classify_request(user_input)
        except Exception:
            # Keep routing available if Ollama is unavailable.
            plan = self._make_plan(user_input)

        return {
            "plan": plan,
            "results": [],
            "current_index": 0,
        }

    def _classify_request(self, text: str) -> list[str]:
        """Classify a request and validate the selected agent names."""
        result = self.router.invoke({"user_input": text})
        output = str(result.content).strip().casefold()

        if output == "none":
            # Use known keyword rules as a secondary attempt.
            return self._make_plan(text)

        candidates = [
            item.strip()
            for item in re.split(r"[,;\n]+", output)
            if item.strip()
        ]

        # Ignore anything the model returns that is not an allowed agent.
        selected = [
            item for item in candidates
            if item in self.AGENT_NAMES
        ]

        # Only allow the explicitly supported multi-agent workflow.
        if set(selected) == {"curriculum_research", "marketing"}:
            return ["curriculum_research", "marketing"]

        if selected:
            return [selected[0]]

        return self._make_plan(text)

    def _make_plan(self, text: str) -> list[str]:
        """Deterministic fallback for common English and Arabic requests."""
        text = text.casefold()

        curriculum_terms = (
            "curriculum", "course outline", "syllabus",
            "منهج", "منهاج", "مخرجات التعلم", "خطة دراسية",
        )

        marketing_terms = (
            "marketing", "campaign", "social media",
            "advertisement", "advertising", "linkedin post",
            "promotional", "تسويق", "حملة تسويقية",
            "وسائل التواصل الاجتماعي", "إعلان", "اعلان",
        )

        # Explicit multi-agent workflow.
        if (
            any(term in text for term in curriculum_terms)
            and any(term in text for term in marketing_terms)
        ):
            return ["curriculum_research", "marketing"]

        # Agent 2: tender and RFP research.
        if any(term in text for term in (
            "tender", "tenders", "rfp", "rfq",
            "request for proposal", "request for quotation",
            "مناقصة", "مناقصات", "مناقصه", "عطاء", "عطاءات",
            "طلب تقديم عروض", "طلب عروض أسعار", "طلب عروض اسعار",
        )):
            return ["rfp_research"]

        # Agent 3: curriculum research.
        if any(term in text for term in curriculum_terms + (
            "learning outcomes", "accreditation",
            "learning objectives", "training plan",
            "أهداف التعلم", "الأهداف التعليمية", "اعتماد",
            "الاعتماد الأكاديمي", "الاعتماد الاكاديمي",
            "تصميم منهج", "تصميم منهج دراسي", "منهج تدريبي",
        )):
            return ["curriculum_research"]

        # Agent 5: marketing.
        if any(term in text for term in marketing_terms):
            return ["marketing"]

        # Agent 4: document generation.
        if any(term in text for term in (
            "generate a document", "create a document",
            "make a pdf", "participant guide", "instructor guide",
            "question bank", "proposal document", "create a proposal",
            "أنشئ مستند", "انشئ مستند", "إنشاء مستند",
            "إنشاء ملف pdf", "ملف pdf", "دليل المتدرب",
            "دليل المدرب", "بنك أسئلة", "بنك اسئلة",
            "إنشاء مقترح", "إنشاء عرض", "إعداد وثيقة",
        )):
            return ["document_generation"]

        # Agent 1: course recommendations.
        if any(term in text for term in (
            "recommend a course", "recommend courses",
            "which course", "course recommendation",
            "course recommendations", "what course should",
            "find me a course", "suggest a course", "training course",
            "courses for beginners", "رشح لي دورة", "رشح دورة",
            "رشح لي كورس", "اقترح دورة", "اقترح كورس",
            "أريد دورة", "اريد دورة", "أريد كورس", "اريد كورس",
            "الدورات المناسبة", "دورات للمبتدئين", "دورات تدريبية",
            "ما هي الدورات", "ماهي الدورات",
            "تعلم البرمجة", "تعلم تحليل البيانات",
        )):
            return ["rex_assistant"]

        return []

    def _execute_node(
        self, state: OrchestratorState
    ) -> dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("current_index", 0)
        results = list(state.get("results", []))
        request = state["request"]

        if not plan:
            results = [{
                "request_id": request["request_id"],
                "agent_name": "orchestrator",
                "status": "error",
                "result": None,
                "error_message": (
                    "I couldn't identify the right agent. Please specify "
                    "course recommendations, tender research, curriculum "
                    "research, document generation, or marketing. "
                    "يمكنك توضيح ما تحتاج إليه: ترشيح الدورات، أو البحث "
                    "عن المناقصات، أو إعداد المناهج، أو إنشاء المستندات، "
                    "أو التسويق."
                ),
            }]
            return {
                "results": results,
                "current_index": index,
                "response": self._combine_results(request, results),
            }

        if index >= len(plan):
            return {
                "results": results,
                "current_index": index,
                "response": self._combine_results(request, results),
            }

        agent_name = plan[index]
        handler = self.agents.get(agent_name)

        if handler is None:
            agent_result = {
                "request_id": request["request_id"],
                "agent_name": agent_name,
                "status": "error",
                "result": None,
                "error_message": (
                    f"Agent '{agent_name}' is not implemented or "
                    "registered yet."
                ),
            }
        else:
            agent_request = {
                "request_id": request["request_id"],
                "agent_name": agent_name,
                "user_input": request["user_input"],
                "context": {
                    **request.get("context", {}),
                    "language": request.get("language", "en"),
                    "previous_results": results,
                },
            }

            try:
                agent_result = handler(agent_request)

                if not isinstance(agent_result, dict):
                    raise TypeError("Agent must return a dictionary.")

                agent_result = {
                    "request_id": request["request_id"],
                    "agent_name": agent_result.get(
                        "agent_name", agent_name
                    ),
                    "status": agent_result.get("status", "success"),
                    "result": agent_result.get("result"),
                    "error_message": agent_result.get("error_message"),
                }
            except Exception as error:
                agent_result = {
                    "request_id": request["request_id"],
                    "agent_name": agent_name,
                    "status": "error",
                    "result": None,
                    "error_message": str(error),
                }

        results.append(agent_result)
        next_index = index + 1

        updates: dict[str, Any] = {
            "results": results,
            "current_index": next_index,
        }

        if next_index >= len(plan):
            updates["response"] = self._combine_results(request, results)

        return updates

    def _combine_results(
        self,
        request: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures = [
            result for result in results
            if result["status"] != "success"
        ]

        return {
            "request_id": request["request_id"],
            "agent_name": "orchestrator",
            "status": "error" if failures else "success",
            "result": results,
            "error_message": (
                "; ".join(
                    result.get("error_message") or "Agent failed"
                    for result in failures
                )
                if failures
                else None
            ),
        }

    def _next_step(self, state: OrchestratorState) -> str:
        if state.get("current_index", 0) < len(state.get("plan", [])):
            return "execute"
        return "finish"

    def _error_response(
        self, request_id: str, message: str
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "agent_name": "orchestrator",
            "status": "error",
            "result": None,
            "error_message": message,
        }


# Convenient default instance for importing elsewhere.
orchestrator = Orchestrator()
