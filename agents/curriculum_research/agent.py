
import os
from typing import Any, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from agents.curriculum_research.curriculum_builder import CurriculumBuilder
from agents.curriculum_research.tools import CurriculumTools


class CurriculumState(TypedDict, total=False):
    request_id: str
    user_input: str
    context: dict[str, Any]
    language: str
    intent: Literal["create_curriculum", "search_examples"]
    response: dict[str, Any]


class CurriculumResearchAgent:
    """Curriculum agent using LangChain and LangGraph."""

    AGENT_NAME = "curriculum_research"

    def __init__(
        self,
        tools: CurriculumTools | None = None,
        builder: CurriculumBuilder | None = None,
        model_name: str | None = None,
    ):
        self.tools = tools or CurriculumTools()
        self.builder = builder or CurriculumBuilder()

        model_name = model_name or os.getenv(
            "OLLAMA_MODEL", "llama3.2:1b"
        )

        self.classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You classify educational curriculum requests. "
                    "Return exactly one label and nothing else:\n"
                    "create_curriculum - the user wants to create, "
                    "generate, design, or develop a new curriculum or course.\n"
                    "search_examples - the user wants to find, search, "
                    "inspect, or learn about existing curriculum examples.\n"
                    "If uncertain, return search_examples.",
                ),
                ("human", "{user_input}"),
            ]
        )

        self.classifier = self.classifier_prompt | ChatOllama(
            model=model_name,
            temperature=0,
        )

        # Build the agent workflow with LangGraph.
        graph = StateGraph(CurriculumState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("create_curriculum", self._create_node)
        graph.add_node("search_examples", self._search_node)

        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            self._route_intent,
            {
                "create_curriculum": "create_curriculum",
                "search_examples": "search_examples",
            },
        )

        graph.add_edge("create_curriculum", END)
        graph.add_edge("search_examples", END)

        self.graph = graph.compile()

    def process_request(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep the existing orchestrator-compatible interface."""

        request_id = request.get("request_id", "unknown")
        user_input = str(request.get("user_input", "")).strip()
        context = request.get("context") or {}

        language = (
            request.get("language")
            or context.get("language")
            or self._detect_language(user_input)
        )

        if not user_input:
            return self._error_response(
                request_id,
                language,
                "The curriculum request cannot be empty.",
            )

        initial_state: CurriculumState = {
            "request_id": request_id,
            "user_input": user_input,
            "context": context,
            "language": language,
        }

        try:
            final_state = self.graph.invoke(initial_state)
            return final_state["response"]
        except Exception as exc:
            return self._error_response(
                request_id, language, str(exc)
            )

    def _classify_node(
        self, state: CurriculumState
    ) -> dict[str, Any]:
        """Classify the request using LangChain and the local Ollama model."""

        user_input = state["user_input"]

        # Explicit creation phrases take priority for predictable behavior.
        if self._is_creation_request(user_input):
            return {"intent": "create_curriculum"}

        try:
            result = self.classifier.invoke(
                {"user_input": user_input}
            )
            label = str(result.content).strip().casefold()

            if "create_curriculum" in label:
                intent = "create_curriculum"
            else:
                intent = "search_examples"

        except Exception:
            # Preserve a usable fallback if Ollama is unavailable.
            intent = (
                "create_curriculum"
                if self._is_creation_request(user_input)
                else "search_examples"
            )

        return {"intent": intent}

    @staticmethod
    def _route_intent(
        state: CurriculumState,
    ) -> str:
        return state.get("intent", "search_examples")

    def _create_node(
        self, state: CurriculumState
    ) -> dict[str, Any]:
        """Generate a curriculum using the existing deterministic builder."""

        request_id = state["request_id"]
        language = state["language"]

        try:
            context = state.get("context", {})
            curriculum = self._build_from_request(
                state["user_input"], context
            )

            message = (
                "تم إنشاء مسودة منهج تدريبي جديدة."
                if language == "ar"
                else "A new curriculum draft has been generated."
            )

            response = {
                "request_id": request_id,
                "agent_name": self.AGENT_NAME,
                "status": "success",
                "result": {
                    "message": message,
                    "curricula": [curriculum],
                    "notice": curriculum["notice"],
                },
                "error_message": None,
            }

        except Exception as exc:
            response = self._error_response(
                request_id, language, str(exc)
            )

        return {"response": response}

    def _search_node(
        self, state: CurriculumState
    ) -> dict[str, Any]:
        """Search existing examples using the existing search tools."""

        request_id = state["request_id"]
        user_input = state["user_input"]
        language = state["language"]

        try:
            result = self.tools.search_examples(user_input)

            if result["status"] != "success":
                return {
                    "response": self._error_response(
                        request_id,
                        language,
                        result.get(
                            "error_message",
                            "Curriculum search failed.",
                        ),
                    )
                }

            # Try extracted phrases if the full request has no match.
            if result["count"] == 0:
                for keyword in self._extract_keywords(user_input):
                    if len(keyword.split()) < 2:
                        continue

                    result = self.tools.search_examples(keyword)

                    if result["status"] != "success":
                        return {
                            "response": self._error_response(
                                request_id,
                                language,
                                result.get(
                                    "error_message",
                                    "Curriculum search failed.",
                                ),
                            )
                        }

                    if result["count"] > 0:
                        break

            if result["count"] == 0:
                message = (
                    "لم يتم العثور على أمثلة مطابقة في البيانات المتاحة."
                    if language == "ar"
                    else "No matching examples were found in the available data."
                )

                response = {
                    "request_id": request_id,
                    "agent_name": self.AGENT_NAME,
                    "status": "success",
                    "result": {
                        "message": message,
                        "curricula": [],
                        "notice": (
                            "The available records are synthetic examples, "
                            "not verified accreditation standards."
                        ),
                    },
                    "error_message": None,
                }

            else:
                message = (
                    "تم العثور على أمثلة للمناهج ذات الصلة."
                    if language == "ar"
                    else "Relevant curriculum examples were found."
                )

                response = {
                    "request_id": request_id,
                    "agent_name": self.AGENT_NAME,
                    "status": "success",
                    "result": {
                        "message": message,
                        "curricula": result["results"],
                        "notice": (
                            "These are synthetic examples. "
                            "They are not verified or accredited curricula."
                        ),
                    },
                    "error_message": None,
                }

        except Exception as exc:
            response = self._error_response(
                request_id, language, str(exc)
            )

        return {"response": response}

    def _build_from_request(
        self,
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract settings and call the existing curriculum builder."""

        topic = self._extract_topic(user_input)

        return self.builder.build_curriculum(
            topic=topic,
            target_audience=context.get(
                "target_audience", "Beginners"
            ),
            training_hours=int(context.get("training_hours", 30)),
            prerequisites=context.get("prerequisites"),
        )

    @staticmethod
    def _is_creation_request(text: str) -> bool:
        normalized = text.casefold()

        creation_phrases = [
            "create a curriculum",
            "create curriculum",
            "generate a curriculum",
            "generate curriculum",
            "design a curriculum",
            "build a curriculum",
            "develop a curriculum",
            "create a course",
            "generate a course",
            "design a course",
            "make a curriculum",
            "إنشاء منهج",
            "انشاء منهج",
            "أنشئ منهج",
            "انشئ منهج",
            "صمم منهج",
            "صمّم منهج",
            "إعداد منهج",
            "اعداد منهج",
            "إنشاء دورة",
            "انشاء دورة",
            "صمم دورة",
        ]

        return any(
            phrase in normalized for phrase in creation_phrases
        )

    @staticmethod
    def _extract_topic(text: str) -> str:
        prefixes = [
            "create a curriculum about ",
            "create a curriculum for ",
            "create curriculum about ",
            "create curriculum for ",
            "generate a curriculum about ",
            "generate a curriculum for ",
            "design a curriculum about ",
            "design a curriculum for ",
            "build a curriculum about ",
            "build a curriculum for ",
            "develop a curriculum about ",
            "develop a curriculum for ",
            "create a course about ",
            "create a course for ",
            "generate a course about ",
            "design a course about ",
            "make a curriculum about ",
            "make a curriculum for ",
            "إنشاء منهج عن ",
            "انشاء منهج عن ",
            "أنشئ منهج عن ",
            "انشئ منهج عن ",
            "صمم منهج عن ",
            "صمّم منهج عن ",
            "إعداد منهج عن ",
            "اعداد منهج عن ",
            "إنشاء دورة عن ",
            "انشاء دورة عن ",
            "صمم دورة عن ",
        ]

        normalized = text.strip()

        for prefix in sorted(prefixes, key=len, reverse=True):
            if normalized.casefold().startswith(prefix.casefold()):
                topic = normalized[len(prefix):].strip(" .،؟?!")
                if topic:
                    return topic

        return normalized

    @staticmethod
    def _detect_language(text: str) -> str:
        if any("\u0600" <= char <= "\u06FF" for char in text):
            return "ar"
        return "en"

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        normalized = text.casefold()
        keywords: list[str] = []

        bilingual_terms = {
            "تحليل البيانات": ["data analysis"],
            "الأمن السيبراني": [
                "cybersecurity awareness", "cybersecurity"
            ],
            "امن سيبراني": [
                "cybersecurity awareness", "cybersecurity"
            ],
            "تصميم المناهج": [
                "learning outcomes", "curriculum"
            ],
            "مخرجات التعلم": ["learning outcomes"],
            "البرمجة": ["programming", "python"],
            "برمجة": ["programming", "python"],
            "التعلم": ["learning outcomes"],
            "البيانات": ["data analysis"],
            "تدريبي": ["training"],
            "تدريب": ["training"],
            "منهج": ["curriculum"],
            "منهاج": ["curriculum"],
        }

        for arabic_phrase, english_phrases in sorted(
            bilingual_terms.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if arabic_phrase in normalized:
                keywords.extend(english_phrases)

        if "data analysis" in normalized:
            keywords.insert(0, "data analysis")

        ignored = {
            "the", "and", "for", "with", "please",
            "find", "show", "create", "about", "curriculum",
        }

        for word in text.split():
            cleaned = word.strip(
                ".,!?؟،؛:()[]{}\"'"
            ).casefold()

            if (
                len(cleaned) > 2
                and cleaned not in ignored
                and not any(
                    "\u0600" <= char <= "\u06FF"
                    for char in cleaned
                )
            ):
                keywords.append(cleaned)

        return list(dict.fromkeys(keywords))

    def _error_response(
        self, request_id: str, language: str, message: str
    ) -> dict[str, Any]:
        if language == "ar":
            message = f"تعذر إكمال الطلب: {message}"

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "error",
            "result": None,
            "error_message": message,
        }
