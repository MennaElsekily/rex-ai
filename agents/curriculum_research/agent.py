
from typing import Any

from agents.curriculum_research.tools import CurriculumTools
from agents.curriculum_research.curriculum_builder import CurriculumBuilder


class CurriculumResearchAgent:
    """Researches curriculum examples and supports curriculum planning."""

    AGENT_NAME = "curriculum_research"

    def __init__(
        self,
        tools: CurriculumTools | None = None,
        builder: CurriculumBuilder | None = None,
    ):
        self.tools = tools or CurriculumTools()
        self.builder = builder or CurriculumBuilder()

    def process_request(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Process curriculum search and generation requests."""

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

        try:
            # Route requests that explicitly ask to create a curriculum.
            if self._is_creation_request(user_input):
                curriculum = self._build_from_request(
                    user_input, context
                )

                message = (
                    "تم إنشاء مسودة منهج تدريبي جديدة."
                    if language == "ar"
                    else "A new curriculum draft has been generated."
                )

                return {
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

            # Otherwise, search the existing curriculum examples.
            result = self.tools.search_examples(user_input)

            if result["status"] != "success":
                return self._error_response(
                    request_id,
                    language,
                    result.get(
                        "error_message",
                        "Curriculum search failed.",
                    ),
                )

            # Try specific phrases if the complete request has no match.
            if result["count"] == 0:
                for keyword in self._extract_keywords(user_input):
                    if len(keyword.split()) < 2:
                        continue

                    result = self.tools.search_examples(keyword)

                    if result["status"] != "success":
                        return self._error_response(
                            request_id,
                            language,
                            result.get(
                                "error_message",
                                "Curriculum search failed.",
                            ),
                        )

                    if result["count"] > 0:
                        break

            if result["count"] == 0:
                message = (
                    "لم يتم العثور على أمثلة مطابقة في البيانات المتاحة."
                    if language == "ar"
                    else "No matching examples were found in the available data."
                )

                return {
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

            message = (
                "تم العثور على أمثلة للمناهج ذات الصلة."
                if language == "ar"
                else "Relevant curriculum examples were found."
            )

            return {
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
            return self._error_response(
                request_id, language, str(exc)
            )

    def _build_from_request(
        self,
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract basic generation settings and build a draft."""

        topic = self._extract_topic(user_input)

        target_audience = context.get(
            "target_audience", "Beginners"
        )

        training_hours = context.get("training_hours", 30)

        prerequisites = context.get("prerequisites")

        return self.builder.build_curriculum(
            topic=topic,
            target_audience=target_audience,
            training_hours=int(training_hours),
            prerequisites=prerequisites,
        )

    @staticmethod
    def _is_creation_request(text: str) -> bool:
        """Identify explicit curriculum-generation requests."""

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
        """Extract the topic from common curriculum requests."""

        normalized = text.strip()

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

        for prefix in sorted(prefixes, key=len, reverse=True):
            if normalized.casefold().startswith(prefix.casefold()):
                topic = normalized[len(prefix):].strip(" .،؟?!")
                if topic:
                    return topic

        # Safe fallback: retain the user's request as the topic.
        # More advanced topic and parameter extraction can be added later.
        return normalized

    @staticmethod
    def _detect_language(text: str) -> str:
        """Basic Arabic/English detection."""

        if any("\u0600" <= char <= "\u06FF" for char in text):
            return "ar"
        return "en"

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract useful English and Arabic search phrases."""

        normalized = text.casefold()
        keywords = []

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
