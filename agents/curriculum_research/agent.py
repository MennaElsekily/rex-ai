
from typing import Any, Optional

from agents.conversation_store import ConversationStore
from agents.curriculum_research.curriculum_builder import CurriculumBuilder
from agents.curriculum_research.data import extract_search_terms
from agents.curriculum_research.refinement import (
    MAX_TRAINING_HOURS,
    RefinementError,
    apply_refinements,
)
from agents.curriculum_research.request_parser import (
    CREATION_PHRASES,
    Refinement,
    detect_refinements,
    has_creation_intent,
    is_accreditation_request,
    is_unsupported_action,
    looks_like_edit_attempt,
    parse_request,
    refinement_takes_priority,
)
from agents.curriculum_research.tools import CurriculumTools
from agents.text_utils import normalize_for_matching


def _text(language: str, english: str, arabic: str) -> str:
    return arabic if language == "ar" else english


_REAL_CHANGE_TYPES = {
    "hours_changed", "module_added", "capstone_added", "audience_changed",
}


class CurriculumResearchAgent:
    """
    Researches curriculum examples, drafts new curricula, and refines the
    latest draft of a conversation.

    Conversation state: when a ConversationStore is supplied, the most
    recent generated draft is kept under (conversation_id,
    "curriculum_research") so follow-ups such as "make it 40 hours" or
    "add a capstone" can refine it. Without a store, or without a
    conversation_id in the request context, the agent is stateless.
    """

    AGENT_NAME = "curriculum_research"
    CREATION_PHRASES = CREATION_PHRASES

    DEFAULT_TRAINING_HOURS = 30
    MIN_TRAINING_HOURS = CurriculumBuilder.MODULE_COUNT
    MAX_TRAINING_HOURS = MAX_TRAINING_HOURS

    def __init__(
        self,
        tools: CurriculumTools | None = None,
        builder: CurriculumBuilder | None = None,
        conversation_store: ConversationStore | None = None,
    ):
        self.tools = tools or CurriculumTools()
        self.builder = builder or CurriculumBuilder()
        self.conversation_store = conversation_store

    # ----------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------

    def process_request(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Process curriculum search, generation and refinement requests."""

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
            # The conversation's latest draft, and any edit of it. A
            # message is only an edit when it starts like one ("add a
            # capstone", "make it 40 hours", or just "40 hours").
            stored = self._load_state(context)
            refinements = (
                detect_refinements(user_input) if stored is not None else []
            )
            refine_first = bool(refinements) and refinement_takes_priority(
                user_input
            )

            # 1. Explicit requests to create a new curriculum/course.
            #    ("add a module on how to design a course" is still an
            #    edit of the draft, not a new curriculum.)
            if self._is_creation_request(user_input) and not refine_first:
                return self._handle_creation(
                    request_id, language, user_input, context
                )

            # 2. Accreditation / gap analysis is not implemented. (A
            #    module that merely has such a word in its title is fine.)
            if is_accreditation_request(user_input) and not (
                refine_first
                and self._accreditation_only_in_titles(
                    user_input, refinements
                )
            ):
                return self._unsupported_response(
                    request_id,
                    language,
                    capability="accreditation_gap_analysis",
                    message=_text(
                        language,
                        "Accreditation alignment and gap analysis are not "
                        "supported yet, and this agent has no verified "
                        "accreditation standards to compare against. I "
                        "can search the sample curriculum examples, draft "
                        "a new curriculum (for example: \"Create a "
                        "curriculum about data analysis\"), and refine "
                        "that draft.",
                        "مواءمة الاعتماد وتحليل الفجوات غير مدعومَين حاليًا، "
                        "وليست لدى هذا الوكيل معايير اعتماد موثّقة للمقارنة "
                        "بها. يمكنني البحث في أمثلة المناهج المتاحة، وإعداد "
                        "مسودة منهج جديد (مثل: «أنشئ منهج عن تحليل "
                        "البيانات»)، وتعديل هذه المسودة.",
                    ),
                )

            # 3. Things this agent cannot do (translate, export, ...):
            #    say so instead of searching for them.
            if is_unsupported_action(user_input):
                return self._unsupported_edit(
                    request_id, language, capability="unsupported_action"
                )

            # 4. Refine the conversation's latest draft, if there is one.
            if stored is not None:
                response = self._handle_refinement(
                    request_id, language, user_input, context, stored,
                    refinements,
                )
                if response is not None:
                    return response

            # 5. Otherwise search the existing examples.
            return self._handle_search(request_id, language, user_input)

        except Exception as exc:
            return self._error_response(
                request_id, language, str(exc)
            )

    def has_refinable_draft(self, request: dict[str, Any]) -> bool:
        """
        True when this conversation has a stored draft AND the message
        asks to change it. Used by the orchestrator so a follow-up like
        "make it 40 hours" reaches this agent even though it contains no
        curriculum keyword. Never true for unrelated messages.
        """

        context = request.get("context") or {}

        if self._load_state(context) is None:
            return False

        return bool(detect_refinements(str(request.get("user_input", ""))))

    # ----------------------------------------------------------------
    # Creation
    # ----------------------------------------------------------------

    def _handle_creation(
        self,
        request_id: str,
        language: str,
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        parsed = parse_request(user_input)

        if not parsed.topic:
            return self._error_response(
                request_id,
                language,
                _text(
                    language,
                    "Please tell me the topic, for example: "
                    "\"Create a curriculum about data analysis\".",
                    "من فضلك حدّد موضوع المنهج، مثال: "
                    "«أنشئ منهج عن تحليل البيانات».",
                ),
            )

        hours = parsed.training_hours
        if hours is None:
            raw_hours = context.get(
                "training_hours", self.DEFAULT_TRAINING_HOURS
            )
            try:
                hours = int(raw_hours)
            except (TypeError, ValueError):
                hours = None

        if (
            hours is None
            or hours < self.MIN_TRAINING_HOURS
            or hours > self.MAX_TRAINING_HOURS
        ):
            return self._error_response(
                request_id,
                language,
                _text(
                    language,
                    "Training hours must be a whole number between "
                    f"{self.MIN_TRAINING_HOURS} and "
                    f"{self.MAX_TRAINING_HOURS}.",
                    "يجب أن تكون ساعات التدريب عددًا صحيحًا بين "
                    f"{self.MIN_TRAINING_HOURS} و{self.MAX_TRAINING_HOURS}.",
                ),
            )

        audience = (
            parsed.target_audience
            or context.get("target_audience")
            or "Beginners"
        )

        draft = self.builder.build_curriculum(
            topic=parsed.topic,
            target_audience=audience,
            training_hours=hours,
            prerequisites=context.get("prerequisites"),
        )
        draft["revision"] = 1

        saved = self._save_state(context, draft)

        message = _text(
            language,
            "A new curriculum draft has been generated.",
            "تم إنشاء مسودة منهج تدريبي جديدة.",
        )

        details = []
        if parsed.training_hours is not None:
            details.append(_text(
                language,
                f"Duration: {hours} hours.",
                f"المدة: {hours} ساعة.",
            ))
        if parsed.target_audience:
            details.append(_text(
                language,
                f"Target audience: {audience}.",
                f"الفئة المستهدفة: {audience}.",
            ))
        if parsed.ignored_hours_rate:
            details.append(_text(
                language,
                "Hours per week or per day are not supported, so "
                f"{hours} total training hours were used.",
                "الساعات في الأسبوع أو اليوم غير مدعومة، لذا استُخدمت "
                f"{hours} ساعة تدريبية إجمالية.",
            ))
        if details:
            message += " " + " ".join(details)

        if saved:
            message += " " + _text(
                language,
                "You can refine this draft, for example: \"make it 40 "
                "hours\", \"add a capstone project\", or \"change the "
                "audience to teachers\".",
                "يمكنك تعديل المسودة، مثل: «اجعلها 40 ساعة» أو «أضف "
                "مشروعًا ختاميًا» أو «غيّر الفئة المستهدفة إلى المعلمين».",
            )

        notice = draft["notice"]
        if is_accreditation_request(user_input):
            notice += " " + _text(
                language,
                "Accreditation alignment was NOT performed: this draft "
                "has not been checked against any accreditation "
                "standards.",
                "لم تُجرَ مواءمة الاعتماد: لم تُراجَع هذه المسودة وفق أي "
                "معايير اعتماد.",
            )

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "success",
            "result": {
                "message": message,
                "curricula": [draft],
                "notice": notice,
            },
            "error_message": None,
        }

    # ----------------------------------------------------------------
    # Refinement
    # ----------------------------------------------------------------

    def _handle_refinement(
        self,
        request_id: str,
        language: str,
        user_input: str,
        context: dict[str, Any],
        stored: dict[str, Any],
        refinements: list[Refinement],
    ) -> Optional[dict[str, Any]]:
        if not refinements:
            if looks_like_edit_attempt(user_input):
                return self._unsupported_edit(request_id, language)
            return None

        if any(item.kind == "unsupported" for item in refinements):
            return self._unsupported_edit(request_id, language)

        try:
            updated, changes = apply_refinements(
                stored["draft"], refinements
            )
        except RefinementError as exc:
            return self._error_response(
                request_id,
                language,
                self._refinement_error_text(exc, language),
            )

        if any(change["type"] in _REAL_CHANGE_TYPES for change in changes):
            self._save_state(context, updated)

        summary = "; ".join(
            self._describe_change(change, language) for change in changes
        )
        message = _text(
            language,
            f"Updated your curriculum draft: {summary}.",
            f"تم تحديث مسودة المنهج: {summary}.",
        )

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "success",
            "result": {
                "message": message,
                "curricula": [updated],
                "notice": updated.get("notice", ""),
                "changes": changes,
            },
            "error_message": None,
        }

    @staticmethod
    def _describe_change(change: dict[str, Any], language: str) -> str:
        kind = change["type"]
        redistributed_en = "module hours were redistributed evenly"
        redistributed_ar = "وأُعيد توزيع ساعات الوحدات بالتساوي"

        if kind == "hours_changed":
            return _text(
                language,
                f"training hours changed from {change['old']} to "
                f"{change['new']} ({redistributed_en})",
                f"تم تغيير ساعات التدريب من {change['old']} إلى "
                f"{change['new']} ({redistributed_ar})",
            )
        if kind == "hours_unchanged":
            return _text(
                language,
                f"training hours were already {change['hours']}",
                f"ساعات التدريب كانت {change['hours']} بالفعل",
            )
        if kind == "module_added":
            return _text(
                language,
                f"added the module \"{change['title']}\" "
                f"({redistributed_en})",
                f"تمت إضافة الوحدة «{change['title']}» ({redistributed_ar})",
            )
        if kind == "module_exists":
            return _text(
                language,
                f"a module named \"{change['title']}\" already exists",
                f"توجد وحدة بعنوان «{change['title']}» بالفعل",
            )
        if kind == "capstone_added":
            return _text(
                language,
                f"added a dedicated Capstone Project module "
                f"({redistributed_en})",
                f"تمت إضافة وحدة المشروع الختامي (Capstone Project) "
                f"({redistributed_ar})",
            )
        if kind == "capstone_exists":
            return _text(
                language,
                "the draft already has a Capstone Project module",
                "يتضمن المنهج وحدة المشروع الختامي بالفعل",
            )
        if kind == "audience_changed":
            return _text(
                language,
                f"target audience changed from \"{change['old']}\" to "
                f"\"{change['new']}\"",
                f"تم تغيير الفئة المستهدفة من «{change['old']}» إلى "
                f"«{change['new']}»",
            )
        if kind == "audience_unchanged":
            return _text(
                language,
                f"the target audience is already \"{change['audience']}\"",
                f"الفئة المستهدفة هي «{change['audience']}» بالفعل",
            )

        return kind

    @staticmethod
    def _refinement_error_text(exc: RefinementError, language: str) -> str:
        if exc.code == "hours_too_low":
            minimum = exc.details["minimum"]
            return _text(
                language,
                f"The draft has {minimum} modules, so it needs at least "
                f"{minimum} training hours.",
                f"يحتوي المنهج على {minimum} وحدات، لذا يلزم ما لا يقل عن "
                f"{minimum} ساعات تدريبية.",
            )
        if exc.code == "hours_too_high":
            maximum = exc.details["maximum"]
            return _text(
                language,
                f"Training hours cannot exceed {maximum}.",
                f"لا يمكن أن تتجاوز ساعات التدريب {maximum} ساعة.",
            )
        if exc.code == "hours_too_low_for_module":
            minimum = exc.details["minimum"]
            return _text(
                language,
                f"Adding a module needs at least {minimum} training "
                "hours. Increase the hours first.",
                f"إضافة وحدة تتطلب {minimum} ساعات تدريبية على الأقل. "
                "زد عدد الساعات أولًا.",
            )

        return _text(
            language,
            "That change is not supported.",
            "هذا التعديل غير مدعوم.",
        )

    @staticmethod
    def _accreditation_only_in_titles(
        text: str, refinements: list[Refinement]
    ) -> bool:
        """True when accreditation words appear only inside module titles."""
        stripped = text

        for refinement in refinements:
            if refinement.kind == "add_module":
                stripped = stripped.replace(str(refinement.value), " ")

        return not is_accreditation_request(stripped)

    def _unsupported_edit(
        self,
        request_id: str,
        language: str,
        capability: str = "curriculum_edit",
    ) -> dict[str, Any]:
        return self._unsupported_response(
            request_id,
            language,
            capability=capability,
            message=_text(
                language,
                "I can't do that yet. I can search the sample curriculum "
                "examples, draft a new curriculum, and refine your draft: "
                "change the training hours (\"make it 40 hours\"), add a "
                "module (\"add a module on SQL\"), add a capstone "
                "project, or change the target audience. Removing "
                "content, translating, exporting and comparing are not "
                "supported yet.",
                "لا يمكنني تنفيذ ذلك حاليًا. يمكنني البحث في أمثلة المناهج "
                "المتاحة، وإعداد مسودة منهج جديد، وتعديل مسودتك: تغيير "
                "ساعات التدريب («اجعلها 40 ساعة»)، أو إضافة وحدة («أضف "
                "وحدة عن SQL»)، أو إضافة مشروع ختامي، أو تغيير الفئة "
                "المستهدفة. حذف المحتوى والترجمة والتصدير والمقارنة غير "
                "مدعومة حاليًا.",
            ),
        )

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def _handle_search(
        self, request_id: str, language: str, user_input: str
    ) -> dict[str, Any]:
        # An hours phrase is never a search topic. An audience phrase is
        # dropped too, unless it is all the user gave us ("curriculum for
        # teachers"), in which case it is the search.
        parsed = parse_request(user_input)
        concepts, words = extract_search_terms(
            normalize_for_matching(parsed.remaining_text)
        )
        query = (
            parsed.remaining_text
            if (concepts or words)
            else parsed.hours_free_text
        ) or user_input

        result = self.tools.search_examples(query)

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

        if result.get("match_type") == "partial":
            message = _text(
                language,
                "No example matched every topic in your request. These "
                "are the closest examples.",
                "لم يطابق أي مثال جميع المواضيع في طلبك. هذه أقرب "
                "الأمثلة.",
            )
        else:
            message = _text(
                language,
                "Relevant curriculum examples were found.",
                "تم العثور على أمثلة للمناهج ذات الصلة.",
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

    # ----------------------------------------------------------------
    # Conversation state (shared ConversationStore, our own agent key)
    # ----------------------------------------------------------------

    @staticmethod
    def _conversation_id(context: dict[str, Any]) -> Optional[str]:
        conversation_id = context.get("conversation_id")

        if isinstance(conversation_id, str) and conversation_id.strip():
            return conversation_id

        return None

    def _load_state(
        self, context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        conversation_id = self._conversation_id(context)

        if self.conversation_store is None or conversation_id is None:
            return None

        try:
            state = self.conversation_store.get(
                conversation_id, self.AGENT_NAME
            )
        except Exception:
            return None

        if isinstance(state, dict) and isinstance(state.get("draft"), dict):
            return state

        return None

    def _save_state(
        self, context: dict[str, Any], draft: dict[str, Any]
    ) -> bool:
        conversation_id = self._conversation_id(context)

        if self.conversation_store is None or conversation_id is None:
            return False

        try:
            self.conversation_store.set(
                conversation_id, self.AGENT_NAME, {"draft": draft}
            )
        except Exception:
            # Failing to remember the draft must not fail the request.
            return False

        return True

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _is_creation_request(text: str) -> bool:
        """Identify explicit curriculum/course-generation requests."""

        return has_creation_intent(text)

    @staticmethod
    def _detect_language(text: str) -> str:
        """Basic Arabic/English detection."""

        if any("؀" <= char <= "ۿ" for char in text):
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
                    "؀" <= char <= "ۿ"
                    for char in cleaned
                )
            ):
                keywords.append(cleaned)

        return list(dict.fromkeys(keywords))

    def _unsupported_response(
        self,
        request_id: str,
        language: str,
        capability: str,
        message: str,
    ) -> dict[str, Any]:
        """A clear 'not supported' answer -- never a fake success."""

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "error",
            "result": {
                "code": "unsupported_request",
                "capability": capability,
            },
            "error_message": message,
        }

    def _error_response(
        self, request_id: str, language: str, message: str
    ) -> dict[str, Any]:
        if language == "ar" and not any(
            "؀" <= char <= "ۿ" for char in message
        ):
            message = f"تعذر إكمال الطلب: {message}"

        return {
            "request_id": request_id,
            "agent_name": self.AGENT_NAME,
            "status": "error",
            "result": None,
            "error_message": message,
        }
