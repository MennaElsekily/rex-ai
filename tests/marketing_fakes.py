"""
Stand-ins and helpers for Agent 5's tests: the real toolbox on temporary storage, canned model answers that pass every
check (English and Arabic, any number of variations), and helpers that save and approve a curriculum the way the
Curriculum Agent and an administrator do (never writing the approved status directly).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.marketing.marketing_store import MarketingStore
from agents.marketing.marketing_tools import MarketingToolbox
from tests.curriculum_fakes import FakeLLM
from tests.curriculum_fixtures import valid_curriculum

ADMIN = "admin-session-test0001"


def in_process_marketing_tools(root: Path) -> MarketingToolbox:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return MarketingToolbox(store=MarketingStore(root / "marketing.sqlite3"),
                            curricula=CurriculumStore(root / "curricula.sqlite3"))


def save_curriculum(toolbox: MarketingToolbox, content: dict | None = None, valid: bool = True,
                    curriculum_id: str | None = None) -> dict:
    """Save as the Curriculum Agent does: pending_review (valid) or draft."""
    return toolbox.curricula.save_version(content or valid_curriculum(), {"valid": valid, "errors": []}, [],
                                          curriculum_id=curriculum_id)


def approved_course(toolbox: MarketingToolbox, content: dict | None = None) -> str:
    saved = save_curriculum(toolbox, content)
    toolbox.curricula.approve(saved["curriculum_id"], saved["version"], ADMIN)
    return saved["curriculum_id"]


# ---------------------------------------------------------------------------- canned model answers

TEXTS: dict[str, dict[str, list[Any]]] = {
    "en": {
        "social_captions": [
            {"text": "Learn to prepare a dataset for modelling in Machine Learning Foundations, a beginner course built on practice.",
             "hashtags": ["MachineLearning", "Python"]},
            {"text": "Curious about machine learning? Build simple models and evaluate a classifier, one module at a time.",
             "hashtags": ["DataSkills"]},
            {"text": "New graduates and analysts: begin your machine learning journey with Python and hands-on exercises.",
             "hashtags": ["Analytics", "Learning"]}],
        "ad_copy": [
            {"headline": "Start machine learning with Python", "primary_text": "A practical first course for new graduates and analysts: prepare data, then evaluate a simple model.", "cta": "Learn more"},
            {"headline": "Build your first classifier", "primary_text": "Two modules, hands-on labs and exercises, and a capstone where you present an end-to-end model.", "cta": "See the modules"},
            {"headline": "Data skills that employers look for", "primary_text": "Practise cleaning data and judging model quality in a course designed for beginners.", "cta": "Join the course"}],
        "linkedin_posts": [
            {"text": "Machine learning starts with good data. Machine Learning Foundations walks new graduates and analysts through preparing a dataset for modelling and evaluating a simple classification model, using Python throughout."},
            {"text": "Employers increasingly expect data and modelling skills. This beginner course covers 40 hours of practice: cleaning data, training a classifier, and presenting an end-to-end model as a capstone project."},
            {"text": "What will you be able to do after the course? Prepare data for modelling, evaluate a classifier, and explain your results. See whether it fits your team and share it with someone who is starting out."}],
        "email": [
            {"subject": "Your first machine learning course", "preheader": "Practical, beginner friendly and built on Python.", "body": "Hello,\n\nMachine Learning Foundations is a practical first course for new graduates and analysts. You will prepare a dataset for modelling and evaluate a simple classification model.\n\nRead the modules and see if it fits you.", "cta": "View the course"},
            {"subject": "From raw data to a working model", "preheader": "Two modules, hands-on exercises and a capstone.", "body": "Hello,\n\nIn this beginner course you clean a dataset in the first module, then train and evaluate a classifier in the second. The capstone project brings both together in a notebook and a presentation.\n\nTake a look at the full outline.", "cta": "See the outline"},
            {"subject": "Is machine learning for you?", "preheader": "Find out what you will practise in this course.", "body": "Hello,\n\nIf you are an analyst or a recent graduate and want to understand modelling, this course starts from the basics and keeps every session practical.\n\nRead what you will be able to do.", "cta": "Read more"}],
        "whatsapp": [
            {"message": "Hello! Machine Learning Foundations is a beginner course for new graduates and analysts. Want to see the modules?"},
            {"message": "Hi, if you want to practise preparing data and evaluating a simple model with Python, this course may suit you. Shall I send the outline?"},
            {"message": "Good day. Looking to start with machine learning? Our beginner course is hands-on. Reply to hear what you will learn."}],
        "headlines": [
            {"text": "Machine learning, step by step"}, {"text": "Learn to prepare data and judge a model"},
            {"text": "Your first practical course in modelling"}],
        "ctas": [{"text": "Explore the course"}, {"text": "Join the first module"}, {"text": "Ask about the outline"}],
    },
    "ar": {
        "social_captions": [
            {"text": "تعلّم كيف تجهّز مجموعة بيانات للنمذجة في دورة أساسيات تعلم الآلة للمبتدئين بأسلوب عملي.", "hashtags": ["تعلم_الآلة", "Python"]},
            {"text": "هل تريد بناء نماذج بسيطة وتقييم نموذج تصنيف؟ ابدأ خطوة بعد خطوة مع وحدات الدورة.", "hashtags": ["مهارات_البيانات"]},
            {"text": "للخريجين الجدد والمهتمين بالبيانات: ابدأ رحلتك مع تعلم الآلة باستخدام بايثون وتمارين تطبيقية.", "hashtags": ["تحليل_البيانات"]}],
        "ad_copy": [
            {"headline": "ابدأ تعلم الآلة مع بايثون", "primary_text": "دورة عملية للمبتدئين تتعلم فيها تجهيز البيانات ثم تقييم نموذج بسيط.", "cta": "اعرف المزيد"},
            {"headline": "ابنِ أول نموذج تصنيف", "primary_text": "وحدتان تطبيقيتان ومشروع ختامي تعرض فيه نموذجًا كاملًا ونتائجه.", "cta": "تصفح الوحدات"},
            {"headline": "مهارات بيانات يحتاجها سوق العمل", "primary_text": "تدرّب على تنظيف البيانات وتقدير جودة النموذج في دورة مصممة للمبتدئين.", "cta": "انضم إلى الدورة"}],
        "linkedin_posts": [
            {"text": "يبدأ تعلم الآلة ببيانات جيدة. تقدم دورة أساسيات تعلم الآلة للخريجين الجدد والمهتمين بالبيانات خطوات تجهيز مجموعة بيانات للنمذجة وتقييم نموذج تصنيف بسيط باستخدام بايثون."},
            {"text": "يحتاج سوق العمل إلى مهارات تحليل البيانات وبناء النماذج. تغطي هذه الدورة 40 ساعة من التدريب العملي: تنظيف البيانات وتدريب نموذج تصنيف وعرض نموذج كامل في مشروع ختامي."},
            {"text": "ماذا ستستطيع أن تفعل بعد الدورة؟ تجهيز البيانات للنمذجة وتقييم أداء نموذج وشرح نتائجك. اطلع على المحتوى وشاركه مع زميل يبدأ طريقه في هذا المجال."}],
        "email": [
            {"subject": "دورتك الأولى في تعلم الآلة", "preheader": "دورة عملية للمبتدئين مبنية على بايثون.", "body": "مرحبًا،\n\nأساسيات تعلم الآلة دورة عملية للخريجين الجدد والمهتمين بالبيانات. ستجهّز مجموعة بيانات للنمذجة وتقيّم نموذج تصنيف بسيطًا.\n\nاطلع على الوحدات وانظر إن كانت تناسبك.", "cta": "اطلع على الدورة"},
            {"subject": "من بيانات خام إلى نموذج يعمل", "preheader": "وحدتان وتمارين تطبيقية ومشروع ختامي.", "body": "مرحبًا،\n\nفي هذه الدورة تنظف مجموعة بيانات في الوحدة الأولى ثم تدرب نموذج تصنيف وتقيّمه في الوحدة الثانية. يجمع المشروع الختامي بين الجزأين في دفتر ملاحظات وعرض تقديمي.\n\nتصفح المخطط الكامل.", "cta": "شاهد المخطط"},
            {"subject": "هل تناسبك دراسة تعلم الآلة؟", "preheader": "اعرف ما ستتدرب عليه في الدورة.", "body": "مرحبًا،\n\nإذا كنت محللًا أو خريجًا حديثًا وتريد فهم النمذجة فهذه الدورة تبدأ من الأساسيات وتبقي كل جلسة تطبيقية.\n\nاقرأ ما ستكون قادرًا على فعله.", "cta": "اقرأ المزيد"}],
        "whatsapp": [
            {"message": "مرحبًا! دورة أساسيات تعلم الآلة للمبتدئين من الخريجين الجدد والمهتمين بالبيانات. هل تريد رؤية الوحدات؟"},
            {"message": "أهلًا، إذا أردت التدرب على تجهيز البيانات وتقييم نموذج بسيط باستخدام بايثون فقد تناسبك هذه الدورة. هل أرسل لك المخطط؟"},
            {"message": "تحية طيبة. هل تبدأ رحلتك مع تعلم الآلة؟ دورتنا للمبتدئين تطبيقية. أجب لتعرف ما ستتعلمه."}],
        "headlines": [{"text": "تعلم الآلة خطوة بخطوة"}, {"text": "تعلم تجهيز البيانات وتقييم النموذج"}, {"text": "دورتك العملية الأولى في النمذجة"}],
        "ctas": [{"text": "اكتشف الدورة"}, {"text": "ابدأ بالوحدة الأولى"}, {"text": "اسأل عن المخطط"}],
    },
}

STRATEGY = {
    "en": {"objective": "Attract new graduates and analysts to the beginner course", "main_message":
           "Learn machine learning by practising data preparation and evaluating simple models.",
           "recommended_channels": ["linkedin_posts", "email", "social_captions"],
           "content_sequence": ["Introduce the course and who it is for", "Share what learners will practise", "Invite readers to take the next step"],
           "call_to_action": "Join the course"},
    "ar": {"objective": "جذب الخريجين الجدد والمهتمين بالبيانات إلى دورة المبتدئين", "main_message":
           "تعلّم تعلم الآلة بالتدرب على تجهيز البيانات وتقييم نماذج بسيطة.",
           "recommended_channels": ["linkedin_posts", "email", "social_captions"],
           "content_sequence": ["قدّم الدورة ولمن هي", "اعرض ما سيتدرب عليه المتعلمون", "ادعُ القراء إلى الخطوة التالية"],
           "call_to_action": "انضم إلى الدورة"},
}

SCHEMA_OF = {"SocialCaptionSet": "social_captions", "AdCopySet": "ad_copy", "LinkedInSet": "linkedin_posts", "EmailSet": "email",
             "WhatsAppSet": "whatsapp", "HeadlineSet": "headlines", "CtaSet": "ctas"}


def requested_count(prompt: str) -> int:
    match = re.search(r"Write exactly (\d+) variation", prompt)
    assert match, "no variation count in the prompt"
    return int(match.group(1))


def channel_answer(channel: str, language: str = "en"):
    """A callable FakeLLM answer: the channel's canned variations, as many as the prompt asks for."""
    def answer(prompt: str) -> dict:
        return {"variations": TEXTS[language][channel][:requested_count(prompt)]}
    return answer


def answers(language: str = "en", **overrides: Any) -> dict[str, Any]:
    """FakeLLM answers for the strategy and every channel schema (20 calls each), with overrides by schema name."""
    base: dict[str, Any] = {"CampaignStrategy": STRATEGY[language]}
    base.update({schema: channel_answer(channel, language) for schema, channel in SCHEMA_OF.items()})
    base.update(overrides)
    return base


def fake_llm(language: str = "en", **overrides: Any) -> FakeLLM:
    return FakeLLM({name: value if isinstance(value, list) else [value] * 20 for name, value in answers(language, **overrides).items()})


def make_campaign(toolbox: MarketingToolbox, curriculum_id: str, language: str = "en", llm: Any = None, **request: Any) -> dict:
    """Run the real agent (fake model) for the Marketing tab's form choices; returns the agent's response envelope."""
    from agents.marketing.marketing_agent import MarketingAgent

    agent = MarketingAgent(tools=toolbox, llm=llm or fake_llm(language))
    return agent.process_request({"request_id": "r", "user_input": "Create a marketing campaign", "context": {
        "language": language, "conversation_id": "c", "marketing_request": {"curriculum_id": curriculum_id, **request}}})
