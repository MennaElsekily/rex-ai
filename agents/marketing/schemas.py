"""
Channels, limits and the structured answers the model gives for Agent 5 (Marketing).

The model answers one small structured object per channel (``CHANNEL_SCHEMAS``) and one for the strategy. What is
stored for a campaign (see composer.py) is plain JSON built from those answers, so an administrator can edit any
text in it. The limits below are checked by validation.py, not by the schemas: an answer that is only too long is
asked for again with the problem named, instead of failing as unreadable output.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field

CHANNELS = ("social_captions", "ad_copy", "linkedin_posts", "email", "whatsapp", "headlines", "ctas")
TONES = ("professional", "friendly", "inspiring", "formal")
DEFAULT_TONE = "professional"
MIN_VARIATIONS, MAX_VARIATIONS, DEFAULT_VARIATIONS = 1, 3, 2

CHANNEL_LABELS = {
    "social_captions": {"en": "Social media captions", "ar": "تعليقات وسائل التواصل الاجتماعي"},
    "ad_copy": {"en": "Facebook / Instagram ad copy", "ar": "نصوص إعلانات فيسبوك وإنستغرام"},
    "linkedin_posts": {"en": "LinkedIn posts", "ar": "منشورات لينكدإن"},
    "email": {"en": "Email marketing", "ar": "البريد التسويقي"},
    "whatsapp": {"en": "WhatsApp promotional messages", "ar": "رسائل واتساب الترويجية"},
    "headlines": {"en": "Campaign headlines", "ar": "عناوين الحملة"},
    "ctas": {"en": "Call-to-action suggestions", "ar": "اقتراحات الدعوة إلى اتخاذ إجراء"},
}
TONE_LABELS = {
    "professional": {"en": "Professional", "ar": "مهني"},
    "friendly": {"en": "Friendly", "ar": "ودّي"},
    "inspiring": {"en": "Inspiring", "ar": "ملهم"},
    "formal": {"en": "Formal", "ar": "رسمي"},
}

# (minimum, maximum) characters of each text field of a variation. The maximums follow what each platform shows well.
LIMITS: dict[str, dict[str, tuple[int, int]]] = {
    "social_captions": {"text": (30, 280)},
    "ad_copy": {"headline": (8, 60), "primary_text": (40, 300), "cta": (3, 30)},
    "linkedin_posts": {"text": (80, 900)},
    "email": {"subject": (8, 70), "preheader": (10, 100), "body": (80, 1200), "cta": (3, 40)},
    "whatsapp": {"message": (20, 450)},
    "headlines": {"text": (8, 70)},
    "ctas": {"text": (3, 60)},
}
HASHTAG_LIMIT = 5
MAX_CTA_WORDS = 8      # a call-to-action suggestion: at most 8 words AND at most LIMITS["ctas"]["text"] characters
# The field of a variation that is compared to the other variations to see that they differ.
MAIN_FIELD = {"social_captions": "text", "ad_copy": "primary_text", "linkedin_posts": "text", "email": "body",
              "whatsapp": "message", "headlines": "text", "ctas": "text"}

Text = Annotated[str, Field(max_length=1500)]
Short = Annotated[str, Field(max_length=300)]


class Caption(BaseModel):
    text: Text = Field(description="The caption, 30-280 characters.")
    hashtags: list[Short] = Field(default_factory=list, max_length=8, description="Up to 5 short hashtags, no spaces.")


class AdVariation(BaseModel):
    headline: Short = Field(description="Ad headline, at most 60 characters.")
    primary_text: Text = Field(description="Main ad text, at most 300 characters.")
    cta: Short = Field(description="Button text, at most 30 characters.")


class LinkedInPost(BaseModel):
    text: Text = Field(description="A professional post of 80-900 characters.")


class EmailVariation(BaseModel):
    subject: Short = Field(description="Subject line, at most 70 characters.")
    preheader: Short = Field(description="Preview text, at most 100 characters.")
    body: Text = Field(description="Email body, 80-1200 characters, short paragraphs.")
    cta: Short = Field(description="Call-to-action button text, at most 40 characters.")


class WhatsAppMessage(BaseModel):
    message: Text = Field(description="A short, friendly message of at most 450 characters.")


class Headline(BaseModel):
    text: Short = Field(description="A campaign headline, at most 70 characters.")


class CallToAction(BaseModel):
    text: Short = Field(description="A short imperative action phrase starting with a verb, such as \"Explore the course\": at most 8 words (and at most 60 characters); aim for 3 to 6 words. Not a sentence.")


class SocialCaptionSet(BaseModel):
    variations: list[Caption] = Field(min_length=1, max_length=MAX_VARIATIONS)


class AdCopySet(BaseModel):
    variations: list[AdVariation] = Field(min_length=1, max_length=MAX_VARIATIONS)


class LinkedInSet(BaseModel):
    variations: list[LinkedInPost] = Field(min_length=1, max_length=MAX_VARIATIONS)


class EmailSet(BaseModel):
    variations: list[EmailVariation] = Field(min_length=1, max_length=MAX_VARIATIONS)


class WhatsAppSet(BaseModel):
    variations: list[WhatsAppMessage] = Field(min_length=1, max_length=MAX_VARIATIONS)


class HeadlineSet(BaseModel):
    variations: list[Headline] = Field(min_length=1, max_length=MAX_VARIATIONS)


class CtaSet(BaseModel):
    variations: list[CallToAction] = Field(min_length=1, max_length=MAX_VARIATIONS)


CHANNEL_SCHEMAS: dict[str, type[BaseModel]] = {
    "social_captions": SocialCaptionSet, "ad_copy": AdCopySet, "linkedin_posts": LinkedInSet, "email": EmailSet,
    "whatsapp": WhatsAppSet, "headlines": HeadlineSet, "ctas": CtaSet,
}


class CampaignStrategy(BaseModel):
    objective: Short = Field(description="The campaign objective, one sentence (for example, to attract learners to enrol).")
    main_message: Short = Field(description="The main marketing message, one or two sentences, from the course facts only.")
    content_sequence: list[Short] = Field(min_length=3, max_length=6, description="Three to six short steps: what to post first, next, last.")
    call_to_action: Short = Field(description="The main call to action, at most 8 words.")


def variation_fields(channel: str) -> tuple[str, ...]:
    """The text fields of one variation of a channel, in display order."""
    return tuple(LIMITS[channel])


def to_plain(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()
