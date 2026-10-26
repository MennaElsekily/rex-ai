"""
LangChain component used by the RFP Research graph (graph.py):

- `build_assessment_chain`: a LangChain prompt template + structured
  output chain (ChatOllama) that judges whether an already-normalized
  API tender record (agents/rfp_research/api_clients/base.py's
  NormalizedTender) genuinely relates to the academy's services, and
  assesses qualification as a 3-state judgment (qualified /
  not_qualified / unknown), grounded only in what the academy's
  capabilities are known to be and what the API record actually
  states -- never a guess.

Unlike the project's earlier web-scraping version of this agent, no
page text is fetched or parsed here -- the API client layer
(agents/rfp_research/api_clients/) already returns structured fields,
so this chain only needs to judge relevance and qualification, not
extract raw fields from HTML.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# What this project actually knows about the academy's own
# capabilities (from the brief and the project's own course-related
# data). No registrations, certifications, or geographic licensing
# status are known, so the assessment prompt below is instructed to
# treat anything requiring those as "unknown," not "qualified."
ACADEMY_CAPABILITIES_SUMMARY = (
    "The academy designs and delivers technical and professional "
    "training courses and curricula (e.g. software, data, digital "
    "skills). No information is available about the academy's formal "
    "business registrations, certifications, financial capacity, or "
    "geographic licensing in Egypt or Saudi Arabia specifically."
)

RELEVANT_CATEGORY_DESCRIPTION = (
    "education and training, technical and IT training, software and "
    "digital skills, professional development, corporate learning, "
    "or digital transformation training"
)


class TenderAssessment(BaseModel):
    """Structured output for one already-normalized API tender record."""

    is_relevant: bool = Field(
        description=(
            f"Whether this tender's actual scope of work genuinely "
            f"relates to {RELEVANT_CATEGORY_DESCRIPTION} for an "
            "academy -- not merely tangential or unrelated to these "
            "services, even if the source API's own category label "
            "mentions a related word."
        )
    )
    summary: str = Field(
        description="A short, one-to-two sentence factual summary of the opportunity."
    )
    qualification_status: Literal["qualified", "not_qualified", "unknown"] = Field(
        description=(
            "'qualified' only if the stated eligibility requirements "
            "are fully consistent with the academy's known "
            "capabilities. 'not_qualified' only if a stated "
            "requirement is clearly NOT met. 'unknown' whenever "
            "eligibility is not stated, or requires information about "
            "the academy (registrations, certifications, geographic "
            "licensing) that is not available. Never default to "
            "'qualified' just because the topic is education/training, "
            "and never default to 'not_qualified' merely because "
            "eligibility was not stated -- that case is 'unknown'."
        )
    )
    qualification_reason: str = Field(
        description="A brief, evidence-based reason for the qualification_status."
    )


@lru_cache(maxsize=2)
def _get_llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, temperature=0)


def build_assessment_chain(model: str = "qwen2.5:3b") -> Runnable:
    """
    Build a LangChain prompt-template -> structured-output chain that
    judges relevance and assesses qualification for one normalized API
    tender record.
    """
    llm = _get_llm(model)
    structured_llm = llm.with_structured_output(TenderAssessment)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You assess real tender/RFP records retrieved from a "
            "government procurement API, for an academy deciding "
            "whether an opportunity is relevant and pursuable. Use "
            "ONLY the information actually present in the supplied "
            "record. Never invent details not stated. "
            f"Academy profile: {ACADEMY_CAPABILITIES_SUMMARY}",
        ),
        (
            "user",
            "Title: {title}\n"
            "Organization: {organization}\n"
            "Country: {country}\n"
            "Category/notice type: {category}\n"
            "Scope of work: {scope_of_work}\n"
            "Eligibility requirements: {eligibility}\n"
            "Deadline: {deadline}",
        ),
    ])

    return prompt | structured_llm
