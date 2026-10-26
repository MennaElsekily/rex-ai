
import uuid

import streamlit as st

from agents.register_agents import create_orchestrator


st.set_page_config(
    page_title="REX AI — Agent Integration Test",
    page_icon="🎓",
)

st.title("🎓 REX AI — Agent Integration Test")
st.caption(
    "Testing the Streamlit interface, orchestrator, and Agent 3."
)


@st.cache_resource
def get_orchestrator():
    """Create and reuse the registered orchestrator."""
    return create_orchestrator()


orchestrator = get_orchestrator()

# One conversation_id per browser session, so REX Assistant's
# conversation memory (profile, consent, language) persists across
# messages within this session but never leaks into another session.
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())


if "test_messages" not in st.session_state:
    st.session_state.test_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask me to find a curriculum example, generate a "
                "new curriculum draft, recommend a course, or search "
                "for education tenders in Egypt or Saudi Arabia."
            ),
        }
    ]


def format_curriculum(curriculum: dict) -> str:
    """Format all curriculum sections for persistent chat history."""

    lines = [
        f"## {curriculum.get('title', 'Untitled Curriculum')}",
        f"**Status:** {curriculum.get('status', 'example')}",
        "",
        "### Course objectives",
    ]

    objectives = curriculum.get("course_objectives", [])
    lines.extend(
        [f"- {item}" for item in objectives]
        if objectives
        else ["Not specified."]
    )

    lines.extend([
        "",
        f"**Target audience:** "
        f"{curriculum.get('target_audience', 'Not specified')}",
        "",
        "### Prerequisites",
    ])

    prerequisites = curriculum.get("prerequisites", [])
    lines.extend(
        [f"- {item}" for item in prerequisites]
        if prerequisites
        else ["Not specified."]
    )

    lines.extend(["", "### Learning outcomes"])

    outcomes = curriculum.get("learning_outcomes", [])
    lines.extend(
        [f"- {item}" for item in outcomes]
        if outcomes
        else ["Not specified."]
    )

    lines.extend([
        "",
        f"### Modules and topics — "
        f"{curriculum.get('training_hours', 'Not specified')} total hours",
    ])

    modules = curriculum.get("modules", [])

    if modules:
        for module in modules:
            lines.extend([
                "",
                f"**Module {module.get('module_number', '')}: "
                f"{module.get('title', 'Untitled module')}** "
                f"({module.get('hours', 0)} hours)",
            ])

            topics = module.get("topics", [])
            if topics:
                lines.extend([f"- {topic}" for topic in topics])

            exercises = module.get("practical_exercises", [])
            if exercises:
                lines.append("- Practical exercises:")
                lines.extend([f"  - {item}" for item in exercises])
    else:
        lines.append("No module details provided.")

    lines.extend(["", "### Practical exercises"])

    exercises = curriculum.get("practical_exercises", [])
    lines.extend(
        [f"- {item}" for item in exercises]
        if exercises
        else ["Not specified."]
    )

    lines.extend(["", "### Assessments"])

    assessments = curriculum.get("assessments", [])
    if assessments:
        for assessment in assessments:
            if isinstance(assessment, dict):
                assessment_type = assessment.get("type", "Assessment")
                description = assessment.get("description", "")
                lines.append(
                    f"- **{assessment_type}:** {description}"
                )
            else:
                lines.append(f"- {assessment}")
    else:
        lines.append("Not specified.")

    lines.extend(["", "### Capstone project"])

    capstone = curriculum.get("capstone_project")

    if isinstance(capstone, dict):
        lines.append(
            f"**{capstone.get('title', 'Capstone project')}**"
        )
        lines.append(
            capstone.get("description", "Description not specified.")
        )
    elif capstone:
        lines.append(str(capstone))
    else:
        lines.append("Not specified.")

    if curriculum.get("description"):
        lines.extend([
            "",
            "### Description",
            curriculum["description"],
        ])

    if curriculum.get("notice"):
        lines.extend([
            "",
            f"> {curriculum['notice']}",
        ])

    return "\n".join(lines)


# Render the full conversation from stored messages.
for message in st.session_state.test_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input(
    "Ask to find or create a curriculum..."
)


if user_input:
    st.session_state.test_messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrator is processing your request..."):
            response = orchestrator.handle_request(
                user_input,
                context={
                    "conversation_id": st.session_state.conversation_id
                },
            )

            if response.get("status") != "success":
                answer = (
                    "I couldn't complete that request.\n\n"
                    f"{response.get('error_message') or 'Please try again.'}"
                )
                st.error(answer)

            else:
                results = response.get("result") or []

                successful_results = [
                    item
                    for item in results
                    if item.get("status") == "success"
                ]

                if not successful_results:
                    answer = (
                        "The orchestrator did not return a successful "
                        "result. Check agent registration and the "
                        "orchestrator response."
                    )
                    st.warning(answer)

                else:
                    answer_parts = []

                    for item in successful_results:
                        agent_name = item.get("agent_name")
                        agent_result = item.get("result") or {}

                        if agent_name == "curriculum_research":
                            message = agent_result.get(
                                "message",
                                "Curriculum request completed.",
                            )

                            curricula = agent_result.get("curricula", [])

                            if curricula:
                                formatted_curricula = [
                                    format_curriculum(curriculum)
                                    for curriculum in curricula
                                ]

                                message += "\n\n" + "\n\n---\n\n".join(
                                    formatted_curricula
                                )
                            else:
                                message += (
                                    "\n\nNo matching curriculum examples "
                                    "were found."
                                )

                            notice = agent_result.get("notice")
                            if notice and notice not in message:
                                message += f"\n\n> {notice}"

                            answer_parts.append(message)

                        elif agent_name == "rex_assistant":
                            answer_parts.append(
                                agent_result.get(
                                    "answer", "No answer returned."
                                )
                            )

                        elif agent_name == "rfp_research":
                            lines = [agent_result.get("message", "")]

                            # Show exactly which API(s) were used and
                            # whether each actually succeeded -- an API
                            # error/unavailability must never look like
                            # "zero tenders found."
                            api_status_labels = {
                                "success": "✅ Success",
                                "unavailable": "⚠️ No API available",
                                "awaiting_credentials": "🔑 Awaiting credentials",
                                "error": "❌ Request failed",
                            }
                            api_results = agent_result.get("api_results", [])
                            if api_results:
                                lines.append("\n**API sources used:**")
                                for api_result in api_results:
                                    status = api_result.get("status")
                                    lines.append(
                                        f"- {api_result.get('source_name')}: "
                                        f"{api_status_labels.get(status, status)} "
                                        f"({api_result.get('raw_record_count', 0)} raw record(s)) "
                                        f"— {api_result.get('message', '')}"
                                    )
                                lines.append("")

                            # review_status describes REVIEW PRIORITY
                            # only (Egypt vs Saudi business rule) -- it
                            # never implies a tender is verified open
                            # or that the academy qualifies.
                            review_labels = {
                                "flagged_for_immediate_review": (
                                    "⚑ Flagged for immediate review (Egypt)"
                                ),
                                "monitored_pipeline": (
                                    "Monitored pipeline (Saudi Arabia)"
                                ),
                            }
                            qualification_labels = {
                                "qualified": "✅ Qualified",
                                "not_qualified": "❌ Not qualified",
                                "unknown": "❓ Unknown (insufficient evidence)",
                            }

                            for opportunity in agent_result.get(
                                "opportunities", []
                            ):
                                source_url = opportunity.get("source_url", "")
                                title = opportunity.get("title") or opportunity.get("source_id", "")
                                if source_url:
                                    lines.append(f"- **[{title}]({source_url})**  ")
                                else:
                                    lines.append(f"- **{title}**  ")
                                lines.append(
                                    f"  ({opportunity.get('country')} — "
                                    f"{review_labels.get(opportunity.get('review_status'), opportunity.get('review_status'))})"
                                )
                                if opportunity.get("reference_number"):
                                    lines.append(
                                        f"  Reference: {opportunity['reference_number']}"
                                    )
                                if opportunity.get("organization"):
                                    lines.append(
                                        f"  Organization: {opportunity['organization']}"
                                    )
                                if opportunity.get("deadline"):
                                    lines.append(
                                        f"  Deadline: {opportunity['deadline']}"
                                    )
                                lines.append(
                                    f"  Source: {opportunity.get('source_name', 'unknown')} "
                                    f"(verification: {opportunity.get('verification_status', 'unknown')})"
                                )
                                lines.append(
                                    f"  {opportunity.get('ai_summary', '')}"
                                )
                                qualification_status = opportunity.get(
                                    "qualification_status", "unknown"
                                )
                                lines.append(
                                    f"  Qualification: "
                                    f"{qualification_labels.get(qualification_status, qualification_status)}"
                                    f" — {opportunity.get('qualification_reason', '')}"
                                )
                                missing = opportunity.get("missing_information")
                                if missing:
                                    lines.append(
                                        f"  Missing/unverified: {', '.join(missing)}"
                                    )

                            answer_parts.append("\n".join(lines))

                        else:
                            answer_parts.append(str(agent_result))

                    answer = "\n\n---\n\n".join(answer_parts)
                    st.markdown(answer)

    # Store the complete formatted response so it remains visible
    # when Streamlit reruns after the next user message.
    st.session_state.test_messages.append(
        {"role": "assistant", "content": answer}
    )
