
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
                "new curriculum draft, or recommend a course."
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

                        else:
                            answer_parts.append(str(agent_result))

                    answer = "\n\n---\n\n".join(answer_parts)
                    st.markdown(answer)

    # Store the complete formatted response so it remains visible
    # when Streamlit reruns after the next user message.
    st.session_state.test_messages.append(
        {"role": "assistant", "content": answer}
    )
