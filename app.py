
import json
from pathlib import Path

import ollama
import streamlit as st

st.set_page_config(
    page_title="REX Assistant",
    page_icon="🎓",
)

COURSES_PATH = Path(__file__).parent / "data" / "courses.json"

st.title("🎓 REX Assistant")
st.caption("Your personal educational and career guidance assistant.")

# Load the existing course knowledge base
with open(COURSES_PATH, "r", encoding="utf-8") as file:
    courses = json.load(file)

# Keep the conversation during the session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm REX, your learning assistant. 👋 "
                "I'll help you find courses that match your goals. "
                "What are you studying or working in, and what would "
                "you like to learn?"
            ),
        }
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Message REX Assistant...")

if user_input:
    # Display and save the student's message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    system_message = f"""
You are REX Assistant, an educational and career guidance chatbot.

Your goal is to understand the student's background, career goals,
current skill level, learning preferences, budget, and available
learning time through natural conversation.

Ask relevant follow-up questions when needed. Do not ask for
everything at once. Be friendly, clear, and concise.
When recommending courses, use only the course data below.

Before recommending a course:
1. Consider the student's current skill level and learning goals.
2. Check the course's skill level, target audience, and prerequisites.
3. Prefer beginner courses when the student is a beginner.
4. Do not present a course as suitable for the student if they lack its listed prerequisites. Explain what they need to learn first.
5. When a beginner lacks the prerequisites for an intermediate course, recommend a suitable beginner course first if one is available in the knowledge base.
6. Explain why each recommended course fits the student's needs.
7. Never invent course prices, durations, prerequisites, or availability. If information is missing, say so honestly.

Recommend only courses supported by the available course data. If no suitable course exists, explain that the current knowledge base does not contain enough information.

Available course knowledge base:
{json.dumps(courses, indent=2)}
"""

    # Prepare the conversation for Ollama
    conversation = [
        {"role": "system", "content": system_message}
    ] + st.session_state.messages

    with st.chat_message("assistant"):
        with st.spinner("REX is thinking..."):
            try:
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=conversation,
                )

                answer = response["message"]["content"]
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "I couldn't connect to the language model. "
                    "Please check that Ollama is running."
                )
                st.caption(str(error))
