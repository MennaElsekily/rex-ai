
import json
import re
from pathlib import Path

import ollama
import streamlit as st


# --------------------------------------------------
# 1. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="REX Assistant",
    page_icon="🎓",
)

st.title("🎓 REX Assistant")
st.caption("Your personal educational and career guidance assistant.")


# --------------------------------------------------
# 2. Load the course catalogue
# --------------------------------------------------

COURSES_PATH = Path(__file__).parent / "data" / "courses.json"

with open(COURSES_PATH, "r", encoding="utf-8") as file:
    courses_data = json.load(file)

if isinstance(courses_data, list):
    courses = courses_data
elif isinstance(courses_data, dict):
    courses = courses_data.get("courses", [])
else:
    courses = []

if not isinstance(courses, list):
    raise ValueError("courses.json must contain a list of courses.")

courses = [course for course in courses if isinstance(course, dict)]


# --------------------------------------------------
# 3. Course search
# --------------------------------------------------
# --------------------------------------------------
# 3. Course search
# --------------------------------------------------

TOPIC_RULES = {
    "python": ("python",),
    "programming": ("programming", "python"),
    "coding": ("programming", "python"),
    "web development": ("web development",),
    "website": ("web development",),
    "web": ("web development",),
    "html": ("web development",),
    "css": ("web development",),
    "javascript": ("web development",),
    "sql": ("sql for data analysis",),
    "database": ("sql for data analysis",),
    "data analysis": ("data analysis",),
    "analytics": ("analytics", "data analysis"),
    "power bi": ("power bi",),
    "excel": ("excel",),
    "machine learning": ("machine learning",),
    "artificial intelligence": (
        "artificial intelligence",
        "responsible ai",
        "generative ai",
    ),
    " ai ": (
        "artificial intelligence",
        "responsible ai",
        "generative ai",
    ),
    "cybersecurity": ("cybersecurity",),
    "marketing": ("marketing",),
    "cloud": ("cloud computing",),
    "project management": ("project management",),
    "business analysis": ("business analysis",),
    "communication": ("communication",),
    "presentation": ("communication",),
    "instructional design": ("instructional design",),
    "train the trainer": ("train the trainer",),
    "human resources": ("human resources",),
    " hr ": ("human resources",),
    "responsible ai": ("responsible ai",),
    "generative ai": ("generative ai",),
}


BEGINNER_TERMS = (
    "beginner",
    "beginners",
    "from scratch",
    "new to",
    "starting",
    "start learning",
    "no experience",
)


def search_courses(user_input, courses, limit=5):
    """Return relevant courses that actually exist in the catalogue."""

    text = user_input.lower()
    padded_text = f" {text} "

    wants_beginner = any(
        term in text for term in BEGINNER_TERMS
    )

    requested_topics = [
        (phrase.strip(), targets)
        for phrase, targets in TOPIC_RULES.items()
        if phrase in padded_text
    ]

    # Avoid treating a general beginner request as a topic request.
    general_beginner = (
        wants_beginner and not requested_topics
    ) or (
        any(
            phrase in text
            for phrase in (
                "what courses",
                "which courses",
                "courses do you have",
                "available courses",
                "show me courses",
                "list courses",
            )
        )
        and not requested_topics
    )

    results = []

    for course in courses:
        title = str(course.get("title", ""))
        category = str(course.get("category", ""))
        level = str(course.get("level", "")).lower()

        title_lower = title.lower()
        category_lower = category.lower()

        is_beginner = level == "beginner"

        if general_beginner:
            if is_beginner:
                results.append((10, course))
            continue

        if not requested_topics:
            continue

        score = 0

        for _, targets in requested_topics:
            if any(target in title_lower for target in targets):
                score += 20
            elif any(target in category_lower for target in targets):
                score += 10

        if score == 0:
            continue

        # A beginner asking from scratch should not receive
        # intermediate or beginner-to-intermediate courses.
        if wants_beginner and not is_beginner:
            continue

        if wants_beginner:
            score += 5 if is_beginner else 0

        results.append((score, course))

    results.sort(key=lambda item: item[0], reverse=True)

    return [course for _, course in results[:limit]]
# --------------------------------------------------
# 4. Identify catalogue-related requests
# --------------------------------------------------

def is_course_request(user_input):
    """Detect requests that should be answered from the catalogue."""

    text = user_input.lower()

    course_terms = (
        "course",
        "courses",
        "training",
        "curriculum",
        "class",
        "classes",
        "learn",
        "learning path",
        "enroll",
        "enrol",
    )

    return any(term in text for term in course_terms)


# --------------------------------------------------
# 5. Build recommendations from real catalogue fields
# --------------------------------------------------

def format_course_recommendations(matching_courses):
    """Display only facts present in the catalogue records."""

    if not matching_courses:
        return (
            "I couldn't find a matching course in the current "
            "catalogue. Could you try another subject or topic?"
        )

    lines = ["Here are the courses from the current catalogue that match your request:\n"]

    for index, course in enumerate(matching_courses, start=1):
        title = course.get("title")
        if not title:
            continue

        lines.append(f"### {index}. {title}")

        # Display only fields that actually exist and have values.
        display_fields = (
            ("level", "Level"),
            ("category", "Category"),
            ("description", "Description"),
            ("prerequisites", "Prerequisites"),
            ("duration", "Duration"),
            ("fee", "Fee"),
            ("learning_outcomes", "Learning outcomes"),
        )

        for field, label in display_fields:
            value = course.get(field)

            if value is None or value == "" or value == []:
                continue

            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)

            lines.append(f"- **{label}:** {value}")

        lines.append("")

    lines.append(
        "_Recommendations are based on the development catalogue. "
        "Course availability and offerings have not been verified._"
    )

    return "\n".join(lines)


# --------------------------------------------------
# 6. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm REX, your learning assistant. 👋 "
                "I'll help you find courses that match your goals. "
                "What would you like to learn?"
            ),
        }
    ]


# --------------------------------------------------
# 7. Display chat history
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 8. Handle user input
# --------------------------------------------------

user_input = st.chat_input("Message REX Assistant...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    matching_courses = search_courses(user_input, courses)

    # Catalogue questions are answered directly from Python.
    # Ollama cannot invent course titles or course details here.
    if is_course_request(user_input):
        answer = format_course_recommendations(matching_courses)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:
        # Use Ollama for general educational conversation.
        system_message = """
You are REX Assistant, an educational and career guidance chatbot.

Be friendly, clear, and concise.
Help users clarify their educational and career goals.

Do not invent catalogue facts, course names, prices, durations,
certificates, prerequisites, or course availability.

When asked for a course recommendation, direct the user to the
course catalogue. Do not invent a course or claim it is officially
offered by Andalusia Academy.

The catalogue contains synthetic development data and has not
been verified as official academy offerings.
"""

        conversation = [
            {"role": "system", "content": system_message}
        ] + st.session_state.messages[-12:]

        with st.chat_message("assistant"):
            with st.spinner("REX is thinking..."):
                try:
                    response = ollama.chat(
                        model="llama3.2:1b",
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
                        "Please check that Ollama is running and "
                        "that llama3.2:1b is installed."
                    )
                    st.caption(str(error))
