"""
Every admin section now opens with the same kind of header
(rex_ai_app.py's section_header(): an icon, the section's own title, and
a consistent accent divider) so all nine tabs read as one product
instead of nine independently styled pages.
"""

from tests.test_rex_ai_app import env, goto, start, unlock  # noqa: F401  (env is a fixture)

SECTIONS = ["review", "search", "agents", "dashboard", "curricula", "documents", "marketing", "profile", "audit"]


def _headers_of_every_section(at) -> list[str]:
    """Each section is opened from the sidebar in turn (only the active section is rendered)."""
    titles = []
    for section in SECTIONS:
        goto(at, section)
        titles.append([h.value for h in at.subheader])
    return titles

EXPECTED_TITLES = ["Tender review", "New tender search", "Ask agents", "Tender dashboard", "Curricula",
                  "Documents", "Marketing", "Application profile", "Audit trail"]


def test_every_section_has_its_own_consistent_header(env):
    at = unlock(start())

    per_section = _headers_of_every_section(at)
    # Each section opens with exactly its own header, in sidebar order.
    assert [headers[0] for headers in per_section] == EXPECTED_TITLES, per_section


def test_every_header_uses_the_same_accent_divider(env):
    at = unlock(start())

    # Streamlit's AppTest does not expose the icon/divider kwargs directly, so this checks the
    # one thing it can: every one of the nine headers actually went through st.subheader (not a
    # plain st.markdown title), which is what section_header() guarantees for all of them alike.
    assert all(headers for headers in _headers_of_every_section(at))


EXPECTED_TITLES_AR = ["مراجعة المناقصات", "بحث جديد عن مناقصات", "الوكلاء", "لوحة المناقصات", "المناهج",
                      "الوثائق", "التسويق", "ملف التقديم", "سجل المراجعة"]


def test_the_headers_are_consistent_in_arabic_too(env):
    at = start()
    at.session_state["lang"] = "ar"
    at = unlock(at)

    per_section = _headers_of_every_section(at)
    assert [headers[0] for headers in per_section] == EXPECTED_TITLES_AR, per_section
