"""
Admin navigation (rex_ai_app.py's admin_sidebar): the active section is
server-side session state that only the sidebar -- or an explicit
"open in ..." action -- changes. The nine sections used to be st.tabs,
whose selected tab lived only in the browser: a message appearing above
them after an action (a save, an approval) moved them on the page and the
browser re-opened the first tab, Tender review. These tests pin the new
contract: every action leaves the admin exactly where they were.
"""

import pytest

from tests.test_rex_ai_app import (  # noqa: F401  (env is a fixture)
    _curriculum, click, egypt_id, env, goto, start, unlock,
)

SECTIONS = ["review", "search", "agents", "dashboard", "curricula", "documents", "marketing", "profile", "audit"]
LABELS = ["Tender review", "New tender search", "Ask agents", "Tender dashboard", "Curricula", "Documents",
          "Marketing", "Application profile", "Audit trail"]


def test_the_sidebar_lists_all_nine_sections_and_opens_on_tender_review(env):
    at = unlock(start())

    assert [b.label for b in at.sidebar.button if b.key.startswith("nav_")] == LABELS
    assert at.session_state.admin_section == "review"
    assert at.sidebar.button(key="nav_review").proto.type == "primary"   # the current section is highlighted


@pytest.mark.parametrize("section, title", list(zip(SECTIONS, LABELS)))
def test_each_section_renders_only_itself(env, section, title):
    at = unlock(start(), section=section)

    assert not at.exception, at.exception
    assert [h.value for h in at.subheader][0] == title
    assert at.sidebar.button(key=f"nav_{section}").proto.type == "primary"
    others = [s for s in SECTIONS if s != section]
    assert all(at.sidebar.button(key=f"nav_{s}").proto.type != "primary" for s in others)


def test_plain_reruns_keep_the_section(env):
    at = unlock(start(), section="marketing")
    for _ in range(3):
        at.run()
        assert at.session_state.admin_section == "marketing"


def test_switching_the_language_keeps_the_section(env):
    at = unlock(start(), section="documents")

    at.segmented_control(key="lang").set_value("ar").run()

    assert not at.exception, at.exception
    assert at.session_state.admin_section == "documents"
    assert [h.value for h in at.subheader][0] == "الوثائق"
    at.segmented_control(key="lang").set_value("en").run()
    assert at.session_state.admin_section == "documents"


def test_approving_a_curriculum_stays_in_the_curriculum_section(env):
    """The reported bug: saving or approving in a section jumped back to Tender review."""
    store, _, curriculum_id = _curriculum()
    at = unlock(start(), section="curricula")

    click(at, f"cur_approve_{curriculum_id}")

    assert store.get(curriculum_id)["status"] == "approved"
    assert at.session_state.admin_section == "curricula"
    assert "Approved." in [s.value for s in at.success]   # the flash shows where the admin is


def test_saving_a_curriculum_edit_stays_in_the_curriculum_section(env):
    store, _, curriculum_id = _curriculum()
    at = unlock(start(), section="curricula")

    at.text_input(key=f"cur_edit_title_{curriculum_id}").input("A Better Title")
    click(at, f"cur_save_edit_{curriculum_id}")

    assert store.get(curriculum_id)["version"] == 2
    assert at.session_state.admin_section == "curricula"


def test_saving_the_profile_stays_in_the_profile_section(env):
    at = unlock(start(), section="profile")

    at.text_input(key="profile_legal_name").input("Andalusia Academy LLC")
    click(at, "save_profile")

    assert at.session_state.admin_section == "profile"
    assert "Saved." in [s.value for s in at.success]


def test_approving_a_tender_stays_in_tender_review(env):
    at = unlock(start())

    click(at, f"approved_{egypt_id(env)}")
    click(at, "confirm_approve")

    assert env["tenders"].get(egypt_id(env))["approval_status"] == "approved"
    assert at.session_state.admin_section == "review"


def test_marking_alerts_seen_from_another_section_stays_there(env):
    at = unlock(start(), section="audit")

    click(at, "mark_seen")

    assert at.session_state.admin_section == "audit"


def test_the_search_form_stays_in_the_search_section(env):
    at = unlock(start(), section="search")

    next(b for b in at.button if b.label == "Search tenders").click().run()

    assert not at.exception, at.exception
    assert at.session_state.admin_section == "search"


def test_sending_a_chat_message_stays_in_ask_agents(env):
    at = unlock(start(), section="agents")

    click(at, "agent_example_0")

    assert at.session_state.admin_section == "agents"
    env["orchestrator"].handle_request.assert_called_once()


def test_the_floating_assistant_never_changes_the_section(env):
    for section in ("review", "dashboard", "documents", "marketing", "profile", "audit"):
        at = unlock(start(), section=section)
        at.text_input(key="agent_input_floating").input("Find education tenders in Egypt")
        at.button(key="agent_send_floating").click().run()
        assert not at.exception, at.exception
        assert at.session_state.admin_section == section


def test_navigation_works_in_arabic(env):
    at = start()
    at.session_state["lang"] = "ar"
    at = unlock(at)

    for section in SECTIONS:
        goto(at, section)
        assert not at.exception, at.exception
    assert [b.label for b in at.sidebar.button if b.key.startswith("nav_")][0] == "مراجعة المناقصات"


def test_the_audit_trail_shows_every_area_in_one_readable_table(env):
    store, _, curriculum_id = _curriculum()
    at = unlock(start(), section="curricula")
    click(at, f"cur_approve_{curriculum_id}")
    goto(at, "review")
    click(at, f"approved_{egypt_id(env)}")
    click(at, "confirm_approve")
    goto(at, "audit")

    [table] = [f.value for f in at.dataframe if "Area" in f.value.columns]
    assert list(table.columns) == ["When (UTC)", "Area", "Item", "Version", "Action", "Status", "By", "Notes"]
    assert {"Tenders", "Curricula"} <= set(table["Area"])
    assert list(table["When (UTC)"]) == sorted(table["When (UTC)"], reverse=True)   # newest first
    curriculum_rows = table[table["Item"] == curriculum_id]
    assert "pending_review → approved" in set(curriculum_rows["Status"])
    assert all(by.startswith(("admin-session-", "system:")) for by in table["By"])

    at.segmented_control(key="audit_area").set_value("curriculum").run()
    [filtered] = [f.value for f in at.dataframe if "Area" in f.value.columns]
    assert set(filtered["Area"]) == {"Curricula"}
    assert at.session_state.admin_section == "audit"
