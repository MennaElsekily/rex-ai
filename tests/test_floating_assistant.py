"""
The floating REX AI chatbot launcher (rex_ai_app.py's floating_assistant):
a popover available from every admin section except Ask agents itself
(where the very same chat already IS the page), reusing the exact same
Ask Agents conversation and the same Central Orchestrator call --
agents_tab() itself, rendered with a key_suffix. Opening it or chatting
in it never changes the active section.
"""

from tests.test_rex_ai_app import click, env, goto, start, unlock  # noqa: F401  (env is a fixture)


def test_the_launcher_is_present_and_reuses_the_same_conversation(env):
    at = unlock(start())

    assert any(b.key == "agent_send_floating" for b in at.button)   # the floating instance's own send button
    goto(at, "agents")
    assert any(b.key == "agent_send" for b in at.button)             # the Ask Agents page's own send button
    assert not any(b.key == "agent_send_floating" for b in at.button)   # never a second chat on the same page


def test_sending_from_the_floating_launcher_updates_the_same_chat_history(env):
    at = unlock(start())

    at.text_input(key="agent_input_floating").input("Find education tenders in Egypt")
    at.button(key="agent_send_floating").click().run()

    assert not at.exception, at.exception
    assert len(at.session_state.admin_chat) == 2
    assert at.session_state.admin_chat[0]["content"] == "Find education tenders in Egypt"
    env["orchestrator"].handle_request.assert_called_once()   # through the same Central Orchestrator
    # The same history shows up through the ordinary Ask Agents page too.
    goto(at, "agents")
    assert any("Found 0 relevant tender" in m.value for m in at.markdown)


def test_sending_from_the_tab_shows_up_in_the_floating_launcher_too(env):
    at = unlock(start(), section="agents")
    conversation_id = at.session_state.admin_conversation_id

    click(at, "agent_example_0")
    goto(at, "curricula")

    assert at.session_state.admin_chat   # one turn happened
    assert at.session_state.admin_conversation_id == conversation_id   # the same conversation
    # The floating launcher renders the same session_state.admin_chat: its "new conversation" is enabled too.
    assert not next(b for b in at.button if b.key == "new_conversation_floating").disabled


def test_opening_the_launcher_never_changes_the_selected_section_or_session(env):
    at = unlock(start(), section="curricula")
    token_before, view_before = at.session_state.admin_token, at.session_state.view

    at.text_input(key="agent_input_floating").input("Find education tenders in Egypt")
    at.button(key="agent_send_floating").click().run()

    assert not at.exception, at.exception
    assert at.session_state.view == view_before == "admin"
    assert at.session_state.admin_section == "curricula"   # still where the admin was
    assert at.session_state.admin_token == token_before   # authentication survived the rerun


def test_a_new_conversation_from_the_floating_launcher_resets_both_instances(env):
    at = unlock(start(), section="agents")
    click(at, "agent_example_0")
    old_conversation_id = at.session_state.admin_conversation_id
    goto(at, "review")

    at.button(key="new_conversation_floating").click().run()

    assert at.session_state.admin_chat == []
    assert at.session_state.admin_conversation_id != old_conversation_id
    assert at.session_state.admin_section == "review"
