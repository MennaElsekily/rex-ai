"""
The redesigned Ask Agents chat (rex_ai_app.py's agents_tab): one
conversational interface over the existing Central Orchestrator. These
tests cover what changed -- the "New conversation" reset, the
multi-agent summary line, and partial-failure wording -- not the
orchestrator's own routing logic, which tests/test_orchestrator.py and
tests/test_routing_deliverables.py already cover.
"""

from tests.test_rex_ai_app import click, env, start, unlock  # noqa: F401  (env is a fixture)


def test_new_conversation_resets_history_and_the_conversation_id(env):
    at = unlock(start(), section="agents")
    click(at, "agent_example_0")
    assert at.session_state.admin_chat   # a turn happened
    old_conversation_id = at.session_state.admin_conversation_id

    click(at, "new_conversation")

    assert at.session_state.admin_chat == []
    assert at.session_state.admin_conversation_id != old_conversation_id


def test_the_new_conversation_button_is_disabled_before_any_message(env):
    at = unlock(start(), section="agents")
    [button] = [b for b in at.button if b.key == "new_conversation"]
    assert button.disabled


def test_multiple_agent_results_show_an_agents_involved_caption(env):
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [
        {"agent_name": "curriculum_research", "status": "success",
         "result": {"message": "Curriculum created.", "curriculum_id": "CUR-1"}},
        {"agent_name": "marketing", "status": "success", "result": {"message": "Campaign created."}},
    ]}
    at = unlock(start(), section="agents")
    click(at, "agent_example_0")

    text = " ".join(c.value for c in at.caption)
    assert "Agents involved" in text
    assert "Curriculum Research" in text and "Marketing" in text


def test_a_partial_failure_is_shown_as_a_warning_not_hidden(env):
    env["orchestrator"].handle_request.return_value = {"status": "error", "result": [
        {"agent_name": "curriculum_research", "status": "success",
         "result": {"message": "Curriculum created.", "curriculum_id": "CUR-1"}},
        {"agent_name": "document_generation", "status": "error", "error_message": "No approved curriculum yet."},
    ], "error_message": "No approved curriculum yet."}
    at = unlock(start(), section="agents")
    click(at, "agent_example_0")

    assert any("could not be completed" in w.value for w in at.warning)
    assert any("No approved curriculum yet" in e.value for e in at.error)
    assert any("Curriculum created" in m.value for m in at.markdown)   # the successful half is still shown


def test_a_single_agent_result_shows_no_agents_involved_caption(env):
    at = unlock(start(), section="agents")
    click(at, "agent_example_0")   # the default fake orchestrator reply: one rfp_research result

    assert not any("Agents involved" in c.value for c in at.caption)


def test_a_resumed_workflow_is_injected_as_a_new_assistant_turn(env):
    """
    Simulates what admin_service.py's approve_curriculum() already did
    (tests/test_pending_workflows.py covers that mechanism itself): a
    resumed result is waiting for this conversation, and the chat picks
    it up and shows it automatically on its next render, with no new
    message typed by the admin.
    """
    from agents.pending_workflows import PendingWorkflowStore

    at = unlock(start(), section="agents")
    conversation_id = at.session_state.admin_conversation_id
    PendingWorkflowStore().save_resumed_result(conversation_id, {
        "request_id": "r", "agent_name": "orchestrator", "status": "success", "error_message": None,
        "result": [{"agent_name": "document_generation", "status": "success",
                   "result": {"kind": "document", "message": "Course Description generated."}}]})

    at.run()

    assert any("Course Description generated" in m.value for m in at.markdown)
    assert any("approved" in m.value.lower() for m in at.markdown)
