"""
The admin "Documents" tab (rex_ai_app.py, streamlit.testing AppTest): the
real AdminService and real stores on the temporary default paths set in
tests/conftest.py; the agents are the fake admin orchestrator of
tests/test_rex_ai_app.py. Documents are created with the real toolbox.
"""

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation import templates
from agents.document_generation.composer import compose
from agents.document_generation.document_store import DocumentStore
from agents.document_generation.document_tools import DocumentToolbox
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, parts_answers
from tests.test_rex_ai_app import click, env, start, unlock  # noqa: F401  (env is a fixture)


@pytest.fixture
def default_toolbox():
    """The toolbox on the (temporary) default paths the app's AdminService also uses."""
    return DocumentToolbox(store=DocumentStore(), curricula=CurriculumStore())


def make_document(toolbox, approve_curriculum=True):
    saved = toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    if approve_curriculum:
        toolbox.curricula.approve(saved["curriculum_id"], 1, ADMIN)
    record = toolbox.curricula.get(saved["curriculum_id"])
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]))
    document = toolbox.save_document_draft(content)
    toolbox.render_document_pdf(document["document_id"])
    return saved["curriculum_id"], document["document_id"]


def test_documents_tab_is_there_and_empty_without_approved_curricula(env):
    at = unlock(start())

    assert "Documents" in [tab.label for tab in at.tabs]
    assert any("No curriculum is approved yet" in i.value for i in at.info)
    assert any("No documents yet" in i.value for i in at.info)


def test_documents_tab_offers_the_form_when_a_curriculum_is_approved(env, default_toolbox):
    default_toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    cid = default_toolbox.curricula.list()[0]["curriculum_id"]
    default_toolbox.curricula.approve(cid, 1, ADMIN)

    at = unlock(start())

    assert not at.exception
    assert [s.label for s in at.selectbox if s.key == "doc_type"]
    assert any(cid in str(o) for s in at.selectbox if s.key == "doc_curriculum" for o in s.options)


def test_generating_from_the_form_asks_the_document_agent_with_the_choices(env, default_toolbox):
    cid = default_toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])["curriculum_id"]
    default_toolbox.curricula.approve(cid, 1, ADMIN)
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [{
        "agent_name": "document_generation", "status": "success",
        "result": {"kind": "document", "message": "Course Description generated."}}]}

    at = unlock(start())
    at.text_input(key="doc_extra_format").input("Blended")
    at.button[[b.label for b in at.button].index("Generate document")].click().run()

    assert not at.exception, at.exception
    call = env["orchestrator"].handle_request.call_args
    request = call.kwargs["context"]["document_request"]
    assert request == {"curriculum_id": cid, "document_type": "course_description", "language": None,
                       "extra": {"training_format": "Blended"}, "document_id": None}
    assert any("Course Description generated" in s.value for s in at.success)   # kept after the redraw


def test_review_edit_and_approve_in_the_tab(env, default_toolbox):
    cid, document_id = make_document(default_toolbox)
    at = unlock(start())
    assert any(document_id in m.value for m in at.markdown)
    assert any("Not specified in the curriculum" in w.value for w in at.warning)

    summary_key = f"doc_{document_id}_1_summary_0"
    assert at.text_area(key=summary_key).value == parts_answers()["DescriptionParts"]["summary"]
    at.text_area(key=summary_key).input("A reviewed summary.")
    at.button(key=f"doc_save_{document_id}").click().run()
    assert not at.exception, at.exception

    stored = default_toolbox.store.get(document_id)
    assert stored["current_version"] == 2 and stored["status"] == "pending_review"
    assert "A reviewed summary." in str(stored["content"]) and stored["filename"].endswith("_v2_en.pdf")

    at.button(key=f"doc_approve_{document_id}").click().run()
    assert not at.exception, at.exception
    approved = default_toolbox.store.get(document_id)
    assert approved["status"] == "approved" and approved["approved_version"] == 2
    assert not [b for b in at.button if b.key == f"doc_approve_{document_id}"]   # nothing left to approve


def test_the_tab_is_available_in_arabic(env, default_toolbox):
    make_document(default_toolbox)
    at = start()
    at.session_state["lang"] = "ar"
    unlock(at)

    assert not at.exception
    assert "الوثائق" in [tab.label for tab in at.tabs]
