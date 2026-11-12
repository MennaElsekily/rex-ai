"""
The curriculum editor in the Curricula tab (rex_ai_app.py's
curriculum_edit_form): title/description/audience/prerequisites/
objectives/outcomes/modules/assessments/capstone/tools/notes, saved as
a new version through agents/admin_service.py's edit_curriculum. Not
raw JSON -- see agents/curriculum_research/editing.py, tested directly
in tests/test_curriculum_editing.py.
"""

from agents.curriculum_research.curriculum_store import CurriculumStore
from tests.test_rex_ai_app import _curriculum, click, env, start, unlock  # noqa: F401  (env is a fixture)


def test_editing_the_title_and_overview_saves_a_new_version(env):
    store, _, cid = _curriculum()
    at = unlock(start(), section="curricula")

    at.text_input(key=f"cur_edit_title_{cid}").input("A hand-edited title")
    at.text_area(key=f"cur_edit_overview_{cid}").input("A hand-edited overview.")
    at.button(key=f"cur_save_edit_{cid}").click().run()

    assert not at.exception, at.exception
    record = store.get(cid)
    assert record["version"] == 2 and record["content"]["title"] == "A hand-edited title"
    assert record["content"]["overview"] == "A hand-edited overview."
    assert record["status"] == "pending_review"   # never auto-approved


def test_editing_does_not_disturb_other_fields(env):
    store, _, cid = _curriculum()
    original = store.get(cid)["content"]
    at = unlock(start(), section="curricula")

    at.text_input(key=f"cur_edit_title_{cid}").input("Only the title changes")
    at.button(key=f"cur_save_edit_{cid}").click().run()

    updated = store.get(cid)["content"]
    assert updated["title"] == "Only the title changes"
    assert updated["target_audience"] == original["target_audience"]
    assert updated["modules"] == original["modules"]


def test_editing_an_approved_curriculum_invalidates_its_approval(env):
    store, _, cid = _curriculum()
    at = unlock(start(), section="curricula")
    at.button(key=f"cur_approve_{cid}").click().run()
    assert store.get(cid)["status"] == "approved"

    at = unlock(start(), section="curricula")   # a fresh page load, as if the admin came back later
    at.text_input(key=f"cur_edit_title_{cid}").input("Edited after approval")
    at.button(key=f"cur_save_edit_{cid}").click().run()

    assert not at.exception, at.exception
    head = store.get(cid)
    assert head["version"] == 2 and head["status"] == "pending_review"
    row = next(r for r in store.list() if r["curriculum_id"] == cid)
    assert row["approved_version"] == 1   # the earlier approval is kept, not silently extended to the edit
    assert store.get(cid, 1)["content"]["title"] != "Edited after approval"   # version 1 is untouched


def test_the_edited_version_shows_an_approve_button_again(env):
    store, _, cid = _curriculum()
    at = unlock(start(), section="curricula")
    at.button(key=f"cur_approve_{cid}").click().run()

    at = unlock(start(), section="curricula")
    at.text_input(key=f"cur_edit_title_{cid}").input("Edited after approval")
    at.button(key=f"cur_save_edit_{cid}").click().run()

    assert f"cur_approve_{cid}" in {b.key for b in at.button}   # reviewable again


def test_adding_and_removing_a_module_through_the_text_area(env):
    store, _, cid = _curriculum()
    at = unlock(start(), section="curricula")
    modules_widget = at.text_area(key=f"cur_edit_modules_{cid}")
    only_first_module = modules_widget.value.split("-" * 20)[0]   # delete the second module's block

    modules_widget.input(only_first_module + ("\n" + "-" * 20 + "\ntitle: A brand new module\nhours: 5\n"
                                              "topics: New topic\ndescription: New.\nactivities: \noutcome_ids: LO1"))
    at.button(key=f"cur_save_edit_{cid}").click().run()

    assert not at.exception, at.exception
    modules = store.get(cid)["content"]["modules"]
    assert len(modules) == 2 and modules[1]["title"] == "A brand new module"   # the removed one is gone, the new one is added


def test_editing_one_bilingual_companion_does_not_affect_the_other(env):
    store, _, en_cid = _curriculum("en")
    _, _, ar_cid = _curriculum("ar")
    store.link_translation(en_cid, ar_cid)
    ar_before = store.get(ar_cid)["content"]
    at = unlock(start(), section="curricula")

    at.text_input(key=f"cur_edit_title_{en_cid}").input("Edited English only")
    at.button(key=f"cur_save_edit_{en_cid}").click().run()

    assert not at.exception, at.exception
    assert store.get(ar_cid)["content"] == ar_before
    assert store.get(en_cid)["content"]["title"] == "Edited English only"
    assert store.translation_of(en_cid) == ar_cid   # the link survives


def test_an_edit_that_fails_validation_is_shown_as_a_draft_not_silently_saved_as_approved(env):
    store, _, cid = _curriculum()
    at = unlock(start(), section="curricula")

    outcomes_widget = at.text_area(key=f"cur_edit_outcomes_{cid}")
    outcomes_widget.input("LO1: Understand data")   # not a measurable action verb
    at.button(key=f"cur_save_edit_{cid}").click().run()

    assert not at.exception, at.exception
    assert store.get(cid)["status"] == "draft"
    assert f"cur_approve_{cid}" not in {b.key for b in at.button}   # a draft offers no approve button
