"""
Application tracking in OpportunityStore: three separate statuses
(approval, submission, application), the atomic submission claim,
attempt history, restart recovery, audit events and the migration.
Temp SQLite files only.
"""

import sqlite3
import threading

import pytest

from agents.rfp_research.opportunity_store import (
    APPLIED,
    IN_PROGRESS,
    NOT_APPLIED,
    NOT_STARTED,
    OUTCOME_UNKNOWN,
    REQUIRES_MANUAL_ACTION,
    SUBMISSION_FAILED,
    SUBMITTED,
    SYSTEM_ACTOR,
    InvalidTransition,
    OpportunityStore,
)

ADMIN = "admin-session-1"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "tenders.sqlite3"


@pytest.fixture
def store(db):
    s = OpportunityStore(db_path=db)
    yield s
    s.close()


def add(store, url="https://etenders.gov.eg/t/1", title="Teacher Training Services Tender", approve=True):
    store.save({"url": url, "title": title, "country": "Egypt", "organization": "Ministry of Education",
                "submission_url": "https://etenders.gov.eg/submit/1"})
    opp_id = store.search(keyword=title.split()[0])[0]["opportunity_id"]
    if approve:
        store.decide(opp_id, "approved", ADMIN)
    return opp_id


def claim(store, opp_id, dry_run=False):
    return store.claim_submission(opp_id, "email_eoi", ADMIN, dry_run=dry_run)


# --------------------------------------------------------------------
# Three separate statuses
# --------------------------------------------------------------------

def test_new_and_approved_tenders_are_not_started_and_not_applied(store):
    opp_id = add(store)

    row = store.get(opp_id)
    assert (row["approval_status"], row["submission_status"], row["application_status"]) == (
        "approved", NOT_STARTED, NOT_APPLIED)
    assert row["submission_authorized"] is False  # approval alone grants nothing


def test_the_submission_authorisation_is_recorded_with_the_approval(store):
    opp_id = add(store, approve=False)

    store.decide(opp_id, "approved", ADMIN, authorize_submission=True)

    row = store.get(opp_id)
    assert row["submission_authorized"] is True and row["submission_authorized_by"] == ADMIN
    review = store.audit_trail(opp_id)[-1]
    assert (review["event"], review["decision"], review["details"]) == (
        "review", "approved", {"authorize_submission": True})


def test_authorisation_is_cleared_when_the_approval_is_withdrawn(store):
    opp_id = add(store, approve=False)
    store.decide(opp_id, "approved", ADMIN, authorize_submission=True)

    store.decide(opp_id, "pending", ADMIN, authorize_submission=True)  # ignored: not an approval

    assert store.get(opp_id)["submission_authorized"] is False


# --------------------------------------------------------------------
# Atomic claim
# --------------------------------------------------------------------

def test_only_approved_tenders_can_be_claimed(store):
    assert claim(store, add(store, approve=False)) is None


def test_a_second_claim_gets_nothing(store):
    opp_id = add(store)

    first = claim(store, opp_id)

    assert first["status"] == IN_PROGRESS and first["attempt_no"] == 1
    assert first["actor"] == SYSTEM_ACTOR and first["triggered_by"] == ADMIN
    assert claim(store, opp_id) is None
    assert store.get(opp_id)["submission_status"] == IN_PROGRESS


def test_simultaneous_claims_from_two_connections_have_one_winner(db, store):
    opp_id = add(store)
    stores = [OpportunityStore(db_path=db) for _ in range(6)]
    results, errors, start = [], [], threading.Barrier(len(stores))

    def run(s):
        start.wait()
        try:
            results.append(claim(s, opp_id))
        except Exception as exc:  # e.g. "database is locked" must not be hidden
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [] and len(results) == len(stores)
    assert sum(1 for r in results if r) == 1
    assert len(store.attempts(opp_id)) == 1


@pytest.mark.parametrize("ended, claimable", [
    (REQUIRES_MANUAL_ACTION, True), (SUBMISSION_FAILED, True), (OUTCOME_UNKNOWN, False),
])
def test_which_results_allow_a_new_run(store, ended, claimable):
    opp_id = add(store)
    store.finish_attempt(claim(store, opp_id)["attempt_id"], ended)

    assert (claim(store, opp_id) is not None) is claimable


# --------------------------------------------------------------------
# Finishing attempts: dry runs are never submissions
# --------------------------------------------------------------------

def test_a_dry_run_can_never_be_recorded_as_submitted(store):
    opp_id = add(store)
    attempt = claim(store, opp_id, dry_run=True)

    with pytest.raises(InvalidTransition):
        store.finish_attempt(attempt["attempt_id"], SUBMITTED, confirmation={"message_id": "x"})

    done = store.finish_attempt(attempt["attempt_id"], REQUIRES_MANUAL_ACTION,
                                confirmation={"prepared_email": {"to": "eoi@agency.gov.eg"}},
                                manual_actions=["Review and send the prepared email."])
    assert done["dry_run"] is True and done["manual_actions"] == ["Review and send the prepared email."]
    assert store.get(opp_id)["application_status"] == NOT_APPLIED


def test_submitted_needs_a_confirmation(store):
    attempt = claim(store, add(store))

    with pytest.raises(InvalidTransition):
        store.finish_attempt(attempt["attempt_id"], SUBMITTED)


def test_a_confirmed_submission_marks_the_tender_applied(store):
    opp_id = add(store)
    attempt = claim(store, opp_id)

    store.finish_attempt(attempt["attempt_id"], SUBMITTED, submission_reference="MSG-1",
                         submission_url="mailto:eoi@agency.gov.eg", confirmation={"message_id": "<1@rex>"})

    row = store.get(opp_id)
    assert (row["submission_status"], row["application_status"], row["submission_method"]) == (
        SUBMITTED, APPLIED, "email_eoi")
    assert row["submitted_via_url"] == "mailto:eoi@agency.gov.eg"
    assert row["submission_url"] == "https://etenders.gov.eg/submit/1"  # the discovered link is untouched


def test_an_attempt_finishes_only_once(store):
    attempt = claim(store, add(store))
    store.finish_attempt(attempt["attempt_id"], SUBMISSION_FAILED, error="Connection refused")

    with pytest.raises(InvalidTransition):
        store.finish_attempt(attempt["attempt_id"], REQUIRES_MANUAL_ACTION)


# --------------------------------------------------------------------
# Restarts and unknown results
# --------------------------------------------------------------------

def test_an_interrupted_attempt_becomes_unknown_and_blocks_new_runs(db, store):
    opp_id = add(store)
    claim(store, opp_id)
    store.close()

    reopened = OpportunityStore(db_path=db)  # e.g. after a restart
    assert reopened.recover_interrupted_attempts() == 1

    assert reopened.get(opp_id)["submission_status"] == OUTCOME_UNKNOWN
    assert reopened.attempts(opp_id)[0]["status"] == OUTCOME_UNKNOWN
    assert claim(reopened, opp_id) is None
    reopened.close()


def test_an_unknown_result_resolved_as_not_sent_allows_a_new_run(store):
    opp_id = add(store)
    store.finish_attempt(claim(store, opp_id)["attempt_id"], OUTCOME_UNKNOWN, error="Timed out")

    store.mark_not_submitted(opp_id, ADMIN, notes="Not in the sent folder.")

    assert store.get(opp_id)["submission_status"] == SUBMISSION_FAILED
    assert claim(store, opp_id)["attempt_no"] == 2
    assert store.audit_trail(opp_id)[-1]["event"] == "application"


def test_an_unknown_result_resolved_as_sent_is_a_manual_submission(store):
    opp_id = add(store)
    store.finish_attempt(claim(store, opp_id)["attempt_id"], OUTCOME_UNKNOWN)

    row = store.record_manual_submission(opp_id, ADMIN, "2026-11-02", reference="Sent folder")

    assert (row["submission_status"], row["application_status"]) == (SUBMITTED, APPLIED)
    assert store.audit_trail(opp_id)[-1]["details"]["resolved_unknown_outcome"] is True


# --------------------------------------------------------------------
# Manual submission, outcomes and review locks
# --------------------------------------------------------------------

def test_an_admin_records_a_verified_manual_submission(store):
    opp_id = add(store)
    store.finish_attempt(claim(store, opp_id)["attempt_id"], REQUIRES_MANUAL_ACTION)

    row = store.record_manual_submission(opp_id, ADMIN, "2026-11-02", reference="ETM-77",
                                         url="https://tenders.etimad.sa/offer/77", notes="Via Nafath")

    assert (row["submission_status"], row["application_status"]) == (SUBMITTED, APPLIED)
    assert (row["submission_method"], row["submitted_at"], row["submission_reference"]) == (
        "manual", "2026-11-02", "ETM-77")
    manual = store.attempts(opp_id)[-1]
    assert (manual["adapter"], manual["actor"], manual["status"]) == ("manual", ADMIN, SUBMITTED)
    audit = store.audit_trail(opp_id)[-1]
    assert (audit["event"], audit["previous_status"], audit["decision"]) == ("application", NOT_APPLIED, APPLIED)


@pytest.mark.parametrize("approve, date, error", [
    (False, "2026-11-02", InvalidTransition), (True, "next week", ValueError), (True, "", ValueError),
])
def test_invalid_manual_submissions(store, approve, date, error):
    opp_id = add(store, approve=approve)

    with pytest.raises(error):
        store.record_manual_submission(opp_id, ADMIN, date)


def test_outcomes_follow_an_application(store):
    opp_id = add(store)
    with pytest.raises(InvalidTransition):
        store.record_application_outcome(opp_id, "awarded", ADMIN)  # not applied yet

    store.record_manual_submission(opp_id, ADMIN, "2026-11-02")
    store.record_application_outcome(opp_id, "not_awarded", ADMIN)
    assert store.get(opp_id)["application_status"] == "not_awarded"

    store.record_application_outcome(opp_id, "applied", ADMIN, notes="Recorded by mistake")  # correction
    store.record_application_outcome(opp_id, "awarded", ADMIN)
    assert store.get(opp_id)["application_status"] == "awarded"

    with pytest.raises(ValueError):
        store.record_application_outcome(opp_id, "approved", ADMIN)


@pytest.mark.parametrize("state", ["in_progress", "unknown", "applied"])
def test_an_application_locks_the_review_decision(store, state):
    opp_id = add(store)
    attempt = claim(store, opp_id)
    if state == "unknown":
        store.finish_attempt(attempt["attempt_id"], OUTCOME_UNKNOWN)
    elif state == "applied":
        store.finish_attempt(attempt["attempt_id"], SUBMITTED, confirmation={"message_id": "x"})

    with pytest.raises(InvalidTransition):
        store.decide(opp_id, "rejected", ADMIN)
    assert store.get(opp_id)["approval_status"] == "approved"


# --------------------------------------------------------------------
# Rediscovery, search, counts, persistence, migration
# --------------------------------------------------------------------

def test_rediscovery_never_touches_decisions_or_applications(store):
    opp_id = add(store)
    store.record_manual_submission(opp_id, ADMIN, "2026-11-02", reference="R-1")
    before = store.get(opp_id)

    store.save({"url": "https://etenders.gov.eg/t/1", "title": "Teacher Training Services Tender",
                "country": "Egypt", "approval_status": "pending", "application_status": "not_applied",
                "submission_status": "not_started"})

    after = store.get(opp_id)
    for field in ("approval_status", "submission_status", "application_status", "submission_reference",
                  "submitted_at", "decided_by"):
        assert after[field] == before[field]
    assert len(store.attempts(opp_id)) == 1


def test_search_and_counts_by_application_state(store):
    applied = add(store)
    manual = add(store, "https://tenders.etimad.sa/t/2", "Digital Skills Programme Tender")
    add(store, "https://c.example/3", "Corporate Learning Platform Tender", approve=False)
    store.record_manual_submission(applied, ADMIN, "2026-11-02")
    store.finish_attempt(claim(store, manual)["attempt_id"], REQUIRES_MANUAL_ACTION)

    assert [r["opportunity_id"] for r in store.search(application_status="applied")] == [applied]
    assert [r["opportunity_id"] for r in store.search(submission_status=REQUIRES_MANUAL_ACTION)] == [manual]
    assert len(store.search(submission_status=NOT_STARTED)) == 1  # the pending one (NULL in the table)
    counts = store.counts()
    assert (counts["approved"], counts["pending"], counts["applied"], counts["needs_manual_action"]) == (2, 1, 1, 1)
    with pytest.raises(ValueError):
        store.search(application_status="won")


def test_history_survives_reopening_the_database(db, store):
    opp_id = add(store)
    store.finish_attempt(claim(store, opp_id)["attempt_id"], REQUIRES_MANUAL_ACTION, missing=["Company profile"])
    store.close()

    reopened = OpportunityStore(db_path=db)
    assert reopened.attempts(opp_id)[0]["missing"] == ["Company profile"]
    assert [a["event"] for a in reopened.audit_trail(opp_id)] == ["review"]
    reopened.close()


def test_a_database_from_before_applications_is_migrated_without_losing_anything(db):
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE opportunities (url TEXT PRIMARY KEY, title TEXT, country TEXT,
            opportunity_id TEXT, approval_status TEXT, decided_by TEXT);
        INSERT INTO opportunities VALUES ('https://old.example/1', 'Old Training Tender', 'Egypt',
            'abc123', 'approved', 'admin-session-0');
        CREATE TABLE opportunity_audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, opportunity_id TEXT NOT NULL,
            title TEXT, previous_status TEXT, decision TEXT NOT NULL, reviewer TEXT NOT NULL, decided_at TEXT NOT NULL);
        INSERT INTO opportunity_audit (opportunity_id, title, previous_status, decision, reviewer, decided_at)
            VALUES ('abc123', 'Old Training Tender', 'pending', 'approved', 'admin-session-0', '2026-10-01');
    """)
    old.commit()
    old.close()

    store = OpportunityStore(db_path=db)

    row = store.get("abc123")
    assert (row["approval_status"], row["decided_by"]) == ("approved", "admin-session-0")
    assert (row["submission_status"], row["application_status"]) == (NOT_STARTED, NOT_APPLIED)
    [audit] = store.audit_trail("abc123")
    assert (audit["event"], audit["decision"], audit["details"]) == ("review", "approved", {})
    assert claim(store, "abc123")["attempt_no"] == 1  # the new table works on the old file
    store.close()
