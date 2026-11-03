"""
Admin sign-in lockout is per client (one browser session), so one client's wrong passcodes cannot lock out the others.
A ceiling over all clients still stops guessing through many fresh sessions. The passcode is shared: there is no
individual administrator identity (a session token is the only identity, recorded as admin-session-<hash>).
"""

import pytest

from agents import admin_service
from agents.admin_service import (
    LOCKOUT_SECONDS, MAX_FAILED_ATTEMPTS, MAX_FAILED_ATTEMPTS_ALL_CLIENTS, MAX_TRACKED_CLIENTS, AdminAuthError, AdminService,
)
from agents.curriculum_research.curriculum_store import CurriculumStore
from tests.curriculum_fixtures import valid_curriculum
from tests.test_rex_ai_app import click, env, start, unlock  # noqa: F401  (env is a fixture)

PASSCODE = "correct horse"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service(clock, tmp_path):
    store = CurriculumStore(tmp_path / "cur.sqlite3")
    return AdminService(orchestrator_factory=lambda: None, passcode=PASSCODE, clock=clock, curriculum_store=store,
                        curriculum_documents_dir=str(tmp_path / "pdfs"))


def fail(service, client, times):
    for _ in range(times):
        assert service.login("wrong", client) is None


def test_the_right_passcode_signs_in_and_a_wrong_one_does_not(service):
    assert service.login(PASSCODE, "a")
    assert service.login("wrong", "a") is None and not service.is_locked("a")
    assert service.login(None, "a") is None


def test_a_client_is_locked_at_the_threshold_even_with_the_right_passcode(service):
    fail(service, "a", MAX_FAILED_ATTEMPTS - 1)
    assert not service.is_locked("a")
    service.login("wrong", "a")
    assert service.is_locked("a")
    assert service.login(PASSCODE, "a") is None


def test_the_lockout_expires(service, clock):
    fail(service, "a", MAX_FAILED_ATTEMPTS)
    clock.now += LOCKOUT_SECONDS - 1
    assert service.is_locked("a") and service.login(PASSCODE, "a") is None
    clock.now += 2
    assert not service.is_locked("a") and service.login(PASSCODE, "a")


def test_after_a_lockout_the_count_starts_again(service, clock):
    fail(service, "a", MAX_FAILED_ATTEMPTS)
    clock.now += LOCKOUT_SECONDS + 1
    fail(service, "a", MAX_FAILED_ATTEMPTS - 1)
    assert not service.is_locked("a")


def test_a_success_clears_the_clients_failures(service):
    fail(service, "a", MAX_FAILED_ATTEMPTS - 1)
    assert service.login(PASSCODE, "a")
    fail(service, "a", MAX_FAILED_ATTEMPTS - 1)
    assert not service.is_locked("a")


def test_one_locked_client_does_not_lock_another(service):
    fail(service, "attacker", MAX_FAILED_ATTEMPTS)
    assert service.is_locked("attacker") and not service.is_locked("admin")
    assert service.login(PASSCODE, "admin")
    assert service.login(PASSCODE, "attacker") is None


def test_failures_of_different_clients_are_not_added_together(service):
    for client in ("a", "b", "c", "d"):
        fail(service, client, MAX_FAILED_ATTEMPTS - 1)
    assert not any(service.is_locked(c) for c in ("a", "b", "c", "d"))
    assert service.login(PASSCODE, "e")


def test_callers_without_a_client_id_share_one_default_client(service):
    fail(service, None, MAX_FAILED_ATTEMPTS)
    assert service.is_locked() and service.is_locked(None)
    assert not service.is_locked("someone-else")


def test_many_fresh_sessions_hit_the_ceiling_over_all_clients(service, clock):
    for number in range(MAX_FAILED_ATTEMPTS_ALL_CLIENTS):
        service.login("wrong", f"session-{number}")                                                          # one failure each
    assert service.is_locked("brand-new") and service.login(PASSCODE, "brand-new") is None
    clock.now += LOCKOUT_SECONDS + 1
    assert service.login(PASSCODE, "brand-new")


def test_the_per_client_records_stay_bounded(service, clock):
    for number in range(MAX_TRACKED_CLIENTS * 3):
        service._record_failure(f"s{number}", clock.now)
        service._all_failures = 0                                                                            # leave the ceiling out of this
    assert len(service._failures) <= MAX_TRACKED_CLIENTS
    for number in range(MAX_TRACKED_CLIENTS * 3):
        service._locked_until[f"l{number}"] = clock.now + 1
        service._forget_expired(clock.now)
    assert len(service._locked_until) <= MAX_TRACKED_CLIENTS
    clock.now += 10
    service._forget_expired(clock.now)
    assert not service._locked_until


def test_a_disabled_service_never_signs_in(clock):
    off = AdminService(orchestrator_factory=lambda: None, passcode="", clock=clock)
    assert off.login("", "a") is None and off.login("x", "a") is None


# ---- authorization is unchanged -------------------------------------------------------------------------------------------------------

def test_admin_actions_need_a_session_whichever_client_signed_in(service, tmp_path):
    store = service._curriculum_store
    cid = store.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])["curriculum_id"]
    for token in (None, "", "forged", "x" * 43):
        with pytest.raises(AdminAuthError):
            service.approve_curriculum(token, cid, 1)

    token = service.login(PASSCODE, "a")
    service.logout(token)
    with pytest.raises(AdminAuthError):
        service.approve_curriculum(token, cid, 1)                                                            # a signed-out token is dead

    fail(service, "attacker", MAX_FAILED_ATTEMPTS)
    live = service.login(PASSCODE, "admin")
    assert service.approve_curriculum(live, cid, 1)["status"] == "approved"                                    # unaffected by the attacker's lockout


def test_a_sessions_expiry_still_applies(service, clock):
    token = service.login(PASSCODE, "a")
    assert service.is_valid(token)
    clock.now += admin_service.SESSION_IDLE_SECONDS + 1
    assert not service.is_valid(token)


def test_in_the_app_one_sessions_wrong_passcodes_do_not_lock_another_session(env):

    locked = start()
    click(locked, "go_admin")
    for _ in range(MAX_FAILED_ATTEMPTS):
        locked.text_input[0].input("wrong")
        locked.button[-1].click().run()
    assert any("Too many" in e.value or "locked" in e.value.lower() for e in locked.error), [e.value for e in locked.error]

    other = unlock(start())                                                                                   # a second browser session
    assert other.session_state.view == "admin" and other.session_state.admin_token
    assert locked.session_state.admin_client_id != other.session_state.admin_client_id
