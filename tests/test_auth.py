import time

import pytest

import auth


@pytest.fixture(autouse=True)
def temporary_users(tmp_path, monkeypatch):
    # Work on a temporary users file so tests never change the real accounts
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    auth.sessions.clear()
    auth.add_user("ana", "Ana", "employee", "Support", ["customer_faq"], "correct-horse-1")
    auth.add_user("eng", "Eng", "ai_engineer", "AI/ML", [], "correct-horse-2")
    auth.add_user("boss", "Boss", "admin", "Governance", [], "correct-horse-3")


def test_password_is_stored_as_hash_only():
    stored = auth.find_user("ana")
    assert "correct-horse-1" not in str(stored)
    assert len(stored["password_hash"]) == 64


def test_login_with_right_password_gives_session():
    token, user = auth.login("ana", "correct-horse-1")
    assert auth.user_for_token(token)["username"] == "ana"
    assert "password_hash" not in user


def test_wrong_password_and_unknown_user_give_same_error():
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login("ana", "wrong-password")
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login("nobody", "whatever-password")


def test_logout_ends_session():
    token, _ = auth.login("ana", "correct-horse-1")
    auth.logout(token)
    assert auth.user_for_token(token) is None


def test_expired_session_is_refused():
    token, _ = auth.login("ana", "correct-horse-1")
    auth.sessions[token]["expires"] = time.time() - 1
    assert auth.user_for_token(token) is None


def test_employee_can_only_use_assistant():
    employee = auth.public(auth.find_user("ana"))
    assert auth.can(employee, "use_assistant")
    assert not auth.can(employee, "view_platform")
    assert not auth.can(employee, "approve_dataset")


def test_engineer_builds_but_cannot_approve():
    engineer = auth.public(auth.find_user("eng"))
    assert auth.can(engineer, "request_run")
    assert auth.can(engineer, "register_model")
    assert not auth.can(engineer, "review_model")
    assert not auth.can(engineer, "promote")
    assert not auth.can(engineer, "approve_dataset")


def test_admin_can_approve_and_deploy():
    admin = auth.public(auth.find_user("boss"))
    for permission in ["approve_dataset", "review_model", "promote", "rollback", "manage_users"]:
        assert auth.can(admin, permission)


def test_employee_only_uses_assigned_assistants():
    employee = auth.public(auth.find_user("ana"))
    assert auth.can_use_assistant(employee, "customer_faq")
    assert not auth.can_use_assistant(employee, "loan_assistant")


def test_account_locks_after_five_wrong_passwords():
    auth.failed_logins.clear()
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(ValueError, match="Invalid"):
            auth.login("ana", "wrong-password")
    # Even the right password is refused while locked
    with pytest.raises(auth.TooManyAttempts):
        auth.login("ana", "correct-horse-1")


def test_successful_login_resets_failed_count():
    auth.failed_logins.clear()
    for _ in range(auth.MAX_FAILED_LOGINS - 1):
        with pytest.raises(ValueError):
            auth.login("ana", "wrong-password")
    auth.login("ana", "correct-horse-1")
    assert "ana" not in auth.failed_logins


def test_rate_limit_blocks_after_limit():
    auth.recent_requests.clear()
    for _ in range(3):
        auth.allow_request("ana", limit=3)
    with pytest.raises(auth.TooManyAttempts):
        auth.allow_request("ana", limit=3)
    auth.allow_request("eng", limit=3)  # other callers have their own limit


def test_short_password_is_refused():
    with pytest.raises(ValueError):
        auth.set_password("ana", "short")


def test_unknown_role_is_refused():
    with pytest.raises(ValueError):
        auth.add_user("x", "X", "superuser", "", [])
