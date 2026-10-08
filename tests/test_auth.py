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
    auth.add_user("rev", "Rev", "reviewer", "Evaluation", [], "correct-horse-4")


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


def test_reviewer_can_view_and_ask_but_not_change_anything():
    reviewer = auth.public(auth.find_user("rev"))
    assert auth.can(reviewer, "view_platform")
    assert auth.can(reviewer, "use_assistant")
    assert auth.can_use_assistant(reviewer, "customer_faq")
    for permission in ["register_dataset", "prepare_dataset", "request_run", "register_model", "create_assistant",
                       "approve_dataset", "review_model", "review_assistant", "promote", "rollback", "manage_users"]:
        assert not auth.can(reviewer, permission)


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


def test_each_session_has_its_own_limit(monkeypatch):
    # Two people signed in to the same account do not block each other
    auth.recent_requests.clear()
    monkeypatch.setattr(auth, "REQUESTS_PER_MINUTE", 2)
    monkeypatch.setattr(auth, "ACCOUNT_REQUESTS_PER_MINUTE", 10)
    for _ in range(2):
        auth.allow_person_request("reviewer", "token-a")
    with pytest.raises(auth.TooManyAttempts):
        auth.allow_person_request("reviewer", "token-a")
    auth.allow_person_request("reviewer", "token-b")


def test_account_ceiling_covers_all_sessions(monkeypatch):
    auth.recent_requests.clear()
    monkeypatch.setattr(auth, "REQUESTS_PER_MINUTE", 5)
    monkeypatch.setattr(auth, "ACCOUNT_REQUESTS_PER_MINUTE", 3)
    for token in ("t1", "t2", "t3"):
        auth.allow_person_request("reviewer", token)
    with pytest.raises(auth.TooManyAttempts):
        auth.allow_person_request("reviewer", "t4")
    auth.allow_person_request("engineer", "t5")  # other accounts are unaffected


def test_blocked_request_is_not_counted(monkeypatch):
    # A refused request must not use up room in the other window
    auth.recent_requests.clear()
    monkeypatch.setattr(auth, "REQUESTS_PER_MINUTE", 1)
    monkeypatch.setattr(auth, "ACCOUNT_REQUESTS_PER_MINUTE", 2)
    auth.allow_person_request("reviewer", "t1")
    with pytest.raises(auth.TooManyAttempts):
        auth.allow_person_request("reviewer", "t1")
    auth.allow_person_request("reviewer", "t2")


def test_applications_have_their_own_limit(monkeypatch):
    auth.recent_requests.clear()
    monkeypatch.setattr(auth, "APP_REQUESTS_PER_MINUTE", 2)
    auth.allow_app_request()
    auth.allow_app_request()
    with pytest.raises(auth.TooManyAttempts):
        auth.allow_app_request()
    auth.allow_person_request("reviewer", "t1")  # people are counted separately


def test_short_password_is_refused():
    with pytest.raises(ValueError):
        auth.set_password("ana", "short")


def test_unknown_role_is_refused():
    with pytest.raises(ValueError):
        auth.add_user("x", "X", "superuser", "", [])


def add_guest_reviewer(role="reviewer"):
    # Like control/users.json: the guest account has no password
    auth.add_user(auth.GUEST_USERNAME, "Guest Reviewer", role, "Evaluation", [])


def test_guest_gets_a_read_only_session():
    add_guest_reviewer()
    token, user = auth.guest_login()
    assert auth.user_for_token(token)["role"] == "reviewer"
    assert set(user["permissions"]) == {"use_assistant", "view_platform"}


def test_guest_account_cannot_sign_in_with_a_password():
    add_guest_reviewer()
    with pytest.raises(ValueError):
        auth.login(auth.GUEST_USERNAME, "")


def test_guest_access_never_hands_out_a_role_that_can_change_things():
    add_guest_reviewer(role="admin")
    with pytest.raises(PermissionError):
        auth.guest_login()


def test_guest_access_can_be_turned_off(monkeypatch):
    add_guest_reviewer()
    monkeypatch.setattr(auth, "GUEST_ACCESS", False)
    with pytest.raises(PermissionError):
        auth.guest_login()


def test_guest_sign_ins_are_rate_limited(monkeypatch):
    add_guest_reviewer()
    monkeypatch.setattr(auth, "GUEST_LOGINS_PER_MINUTE", 2)
    auth.recent_requests.pop("guest_login", None)
    auth.guest_login()
    auth.guest_login()
    with pytest.raises(auth.TooManyAttempts):
        auth.guest_login()
    auth.recent_requests.pop("guest_login", None)
