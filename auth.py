"""Users, login sessions and role permissions.

Passwords are never stored: only a salted PBKDF2 hash. Sessions live in memory, so a server restart logs everyone out.

Usage:
  python auth.py --list                                   # show users and roles
  python auth.py --set-password admin                     # set or change a password (asks for it)
  python auth.py --add-user priya --name "Priya S" --role employee --team "Retail Banking" --assistants customer_faq
"""
import argparse
import getpass
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

USERS_FILE = Path("control/users.json")
ASSISTANTS_FILE = Path("control/assistants.json")
SESSION_HOURS = 8

# What each role may do. Admin approves; engineers build but cannot approve their own work.
PERMISSIONS = {
    "employee": {"use_assistant"},
    "ai_engineer": {"use_assistant", "view_platform", "register_dataset", "prepare_dataset", "request_run",
                    "register_model", "create_assistant"},
    "admin": {"use_assistant", "view_platform", "register_dataset", "prepare_dataset", "request_run",
              "register_model", "create_assistant", "approve_dataset", "review_model", "review_assistant",
              "promote", "rollback", "manage_users"},
}

sessions = {}  # token -> {"username": ..., "expires": ...}

# Abuse protection. Counters live in memory (a restart resets them); production would keep them in Redis.
MAX_FAILED_LOGINS = 5          # wrong passwords allowed per username ...
LOCKOUT_SECONDS = 15 * 60      # ... within this window before the account is locked for the rest of it
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "20"))  # questions per signed-in session per minute
# All sessions of one account together, so signing in many times does not multiply the limit
ACCOUNT_REQUESTS_PER_MINUTE = int(os.getenv("ACCOUNT_REQUESTS_PER_MINUTE", str(5 * REQUESTS_PER_MINUTE)))
# Applications share the app key, so they get their own, higher limit
APP_REQUESTS_PER_MINUTE = int(os.getenv("APP_REQUESTS_PER_MINUTE", "120"))
failed_logins = {}             # username -> times of recent wrong passwords
recent_requests = {}           # caller -> times of recent gateway calls


class TooManyAttempts(ValueError):
    """Raised when a login or request limit is hit (the API turns it into HTTP 429)."""


def recent(times, window):
    now = time.time()
    return [t for t in times if now - t < window]


def allow_requests(limits):
    """Sliding one-minute windows: limits maps a key to its limit. Every key must have room before any is counted."""
    windows = {key: recent(recent_requests.get(key, []), 60) for key in limits}
    for key, limit in limits.items():
        if len(windows[key]) >= limit:
            raise TooManyAttempts(f"Rate limit reached: {limit} requests per minute. Please wait and try again.")
    now = time.time()
    for key, times in windows.items():
        recent_requests[key] = times + [now]


def allow_request(caller, limit=REQUESTS_PER_MINUTE):
    """One sliding one-minute window per caller; raises TooManyAttempts when the limit is reached."""
    allow_requests({caller: limit})


def allow_person_request(username, token):
    # Each sign-in has its own limit, so people sharing a demo account do not block each other,
    # and the account as a whole has a ceiling. Sessions are keyed by a hash, not the raw token.
    session = "session:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    allow_requests({session: REQUESTS_PER_MINUTE, "account:" + username: ACCOUNT_REQUESTS_PER_MINUTE})


def allow_app_request():
    allow_requests({"application": APP_REQUESTS_PER_MINUTE})


def load_users():
    return json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else []


def save_users(users):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2) + "\n")


def load_assistants():
    return json.loads(ASSISTANTS_FILE.read_text())


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def find_user(username):
    for user in load_users():
        if user["username"] == username:
            return user
    return None


def public(user):
    """The user record without password fields, safe to send to the browser."""
    info = {k: v for k, v in user.items() if k not in ("password_hash", "salt")}
    info["permissions"] = sorted(PERMISSIONS[user["role"]])
    return info


def login(username, password):
    # Lock the username after too many wrong passwords, so passwords can't be guessed by trying many
    failures = recent(failed_logins.get(username, []), LOCKOUT_SECONDS)
    if len(failures) >= MAX_FAILED_LOGINS:
        raise TooManyAttempts("Too many failed sign-in attempts. Try again in 15 minutes.")

    user = find_user(username)
    # Same error for unknown user and wrong password, so the response doesn't reveal which usernames exist
    if not user or not user.get("password_hash") or \
            not secrets.compare_digest(hash_password(password, user["salt"]), user["password_hash"]):
        failed_logins[username] = failures + [time.time()]
        raise ValueError("Invalid username or password")

    failed_logins.pop(username, None)
    token = secrets.token_urlsafe(32)
    sessions[token] = {"username": username, "expires": time.time() + SESSION_HOURS * 3600}
    return token, public(user)


def logout(token):
    sessions.pop(token, None)


def user_for_token(token):
    session = sessions.get(token)
    if not session or session["expires"] < time.time():
        sessions.pop(token, None)
        return None
    user = find_user(session["username"])
    return public(user) if user else None


def can(user, action):
    return action in PERMISSIONS[user["role"]]


def can_use_assistant(user, assistant_id):
    # Admins and engineers can test every assistant; employees only the ones assigned to them
    return user["role"] in ("admin", "ai_engineer") or assistant_id in user["assistants"]


def add_user(username, name, role, team, assistants, password=None):
    if role not in PERMISSIONS:
        raise ValueError(f"Role must be one of {sorted(PERMISSIONS)}")
    unknown = set(assistants) - set(load_assistants())
    if unknown:
        raise ValueError(f"Unknown assistants: {sorted(unknown)}")
    users = load_users()
    if any(u["username"] == username for u in users):
        raise ValueError(f"User {username} already exists")
    user = {"username": username, "name": name, "role": role, "team": team, "assistants": assistants,
            "password_hash": None, "salt": None}
    save_users(users + [user])
    if password:
        set_password(username, password)
    return public(user)


def set_password(username, password):
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    users = load_users()
    for user in users:
        if user["username"] == username:
            user["salt"] = secrets.token_hex(16)
            user["password_hash"] = hash_password(password, user["salt"])
            save_users(users)
            return
    raise ValueError(f"Unknown user: {username}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set-password", metavar="USERNAME")
    parser.add_argument("--add-user", metavar="USERNAME")
    parser.add_argument("--name")
    parser.add_argument("--role", choices=sorted(PERMISSIONS))
    parser.add_argument("--team", default="")
    parser.add_argument("--assistants", nargs="*", default=[])
    args = parser.parse_args()

    if args.add_user:
        add_user(args.add_user, args.name or args.add_user, args.role, args.team, args.assistants)
        args.set_password = args.add_user
    if args.set_password:
        set_password(args.set_password, getpass.getpass(f"New password for {args.set_password}: "))
        print("Password set.")
    if args.list or not (args.add_user or args.set_password):
        for user in load_users():
            has_password = "yes" if user["password_hash"] else "NO"
            print(f"{user['username']:12} {user['role']:12} {user['team']:24} assistants={user['assistants']} password set: {has_password}")
