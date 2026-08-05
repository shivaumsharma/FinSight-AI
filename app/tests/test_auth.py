"""
Unit tests for app/api/auth.py (password hashing) and the users/
sessions functions in app/api/db.py.

No network, no models -- pure stdlib crypto and SQLite against a
per-test temp DB, so these run fast and belong in CI.
"""

import time

import pytest

from app.api import auth, db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


# ---------------------------------------------------------------- password hashing

def test_verify_password_accepts_the_correct_password():
    password_hash, salt = auth.hash_password("correcthorsebatterystaple")
    assert auth.verify_password("correcthorsebatterystaple", password_hash, salt)


def test_verify_password_rejects_the_wrong_password():
    password_hash, salt = auth.hash_password("correcthorsebatterystaple")
    assert not auth.verify_password("wrong password", password_hash, salt)


def test_hash_password_uses_a_fresh_salt_every_call():
    hash1, salt1 = auth.hash_password("same password")
    hash2, salt2 = auth.hash_password("same password")
    # Same input password, different salt -> different hash. If this
    # ever failed, two users choosing the same password would have
    # identical rows in the users table -- a real information leak.
    assert salt1 != salt2
    assert hash1 != hash2


def test_verify_password_rejects_a_hash_from_a_different_password():
    password_hash, salt = auth.hash_password("password-a")
    other_hash, other_salt = auth.hash_password("password-b")
    # Cross-checking password-a against password-b's own hash/salt
    # pair must fail even though both are individually valid pairs.
    assert not auth.verify_password("password-a", other_hash, other_salt)


# ---------------------------------------------------------------- sessions

def test_create_session_round_trips_to_the_right_user(temp_db):
    user_id = temp_db.create_user("alice@example.com", "hash", "salt")
    token = temp_db.create_session(user_id)
    assert temp_db.get_session(token) == user_id


def test_get_session_returns_none_for_an_unknown_token(temp_db):
    assert temp_db.get_session("this-token-was-never-issued") is None


def test_get_session_returns_none_for_an_expired_token(temp_db):
    user_id = temp_db.create_user("alice@example.com", "hash", "salt")
    token = temp_db.create_session(user_id, ttl_seconds=-1)
    assert temp_db.get_session(token) is None


def test_delete_session_invalidates_the_token(temp_db):
    user_id = temp_db.create_user("alice@example.com", "hash", "salt")
    token = temp_db.create_session(user_id)
    temp_db.delete_session(token)
    assert temp_db.get_session(token) is None


def test_delete_session_on_an_unknown_token_is_a_silent_no_op(temp_db):
    temp_db.delete_session("never-issued")  # must not raise


def test_a_user_can_hold_two_independent_sessions(temp_db):
    # session_token, not user_id, is the sessions table's primary key
    # (see db.py's own comment on why) -- two logins for the same user
    # must both stay valid, and deleting one must not affect the other.
    user_id = temp_db.create_user("alice@example.com", "hash", "salt")
    token1 = temp_db.create_session(user_id)
    token2 = temp_db.create_session(user_id)

    temp_db.delete_session(token1)

    assert temp_db.get_session(token1) is None
    assert temp_db.get_session(token2) == user_id


# ---------------------------------------------------------------- users

def test_get_user_by_email_returns_none_when_not_registered(temp_db):
    assert temp_db.get_user_by_email("nobody@example.com") is None


def test_create_user_raises_on_duplicate_email(temp_db):
    import sqlite3

    temp_db.create_user("alice@example.com", "hash", "salt")
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.create_user("alice@example.com", "different-hash", "different-salt")


# ---------------------------------------------------------------- rate limiting (count_recent_jobs)

def test_count_recent_jobs_only_counts_within_the_window(temp_db):
    user_id = temp_db.create_user("alice@example.com", "hash", "salt")
    temp_db.create_job(question="q1", orchestrator="hand_rolled", user_id=user_id)

    assert temp_db.count_recent_jobs(user_id, since=time.time() - 3600) == 1
    assert temp_db.count_recent_jobs(user_id, since=time.time() + 3600) == 0


def test_count_recent_jobs_does_not_count_other_users(temp_db):
    alice = temp_db.create_user("alice@example.com", "hash", "salt")
    bob = temp_db.create_user("bob@example.com", "hash", "salt")
    temp_db.create_job(question="q1", orchestrator="hand_rolled", user_id=alice)

    assert temp_db.count_recent_jobs(bob, since=time.time() - 3600) == 0


# ---------------------------------------------------------------- signed PDF share links

def test_verify_pdf_signature_accepts_a_freshly_signed_url():
    expires_at, signature = auth.sign_pdf_url("job-123", ttl_seconds=3600)
    assert auth.verify_pdf_signature("job-123", expires_at, signature)


def test_verify_pdf_signature_rejects_a_mismatched_job_id():
    expires_at, signature = auth.sign_pdf_url("job-123", ttl_seconds=3600)
    assert not auth.verify_pdf_signature("job-999", expires_at, signature)


def test_verify_pdf_signature_rejects_an_expired_link():
    expires_at, signature = auth.sign_pdf_url("job-123", ttl_seconds=-10)
    assert not auth.verify_pdf_signature("job-123", expires_at, signature)


def test_verify_pdf_signature_rejects_a_tampered_signature():
    expires_at, signature = auth.sign_pdf_url("job-123", ttl_seconds=3600)
    flipped_last_char = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    assert not auth.verify_pdf_signature("job-123", expires_at, flipped_last_char)


def test_verify_pdf_signature_rejects_an_extended_expiry():
    # The real attack this guards against: a link recipient editing the
    # exp query param upward to grant themselves more time than the
    # owner actually shared, while reusing the original signature.
    expires_at, signature = auth.sign_pdf_url("job-123", ttl_seconds=3600)
    assert not auth.verify_pdf_signature("job-123", expires_at + 100_000, signature)
