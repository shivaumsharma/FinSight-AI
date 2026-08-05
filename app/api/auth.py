"""
auth.py

Password hashing/verification and the get_current_user FastAPI
dependency for real per-user auth -- layered on top of, not instead
of, the existing shared X-API-Key gate in main.py (see that gate's own
comment for why both exist: X-API-Key protects this deployment's
Finnhub/Groq quota from anyone on the internet; this file identifies
*which* user is making an already-authorized-to-the-deployment call).

Stdlib only, deliberately no bcrypt/passlib dependency -- PBKDF2-HMAC
via hashlib is a real, slow, salted KDF suitable for password hashing
on its own; adding a new dependency for something the standard library
already does correctly would go against this project's existing
pattern (stdlib sqlite3 instead of an ORM, requests instead of an SEC
EDGAR SDK).
"""

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional, Tuple

from fastapi import Header
from py_vapid import Vapid

from app.api import db, errors

# 600,000 rounds matches current OWASP guidance for PBKDF2-HMAC-SHA256
# (as of their latest published recommendation) -- high enough to make
# offline brute-forcing a stolen hash expensive, low enough that one
# login request still completes in well under a second.
_PBKDF2_ITERATIONS = 600_000
_HASH_ALGORITHM = "sha256"

# Falls back to a random per-process secret when unset, same
# "empty/unset degrades gracefully, but you should really set this in
# production" pattern main.py's own _API_KEY uses -- share-link signing
# still works with zero config for local dev, at the cost of every
# previously-issued link becoming invalid on the next restart if a real
# value was never explicitly configured (documented in .env.example).
_PDF_SHARE_SECRET = os.environ.get("PDF_SHARE_SECRET") or secrets.token_urlsafe(32)

# Same fallback pattern as _PDF_SHARE_SECRET above, adapted for an
# asymmetric EC keypair instead of a random token: VAPID_PRIVATE_KEY
# (a PEM string) if set, otherwise a fresh keypair generated once per
# process. Push notifications still work with zero config for local
# dev -- but unlike a symmetric secret, restarting without a real
# configured key doesn't just invalidate *new* signing: every browser
# subscription registered against the old process's public key becomes
# permanently unusable (the new process's private key can't produce a
# signature that subscription's public key would trust), not just
# links issued going forward. Generate a real keypair once
# (scripts/generate_vapid_keys.py) for anything beyond local dev.
_VAPID_PRIVATE_KEY_PEM = os.environ.get("VAPID_PRIVATE_KEY")
if _VAPID_PRIVATE_KEY_PEM:
    # Most platform env-var UIs (Cloud Run, Railway, Vercel...) have no
    # way to enter a real multi-line value, so this is stored with
    # literal "\n" escapes (see scripts/generate_vapid_keys.py's
    # output) and unescaped here. A .env file loaded via python-dotenv
    # already unescapes this itself when the value is double-quoted,
    # so this is a no-op in that path -- not a second, conflicting
    # unescape.
    _vapid = Vapid.from_pem(_VAPID_PRIVATE_KEY_PEM.replace("\\n", "\n").encode("utf-8"))
else:
    _vapid = Vapid()
    _vapid.generate_keys()

VAPID_CONTACT_EMAIL = os.environ.get("VAPID_CONTACT_EMAIL", "sshivaum@gmail.com")


def _vapid_public_key_b64url() -> str:
    """The browser-facing form of the VAPID public key:
    base64url-encoded, uncompressed EC point -- the exact shape
    PushManager.subscribe()'s applicationServerKey option requires.
    Not the same encoding as the PEM py_vapid itself works with
    internally."""
    from cryptography.hazmat.primitives import serialization

    raw = _vapid.public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


VAPID_PUBLIC_KEY = _vapid_public_key_b64url()


def send_push_notification(subscription: dict, title: str, body: str, status: str) -> None:
    """Sends one Web Push message. Raises WebPushException on failure
    (including an expired/invalid subscription reported as 404/410) --
    callers (jobs.py's _notify_job_complete) decide what a failure
    means (prune the subscription vs. just log it), this function's
    only job is the actual send."""
    import json

    from pywebpush import webpush

    webpush(
        subscription_info={
            "endpoint": subscription["endpoint"],
            "keys": {"p256dh": subscription["p256dh"], "auth": subscription["auth_key"]},
        },
        data=json.dumps({"title": title, "body": body, "status": status}),
        vapid_private_key=_vapid,
        vapid_claims={"sub": f"mailto:{VAPID_CONTACT_EMAIL}"},
    )


def hash_password(password: str) -> tuple[str, str]:
    """Returns (password_hash_hex, salt_hex). A fresh random salt every
    call means two users with the same password get different hashes
    -- salt is stored alongside the hash (db.py's users.salt column),
    not derived from anything, so verify_password needs it passed back in."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(_HASH_ALGORITHM, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time comparison (hmac.compare_digest) -- a naive `==`
    on the two hex strings would leak how many leading bytes matched
    via response-time differences, a real (if narrow) timing side
    channel for exactly the kind of secret this function exists to
    protect."""
    candidate = hashlib.pbkdf2_hmac(
        _HASH_ALGORITHM, password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate.hex(), password_hash)


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Shared by get_current_user (below) and main.py's logout endpoint
    -- logout needs the raw token (to delete that one session), not the
    user_id get_current_user resolves it to, but both start from the
    same "is this even a well-formed Bearer header" check, so that
    check lives in exactly one place."""
    if not authorization or not authorization.startswith("Bearer "):
        raise errors.unauthorized()
    return authorization.removeprefix("Bearer ").strip()


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency -- raises a 401 APIError (caught by main.py's
    existing api_error_handler, same structured-error shape as every
    other failure mode) if there's no valid session. `Authorization:
    Bearer <token>` rather than a custom header name, matching the
    conventional REST pattern rather than inventing FinSight's own.
    """
    token = extract_bearer_token(authorization)
    user_id = db.get_session(token)
    if user_id is None:
        raise errors.unauthorized()

    return user_id


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Same resolution as get_current_user, but returns None instead of
    raising when there's no valid session -- for endpoints (the PDF
    endpoint's signed-link fallback, see main.py's get_research_pdf)
    that have a second, independent way to authorize a request rather
    than unconditionally requiring a session."""
    try:
        token = extract_bearer_token(authorization)
    except errors.APIError:
        return None
    return db.get_session(token)


def sign_pdf_url(job_id: str, ttl_seconds: int) -> Tuple[int, str]:
    """Returns (expires_at_unix_ts, signature_hex) for a shareable PDF
    link -- see main.py's POST /v1/research/{job_id}/pdf/share. The
    expiry is signed INTO the message, not just checked separately
    after the fact -- binding it into the HMAC is what stops a link
    recipient from editing the exp query param to grant themselves
    more time than the owner actually shared."""
    expires_at = int(time.time()) + ttl_seconds
    message = f"{job_id}:{expires_at}".encode("utf-8")
    signature = hmac.new(_PDF_SHARE_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return expires_at, signature


def verify_pdf_signature(job_id: str, expires_at: int, signature: str) -> bool:
    """True only if `signature` is a valid HMAC for this exact
    job_id/expires_at pair AND that expiry hasn't passed yet -- both
    conditions matter: a valid-but-expired signature must fail the
    same as a forged one, not a separate error path a caller could
    forget to check."""
    if expires_at < time.time():
        return False
    message = f"{job_id}:{expires_at}".encode("utf-8")
    expected = hmac.new(_PDF_SHARE_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
