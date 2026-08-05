"""
generate_vapid_keys.py

One-off utility: generates a real, persistent VAPID keypair for Web
Push notifications (app/api/auth.py, app/api/jobs.py's
_notify_job_complete). Mirrors the pattern already used for API_KEY/
PDF_SHARE_SECRET (`python -c "import secrets; print(secrets.token_urlsafe(32))"`
in .env.example) -- just for a real EC keypair instead of a random
token, since VAPID needs an asymmetric key, not a symmetric secret.

Without a real configured VAPID_PRIVATE_KEY, app/api/auth.py generates
a fresh keypair every process start -- fine for local dev, but every
browser subscription registered against one process's public key
becomes permanently unusable the moment that process restarts (see
auth.py's own comment on why this is a bigger deal than
PDF_SHARE_SECRET's equivalent fallback). Run this once per deployment,
not per developer machine, and keep the private key secret the same
way any other credential in .env is kept secret.

Usage:
    python scripts/generate_vapid_keys.py
"""

from py_vapid import Vapid


def main():
    vapid = Vapid()
    vapid.generate_keys()

    private_pem = vapid.private_pem().decode("utf-8")
    escaped_pem = private_pem.replace("\n", "\\n")

    print("Add these to your .env (or your deployment's env var config):")
    print()
    # Double-quoted, not bare -- python-dotenv only unescapes \n back
    # into a real newline inside a quoted value (verified directly:
    # an unquoted value keeps the literal two-character "\n" and
    # Vapid.from_pem() fails to parse it as PEM).
    print(f'VAPID_PRIVATE_KEY="{escaped_pem}"')
    print()
    print("# A real address the push services can contact about this sender.")
    print("VAPID_CONTACT_EMAIL=you@example.com")
    print()
    print("(VAPID_PUBLIC_KEY does not need to be set separately -- app/api/auth.py")
    print(" derives it from VAPID_PRIVATE_KEY and serves it at GET /v1/push/vapid-public-key.)")


if __name__ == "__main__":
    main()
