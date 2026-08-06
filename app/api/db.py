"""
db.py

SQLite-backed job store for the research API. Deliberately stdlib
sqlite3, no ORM -- the schema is small and simple enough not to need
one, and it keeps this pass's dependency footprint at zero beyond
fastapi/uvicorn themselves.

Why SQLite over an in-memory dict (the thing this file specifically
exists to NOT be): a user starts a query, backgrounds their client,
comes back minutes later expecting their report -- if the process
that was running the job restarts or redeploys in between, an
in-memory job store loses the row entirely and the client polls a
job_id that never existed as far as the new process is concerned.
SQLite survives that. It does NOT, on its own, make a `running` row
after a restart mean anything -- see reconcile_interrupted_jobs()
below for why that needs its own explicit handling, not just
persistence.

Thread safety: every function here opens its own short-lived
connection rather than sharing one connection object across threads
(the FastAPI request thread and the ThreadPoolExecutor worker thread
in jobs.py both touch this module) -- a bare sqlite3.Connection is not
safe to hand between threads by default, and reusing one here would
be a five-minute bug the first time two threads touch it at once.
"""

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from app.core.paths import DB_PATH  # noqa: F401 -- re-exported; see app/core/paths.py

# How long a job may sit in "running" before the periodic sweep (see
# main.py's background thread) treats it as hung and marks it TIMEOUT.
# Documented limitation: this stops the ROW from lying to the client;
# it does not, and cannot, forcibly kill the Python thread actually
# running the job -- that needs a ProcessPoolExecutor (killable) or
# hosted inference removing the long local blocking call, neither of
# which is in this pass. See jobs.py's module docstring.
JOB_TIMEOUT_SECONDS = 600

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

ERROR_INTERRUPTED = "INTERRUPTED"
ERROR_TIMEOUT = "TIMEOUT"

# Default session lifetime and job rate limit -- both env-configurable
# (see app/api/main.py) so a deployment can tighten/loosen either
# without a code change. 30 days matches typical "stay logged in"
# web-app behavior; 20/day/user is generous for real use while still
# bounding one compromised account's Groq/Finnhub spend.
DEFAULT_SESSION_TTL_SECONDS = 30 * 24 * 3600
DEFAULT_DAILY_JOB_LIMIT = 20


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates the jobs/users/sessions tables if they don't exist. Safe
    to call on every startup -- CREATE TABLE IF NOT EXISTS is
    idempotent."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                status TEXT NOT NULL,
                question TEXT NOT NULL,
                orchestrator TEXT NOT NULL,
                result_json TEXT,
                pdf_path TEXT,
                error_code TEXT,
                error_message TEXT,
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        # Denormalized onto jobs rather than a separate report_summaries
        # table -- these three are updated in the same UPDATE statement
        # mark_done() already issues (same result dict already in
        # scope), so there's no second write and no risk of the two
        # drifting out of sync. sqlite3 has no ADD COLUMN IF NOT EXISTS,
        # so each gets its own try/except: one genuine failure (disk
        # full, locked file) must not silently skip the other two the
        # way a single wrapping try/except would.
        for column in ("ticker", "company_name", "rating"):
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
        # password_hash/salt are separate columns, not one combined
        # string -- keeps app/api/auth.py's PBKDF2 hash/verify
        # symmetric (hash_password returns exactly what verify_password
        # needs, no ad-hoc delimiter-splitting on read).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # session_token is the primary key, not user_id -- a user can
        # hold multiple valid sessions at once (e.g. two browser tabs
        # each logging in independently), and logout must invalidate
        # only the one session being logged out of, not every session
        # that user has.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        # endpoint (the browser push service's own per-subscription
        # URL) is the primary key, not user_id -- one user can hold
        # several subscriptions at once (multiple devices/browsers),
        # each independently valid/prunable. auth_key, not `auth` --
        # avoids any confusion with this module's own users/sessions
        # concept of "auth", even though `auth` isn't a reserved SQLite
        # word.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                endpoint TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                p256dh TEXT NOT NULL,
                auth_key TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # (user_id, ticker) as a composite primary key, not a surrogate
        # id -- a user can only watch a given ticker once, and that's
        # exactly what the primary key should enforce; add_watchlist_item
        # relies on this for its INSERT OR IGNORE idempotency.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                added_at REAL NOT NULL,
                PRIMARY KEY (user_id, ticker)
            )
            """
        )


# ---------------------------------------------------------------- users/sessions
#
# Password hashing itself lives in app/api/auth.py, not here -- this
# module stays a pure data-access layer (same separation jobs.py/db.py
# already have: jobs.py owns orchestration, db.py owns persistence).

def create_user(email: str, password_hash: str, salt: str) -> str:
    """Raises sqlite3.IntegrityError if email is already registered
    (UNIQUE constraint) -- callers (app/api/main.py's signup endpoint)
    catch that and translate it to the EMAIL_ALREADY_REGISTERED API
    error, rather than this layer knowing about HTTP status codes."""
    user_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, email, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, password_hash, salt, time.time()),
        )
    return user_id


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row is not None else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """For the Profile view -- a session only carries user_id, not
    email, so displaying account details needs this lookup by id
    (get_user_by_email above is keyed the other direction, for login)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row is not None else None


def create_session(user_id: str, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> str:
    # token_urlsafe(32) -- 256 bits of entropy, the same order of
    # magnitude Django/Flask's session-token generators use; a bare
    # uuid4 (122 bits) would still be fine here but this is the
    # standard-library-recommended primitive for exactly this purpose.
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + ttl_seconds),
        )
    return token


def get_session(token: str) -> Optional[str]:
    """Returns the session's user_id, or None if the token doesn't
    exist OR has expired -- callers (app/api/auth.py's
    get_current_user dependency) don't need to distinguish the two;
    both mean "not logged in" from the caller's point of view."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE session_token=?", (token,)
        ).fetchone()
    if row is None or row["expires_at"] < time.time():
        return None
    return row["user_id"]


def delete_session(token: str) -> None:
    """Logout. A no-op if the token doesn't exist -- calling this
    twice, or on an already-expired token, isn't an error."""
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE session_token=?", (token,))


def count_recent_jobs(user_id: str, since: float) -> int:
    """Number of jobs `user_id` has started since the unix timestamp
    `since` -- the entire rate-limiting mechanism (see main.py's
    DAILY_JOB_LIMIT check) is this one query against the jobs table
    that already exists; no separate usage-tracking table needed."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE user_id=? AND started_at > ?", (user_id, since)
        ).fetchone()
    return row[0]


# ---------------------------------------------------------------- push subscriptions

def add_push_subscription(user_id: str, endpoint: str, p256dh: str, auth_key: str) -> None:
    """INSERT OR REPLACE, not a duplicate-key error -- a browser
    calling pushManager.subscribe() again for a subscription it
    already holds (e.g. the user clicks "Enable notifications" twice,
    or a service worker re-registers) should just refresh the row, not
    fail."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO push_subscriptions (endpoint, user_id, p256dh, auth_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (endpoint, user_id, p256dh, auth_key, time.time()),
        )


def get_push_subscriptions(user_id: str) -> list:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM push_subscriptions WHERE user_id=?", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def remove_push_subscription(endpoint: str) -> None:
    """Used both for an explicit user-initiated unsubscribe AND to
    prune a subscription the push service itself has reported dead
    (404/410 -- see jobs.py's _notify_job_complete) -- a no-op either
    way if the endpoint is already gone, same as delete_session above."""
    with _connect() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE endpoint=?", (endpoint,))


def create_job(question: str, orchestrator: str, user_id: str = "anonymous") -> str:
    """
    Starts as PENDING, not RUNNING -- with a single worker
    (ThreadPoolExecutor(max_workers=1) in jobs.py), a submitted job may
    sit queued behind another one for a while before it actually
    starts. Claiming RUNNING at creation time would be a real lie about
    state, not just an implementation detail: it would make a queued
    job's row indistinguishable from one whose worker is genuinely
    processing it. See mark_running() for the actual transition.
    """
    job_id = str(uuid.uuid4())
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, user_id, status, question, orchestrator, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, user_id, STATUS_PENDING, question, orchestrator, now, now),
        )
    return job_id


def mark_running(job_id: str) -> None:
    """
    Called by jobs.py the moment a worker thread actually picks up a
    job -- also resets started_at to this moment, not job creation
    time, so JOB_TIMEOUT_SECONDS measures actual execution time, not
    time spent queued behind another job.
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, started_at=?, updated_at=? WHERE job_id=? AND status=?",
            (STATUS_RUNNING, time.time(), time.time(), job_id, STATUS_PENDING),
        )


def mark_done(job_id: str, result: Dict[str, Any], pdf_path: str) -> None:
    """
    Only takes effect if the row is still `running` -- if it was
    already reconciled away (timeout/interrupted) by the time this is
    called, that means this update is coming from a zombie thread that
    kept running after its row was marked failed (see
    JOB_TIMEOUT_SECONDS's documented limitation above). The WHERE
    clause makes that a silent no-op instead of a late result quietly
    overwriting an error status the client may have already seen and
    acted on.

    ticker/company_name/rating are pulled out of `result` here (same
    .get()-chain path jobs.py's _notify_job_complete already reads for
    push-notification text) and denormalized onto the row so
    list_recent_jobs/get_latest_rating_for_ticker don't need to
    json.loads() every row's full result_json just to list them. Uses
    .get() throughout, never direct indexing -- a malformed/partial
    result dict must still let this job complete successfully with
    NULL columns, not raise and lose an otherwise-good result.
    """
    ticker = result.get("ticker")
    report_data = result.get("report_data") or {}
    company_name = (report_data.get("company_overview") or {}).get("name")
    rating = (report_data.get("recommendation") or {}).get("rating")
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs SET status=?, result_json=?, pdf_path=?, updated_at=?,
                            ticker=?, company_name=?, rating=?
            WHERE job_id=? AND status=?
            """,
            (STATUS_DONE, json.dumps(result), pdf_path, time.time(),
             ticker, company_name, rating, job_id, STATUS_RUNNING),
        )


def mark_error(job_id: str, error_code: str, error_message: str) -> None:
    """Same "only if still running" guard as mark_done -- see there."""
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs SET status=?, error_code=?, error_message=?, updated_at=?
            WHERE job_id=? AND status=?
            """,
            (STATUS_ERROR, error_code, error_message, time.time(), job_id, STATUS_RUNNING),
        )


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["result"] = json.loads(result["result_json"]) if result["result_json"] else None
    return result


def list_recent_jobs(user_id: str, limit: int = 10) -> list:
    """For the home page's Recent Reports section. Filters to status
    'done' (a queued/failed job isn't a "report") AND ticker IS NOT
    NULL -- the latter excludes rows written before the ticker/
    company_name/rating columns existed (see init_db's ALTER TABLE
    migration above), so a pre-migration job shows up nowhere rather
    than as a blank card with no ticker to display."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT job_id, ticker, company_name, rating, started_at
            FROM jobs
            WHERE user_id=? AND status=? AND ticker IS NOT NULL
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (user_id, STATUS_DONE, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_rating_for_ticker(user_id: str, ticker: str) -> Optional[str]:
    """The rating a Watchlist row shows -- the user's own last completed
    report for that ticker, not a live re-run of the DCF/LLM pipeline
    (which would be slow and expensive just to refresh a tile). None if
    the user has never researched this ticker."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT rating FROM jobs
            WHERE user_id=? AND ticker=? AND status=? AND rating IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (user_id, ticker, STATUS_DONE),
        ).fetchone()
    return row[0] if row is not None else None


# ---------------------------------------------------------------- watchlist

def add_watchlist_item(user_id: str, ticker: str) -> None:
    """INSERT OR IGNORE, not a duplicate-key error -- adding a ticker
    that's already on the watchlist is a no-op, same idempotency
    pattern as add_push_subscription above."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, ticker, added_at) VALUES (?, ?, ?)",
            (user_id, ticker, time.time()),
        )


def remove_watchlist_item(user_id: str, ticker: str) -> None:
    """No-op if the ticker isn't on the watchlist, same as
    delete_session/remove_push_subscription above."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM watchlist WHERE user_id=? AND ticker=?", (user_id, ticker)
        )


def get_watchlist(user_id: str) -> list:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticker, added_at FROM watchlist WHERE user_id=? ORDER BY added_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def reconcile_interrupted_jobs() -> int:
    """
    Run once at startup, before the app accepts requests. Any row
    still `pending` OR `running` at this point cannot possibly still be
    live -- the ThreadPoolExecutor and its queue only ever existed
    in-memory in whatever process owned them, and that process is gone
    (this function runs in a fresh one), so a queued-but-not-started
    job is exactly as orphaned as one mid-execution. Marking these
    `error`/INTERRUPTED rather than re-queuing them is a deliberate
    choice, not an oversight: resuming a partially-run agent pipeline
    isn't well-defined here (no intermediate ResearchContext state was
    ever persisted, only the final result) -- the only honest thing to
    tell the client is "that one failed, ask again."

    Returns the number of rows reconciled (0 in the common case).
    """
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status=?, error_code=?, error_message=?, updated_at=? WHERE status IN (?, ?)",
            (STATUS_ERROR, ERROR_INTERRUPTED,
             "Server restarted while this job was pending or running.", time.time(),
             STATUS_PENDING, STATUS_RUNNING),
        )
        return cursor.rowcount


def reconcile_timed_out_jobs(timeout_seconds: int = JOB_TIMEOUT_SECONDS) -> int:
    """
    Run periodically (see main.py's background sweep thread), not just
    at startup. Marks any `running` row older than `timeout_seconds`
    as `error`/TIMEOUT. See JOB_TIMEOUT_SECONDS's docstring above for
    the documented limitation this does NOT solve (the worker thread
    itself doesn't stop).
    """
    cutoff = time.time() - timeout_seconds
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status=?, error_code=?, error_message=?, updated_at=? "
            "WHERE status=? AND started_at < ?",
            (STATUS_ERROR, ERROR_TIMEOUT,
             f"Job exceeded {timeout_seconds}s and was marked timed out.",
             time.time(), STATUS_RUNNING, cutoff),
        )
        return cursor.rowcount
