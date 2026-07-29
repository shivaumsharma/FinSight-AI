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


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates the jobs table if it doesn't exist. Safe to call on
    every startup -- CREATE TABLE IF NOT EXISTS is idempotent."""
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
    """
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs SET status=?, result_json=?, pdf_path=?, updated_at=?
            WHERE job_id=? AND status=?
            """,
            (STATUS_DONE, json.dumps(result), pdf_path, time.time(), job_id, STATUS_RUNNING),
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
