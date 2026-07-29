"""
paths.py

Single source of truth for where the API's persistent state lives --
jobs.db (app/api/db.py), rendered PDFs (app/api/jobs.py), and LLM call
logs (app/core/llm_call_logger.py) all derive their paths from
DATA_DIR here, instead of each hardcoding a path relative to the repo
root independently and risking drift between them.

Why this exists: a Railway (or any container platform) deployment's
container filesystem is wiped on every redeploy -- a hardcoded
repo-relative path silently loses every job record and every generated
PDF the moment the service restarts, which defeats db.py's own
reconcile_interrupted_jobs()/reconcile_timed_out_jobs() persistence
work entirely (there's nothing to reconcile if the rows never survived
the redeploy that triggered the need to reconcile them). DATA_DIR
points this at a mounted persistent volume in production (Railway:
attach a volume, set DATA_DIR to its mount path); unset, it defaults
to the repo root, preserving the exact local-dev behavior this had
before DATA_DIR existed.
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent.parent))

DB_PATH = DATA_DIR / "jobs.db"
REPORTS_DIR = DATA_DIR / "reports"
LLM_LOG_DIR = DATA_DIR / "llm_logs"
