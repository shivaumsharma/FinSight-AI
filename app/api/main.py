"""
main.py

FastAPI service boundary in front of the research pipeline -- step 1 of
a staged, mobile-aware rollout (see the approved plan for the full
context and the later steps deliberately NOT in this pass: real per-
user auth/rate-limiting, push notifications, signed PDF URLs,
positioning/disclaimer language -- hosted inference and a single-
shared-secret API key check, see below, are now both in place).
streamlit_app.py is unchanged by
this file and still calls ResearchAgent/LangGraphResearchAgent
in-process directly -- rewiring it to call this API instead is a later
step, done deliberately minimally when it happens, since Streamlit is
staying a debug tool, not becoming the product.

Startup does two things before the app accepts any request:
1. db.init_db() -- create the jobs table if it doesn't exist.
2. db.reconcile_interrupted_jobs() -- any row left PENDING/RUNNING from
   a previous process is guaranteed orphaned (see db.py's docstring
   for why), so it gets marked INTERRUPTED here rather than silently
   staying RUNNING forever from the next client's point of view.

A background thread then periodically calls
db.reconcile_timed_out_jobs() so a hung job's row stops lying to
clients even between restarts -- see db.py's JOB_TIMEOUT_SECONDS
docstring for the documented limitation this does NOT solve (it can't
force-kill the actual stuck worker thread).
"""

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.api import db, errors, jobs
from app.core.company_resolver import resolve_companies

TIMEOUT_SWEEP_INTERVAL_SECONDS = 60

# Comma-separated origins, e.g. "https://app.example.com,https://staging.example.com".
# Empty/unset means no browser origin is allowed cross-origin -- safe by
# default (this API has no browser client yet), and server-to-server /
# curl requests are unaffected either way since CORS is a browser-
# enforced restriction, not a server-side one. allow_credentials stays
# False: auth here is the X-API-Key header (see the middleware below),
# not cookies, so credentialed CORS mode isn't needed.
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# NOT real authentication or authorization: a single shared secret,
# checked with a plain string comparison -- no per-user identity, no
# expiry, no scopes, no rate limiting of its own. Its actual job: once
# this service has a public Railway URL, it's one HTTP request away
# from anyone on the internet burning through this deployment's real
# Finnhub free-tier quota and Groq API spend on every question
# submitted. This is a lock on the front door against that, not a
# security boundary -- a real product needs real per-user auth on top
# of this, not instead of it. If API_KEY isn't set at all, the check
# below is a no-op (existing local-dev/test behavior is unchanged) --
# it only activates once a real key is actually configured, so this
# can't accidentally break every existing caller the moment this
# middleware landed.
_API_KEY = os.environ.get("API_KEY")


def _timeout_sweep_loop():
    while True:
        time.sleep(TIMEOUT_SWEEP_INTERVAL_SECONDS)
        try:
            db.reconcile_timed_out_jobs()
        except Exception:
            # A failed sweep iteration must never kill the sweep thread
            # itself -- the next iteration should still get a chance to run.
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    reconciled = db.reconcile_interrupted_jobs()
    if reconciled:
        print(f"[startup] reconciled {reconciled} orphaned job(s) from a previous run -> INTERRUPTED")

    sweep_thread = threading.Thread(target=_timeout_sweep_loop, daemon=True)
    sweep_thread.start()

    yield


app = FastAPI(title="FinSight AI Research API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # OPTIONS (CORS preflight) is always exempt, registration order
    # relative to CORSMiddleware notwithstanding -- a real browser
    # preflight request never carries X-API-Key (browsers don't attach
    # custom headers to their own preflight), so gating it here would
    # break real cross-origin calls even when the caller's actual
    # request does carry a valid key.
    if (
        _API_KEY
        and request.url.path.startswith("/v1")
        and request.method != "OPTIONS"
        and request.headers.get("X-API-Key") != _API_KEY
    ):
        return JSONResponse(
            status_code=401,
            content={"code": "UNAUTHORIZED", "message": "Missing or invalid X-API-Key header."},
        )
    return await call_next(request)


@app.exception_handler(errors.APIError)
async def api_error_handler(request, exc: errors.APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


class ResearchRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/research")
def submit_research(
    body: ResearchRequest,
    orchestrator: str = Query("hand_rolled", pattern="^(hand_rolled|langgraph)$"),
):
    companies = resolve_companies(body.question)
    if not companies:
        raise errors.no_company_detected(body.question)

    job_id = jobs.submit_job(question=body.question, orchestrator=orchestrator)
    return {"job_id": job_id}


@app.get("/v1/research/{job_id}")
def get_research(job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "question": job["question"],
        "orchestrator": job["orchestrator"],
    }
    if job["status"] == db.STATUS_DONE:
        response["result"] = job["result"]
    if job["status"] == db.STATUS_ERROR:
        response["error_code"] = job["error_code"]
        response["error_message"] = job["error_message"]
    return response


@app.get("/v1/research/{job_id}/pdf")
def get_research_pdf(job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)
    if job["status"] != db.STATUS_DONE or not job["pdf_path"]:
        raise errors.job_not_done(job_id, job["status"])

    return FileResponse(job["pdf_path"], media_type="application/pdf",
                         filename=f"{job['job_id']}.pdf")
