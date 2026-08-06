"""
main.py

FastAPI service boundary in front of the research pipeline -- step 1 of
a staged, mobile-aware rollout (see the approved plan for the full
context and the later step deliberately NOT in this pass: push
notifications -- hosted inference, the shared-secret API key check,
real per-user auth/rate-limiting, signed PDF share links, and
disclaimer language, see below, are now all in place). streamlit_app.py
is unchanged by this file and still calls ResearchAgent/
LangGraphResearchAgent in-process directly -- rewiring it to call this
API instead is a later step, done deliberately minimally when it
happens, since Streamlit is staying a debug tool, not becoming the
product.

Two auth layers, not one, stacked deliberately (see _API_KEY's own
comment below for the full reasoning): X-API-Key gates the deployment
itself (anyone allowed to call this service at all), the session-token
Depends(auth.get_current_user) on /v1/research/* identifies *which*
user is calling and scopes rate-limiting/job ownership to them. Auth
endpoints (/v1/auth/*) still sit behind X-API-Key -- signup/login are
not exempted -- but obviously can't require a session token themselves.

Startup does two things before the app accepts any request:
1. db.init_db() -- create the jobs/users/sessions tables if they don't exist.
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
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from app.api import auth, db, errors, jobs
from app.core.company_resolver import resolve_companies
from app.data.market_data import TickerNotFoundError, get_quote

TIMEOUT_SWEEP_INTERVAL_SECONDS = 60

# How many research jobs one user may start per rolling 24h window --
# see db.count_recent_jobs's own docstring for why this needs no new
# table. Env-configurable so a deployment can tighten/loosen it without
# a code change.
DAILY_JOB_LIMIT = int(os.environ.get("DAILY_JOB_LIMIT", db.DEFAULT_DAILY_JOB_LIMIT))
RATE_LIMIT_WINDOW_SECONDS = 24 * 3600

# How long a login/signup session stays valid -- env-configurable
# (previously hardcoded to db.DEFAULT_SESSION_TTL_SECONDS regardless of
# any env var; this closes that gap).
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", db.DEFAULT_SESSION_TTL_SECONDS))

# How long a signed PDF share link (see auth.sign_pdf_url) stays valid
# after being generated -- 7 days by default, long enough to be useful
# for sharing a report with someone, short enough to bound how long an
# unauthenticated link stays live.
PDF_SHARE_TTL_SECONDS = int(os.environ.get("PDF_SHARE_TTL_SECONDS", 7 * 24 * 3600))

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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


class SignupRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        # Deliberately loose (no DNS/MX check, no full RFC 5322
        # grammar) -- this only needs to catch obvious typos before
        # they become an unrecoverable account (no email is ever
        # actually sent to it in this pass), not validate deliverability.
        if not _EMAIL_RE.match(v):
            raise ValueError("Not a valid email address.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def _min_length(cls, v: str) -> str:
        # 8 chars is NIST 800-63B's own minimum, not an arbitrary
        # round number -- that guidance explicitly recommends length
        # over composition rules (no forced uppercase/digit/symbol),
        # so this is deliberately the ONLY password rule here.
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class PushSubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscriptionRequest(BaseModel):
    # Matches the shape of the browser's own PushSubscription.toJSON()
    # exactly (endpoint + keys.{p256dh,auth}) -- the frontend forwards
    # that object unmodified rather than this API inventing its own
    # field names for the same three values.
    endpoint: str
    keys: PushSubscriptionKeys


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


class WatchlistRequest(BaseModel):
    ticker: str

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        # Uppercased/stripped here (not left to the endpoint) so
        # "aapl" and "AAPL" always land as the same watchlist row --
        # matches MarketDataLoader's own self.ticker = ticker.upper().
        return v.strip().upper()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/auth/signup")
def signup(body: SignupRequest):
    if db.get_user_by_email(body.email) is not None:
        raise errors.email_already_registered()

    password_hash, salt = auth.hash_password(body.password)
    user_id = db.create_user(body.email, password_hash, salt)
    token = db.create_session(user_id, ttl_seconds=SESSION_TTL_SECONDS)
    return {"session_token": token}


@app.post("/v1/auth/login")
def login(body: LoginRequest):
    user = db.get_user_by_email(body.email.lower())
    # Same failure path (and same errors.invalid_credentials() message)
    # whether the email doesn't exist or the password is wrong -- see
    # that error constructor's own comment on why.
    if user is None or not auth.verify_password(body.password, user["password_hash"], user["salt"]):
        raise errors.invalid_credentials()

    token = db.create_session(user["user_id"], ttl_seconds=SESSION_TTL_SECONDS)
    return {"session_token": token}


@app.post("/v1/auth/logout")
def logout(authorization: str = Header(default=None)):
    # Deletes only the session named by this specific token, not every
    # session this user holds (see db.py's sessions-table comment on
    # why session_token is the primary key) -- so doesn't need
    # Depends(auth.get_current_user) at all, just the raw token.
    token = auth.extract_bearer_token(authorization)
    db.delete_session(token)
    return {"status": "ok"}


@app.get("/v1/auth/me")
def me(authorization: str = Header(default=None), current_user: str = Depends(auth.get_current_user)):
    # useAuth.ts already calls this on every mount -- extending its
    # response (rather than adding a second endpoint) is what powers
    # the Profile view for free, no new round-trip. jobs_used_today/
    # daily_limit reuse the exact same rate-limit machinery
    # POST /v1/research already checks against, not a new counter.
    user = db.get_user_by_id(current_user)
    jobs_used_today = db.count_recent_jobs(current_user, time.time() - RATE_LIMIT_WINDOW_SECONDS)
    # authorization is guaranteed well-formed here -- Depends(auth.get_current_user)
    # above already validated it (raised 401 otherwise) before this
    # body ever runs, so extract_bearer_token can't itself fail.
    session_expires_at = db.get_session_expiry(auth.extract_bearer_token(authorization))
    return {
        "user_id": current_user,
        "email": user["email"] if user else None,
        "created_at": user["created_at"] if user else None,
        "jobs_used_today": jobs_used_today,
        "daily_limit": DAILY_JOB_LIMIT,
        "total_reports": db.count_all_jobs(current_user),
        "session_expires_at": session_expires_at,
        # NULL on a fresh/pre-migration row defaults to "Moderate" here
        # (read time), not backfilled in the DB -- same nullable
        # pattern the jobs-table migration above already uses.
        "risk_tolerance": user["risk_tolerance"] if user and user.get("risk_tolerance") else "Moderate",
    }


_RISK_TOLERANCE_LEVELS = ("Conservative", "Moderate", "Aggressive")


class RiskToleranceRequest(BaseModel):
    risk_tolerance: str

    @field_validator("risk_tolerance")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        if v not in _RISK_TOLERANCE_LEVELS:
            raise ValueError(f"risk_tolerance must be one of {_RISK_TOLERANCE_LEVELS}")
        return v


@app.patch("/v1/auth/risk-tolerance")
def update_risk_tolerance(body: RiskToleranceRequest, current_user: str = Depends(auth.get_current_user)):
    # Persisted preference only -- not yet applied to the research
    # pipeline's own WACC/discount assumptions or recommendation logic
    # (a separate, larger feature). Real and saved, not decorative.
    db.set_risk_tolerance(current_user, body.risk_tolerance)
    return {"status": "ok", "risk_tolerance": body.risk_tolerance}


class DeleteAccountRequest(BaseModel):
    password: str


@app.delete("/v1/auth/me")
def delete_account(body: DeleteAccountRequest, current_user: str = Depends(auth.get_current_user)):
    # Requires the password again, not just an active session -- an
    # irreversible action (every job, every watchlist item, the
    # account itself, all gone) deserves a stronger confirmation than
    # "you happened to have a valid cookie right now."
    user = db.get_user_by_id(current_user)
    if user is None or not auth.verify_password(body.password, user["password_hash"], user["salt"]):
        raise errors.invalid_credentials()

    db.delete_user_account(current_user)
    return {"status": "ok"}


@app.get("/v1/push/vapid-public-key")
def push_vapid_public_key():
    # Not a secret -- the browser needs this to call
    # pushManager.subscribe({applicationServerKey: ...}); it identifies
    # this server as the sender, it doesn't authorize anything on its
    # own (that's what the matching VAPID_PRIVATE_KEY does when this
    # server later signs an actual push message). No auth required.
    return {"public_key": auth.VAPID_PUBLIC_KEY}


@app.post("/v1/push/subscribe")
def push_subscribe(body: PushSubscriptionRequest, current_user: str = Depends(auth.get_current_user)):
    db.add_push_subscription(current_user, body.endpoint, body.keys.p256dh, body.keys.auth)
    return {"status": "ok"}


@app.post("/v1/push/unsubscribe")
def push_unsubscribe(body: PushUnsubscribeRequest, current_user: str = Depends(auth.get_current_user)):
    # Deliberately does not check that this endpoint belongs to
    # current_user before deleting -- an unsubscribe request for a
    # subscription that either doesn't exist or belongs to someone else
    # is already a no-op via db.remove_push_subscription's own
    # unconditional DELETE, and leaking whether a given endpoint string
    # belongs to another account isn't a distinction worth making here.
    db.remove_push_subscription(body.endpoint)
    return {"status": "ok"}


@app.post("/v1/research")
def submit_research(
    body: ResearchRequest,
    orchestrator: str = Query("hand_rolled", pattern="^(hand_rolled|langgraph)$"),
    current_user: str = Depends(auth.get_current_user),
):
    companies = resolve_companies(body.question)
    if not companies:
        raise errors.no_company_detected(body.question)

    recent = db.count_recent_jobs(current_user, time.time() - RATE_LIMIT_WINDOW_SECONDS)
    if recent >= DAILY_JOB_LIMIT:
        raise errors.rate_limit_exceeded(DAILY_JOB_LIMIT)

    job_id = jobs.submit_job(question=body.question, orchestrator=orchestrator, user_id=current_user)
    return {"job_id": job_id}


@app.get("/v1/research/recent")
def get_recent_research(limit: int = Query(10, ge=1, le=50), current_user: str = Depends(auth.get_current_user)):
    # Registered BEFORE /v1/research/{job_id} deliberately -- Starlette
    # matches routes in registration order, so if {job_id} came first,
    # a request to /v1/research/recent would match it with job_id="recent"
    # and never reach this handler.
    return {"reports": db.list_recent_jobs(current_user, limit)}


@app.get("/v1/research/{job_id}")
def get_research(job_id: str, current_user: str = Depends(auth.get_current_user)):
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)
    if job["user_id"] != current_user:
        raise errors.forbidden(job_id)

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


@app.post("/v1/research/{job_id}/pdf/share")
def share_research_pdf(job_id: str, current_user: str = Depends(auth.get_current_user)):
    """Mints a time-limited, signature-verified link to this job's PDF
    that works with NO session at all (see get_research_pdf below) --
    for sharing a report with someone who doesn't have a FinSight
    account. Only the report's own owner may mint one; the same
    ownership check get_research_pdf itself uses."""
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)
    if job["user_id"] != current_user:
        raise errors.forbidden(job_id)
    if job["status"] != db.STATUS_DONE or not job["pdf_path"]:
        raise errors.job_not_done(job_id, job["status"])

    expires_at, signature = auth.sign_pdf_url(job_id, PDF_SHARE_TTL_SECONDS)
    return {
        "url": f"/v1/research/{job_id}/pdf?exp={expires_at}&sig={signature}",
        "expires_at": expires_at,
    }


@app.get("/v1/research/{job_id}/pdf")
def get_research_pdf(
    job_id: str,
    exp: Optional[int] = Query(None),
    sig: Optional[str] = Query(None),
    current_user: Optional[str] = Depends(auth.get_current_user_optional),
):
    """Two independent ways in, either is sufficient: a real session
    belonging to the job's owner (existing behavior, unchanged), OR a
    valid, unexpired exp/sig pair from share_research_pdf above (new --
    deliberately skips the ownership check entirely in this branch,
    since possessing a valid signature for this exact job_id *is* the
    authorization, the same way a valid session already is)."""
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)

    if current_user is not None:
        if job["user_id"] != current_user:
            raise errors.forbidden(job_id)
    elif not (exp is not None and sig is not None and auth.verify_pdf_signature(job_id, exp, sig)):
        raise errors.unauthorized()

    if job["status"] != db.STATUS_DONE or not job["pdf_path"]:
        raise errors.job_not_done(job_id, job["status"])

    return FileResponse(job["pdf_path"], media_type="application/pdf",
                         filename=f"{job['job_id']}.pdf")


@app.get("/v1/watchlist")
def get_watchlist(current_user: str = Depends(auth.get_current_user)):
    items = []
    for row in db.get_watchlist(current_user):
        ticker = row["ticker"]
        try:
            quote = get_quote(ticker)
        except Exception:
            # One bad/delisted ticker's quote failure must not 500 the
            # whole list -- best-effort per ticker, null quote on failure.
            quote = None
        items.append({
            "ticker": ticker,
            "price": quote["price"] if quote else None,
            "change_pct": quote["change_pct"] if quote else None,
            "rating": db.get_latest_rating_for_ticker(current_user, ticker),
            "added_at": row["added_at"],
        })
    return {"items": items}


@app.post("/v1/watchlist")
def add_to_watchlist(body: WatchlistRequest, current_user: str = Depends(auth.get_current_user)):
    # Fast path: body.ticker is already a real, directly-quotable
    # symbol ("AAPL", "TSLA"). Only when that fails do we fall back to
    # resolve_companies -- the same NLP/NSE-aware resolver the main
    # search bar uses -- so a fuzzy company name (including a non-US
    # one, e.g. "Bajaj Finance" -> BAJFINANCE.NS, resolve_companies'
    # own docstring example) works here too, not just a bare ticker.
    # Without this fallback the watchlist quietly required stricter,
    # more technical input than the rest of the app.
    ticker = body.ticker
    try:
        get_quote(ticker)
    except TickerNotFoundError:
        resolved = resolve_companies(body.ticker)
        if not resolved:
            raise errors.ticker_not_found(body.ticker)
        ticker = resolved[0]
        try:
            get_quote(ticker)
        except TickerNotFoundError:
            raise errors.ticker_not_found(body.ticker)

    db.add_watchlist_item(current_user, ticker)
    return {"status": "ok"}


@app.delete("/v1/watchlist/{ticker}")
def remove_from_watchlist(ticker: str, current_user: str = Depends(auth.get_current_user)):
    # Idempotent, same pattern as push_unsubscribe -- removing a ticker
    # that isn't on the watchlist (or was already removed) is a no-op,
    # not an error.
    db.remove_watchlist_item(current_user, ticker.upper())
    return {"status": "ok"}
