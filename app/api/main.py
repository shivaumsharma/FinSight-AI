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
from datetime import date, timedelta
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from app.api import auth, db, errors, jobs
from app.core.company_resolver import get_company_name, resolve_companies, suggest_companies
from app.data.market_data import TickerNotFoundError, get_corporate_actions, get_quote, get_usd_conversion_rate
from app.reasoning.backtest_stats import get_backtest_accuracy_summary
from app.reasoning.market_movers import get_top_movers
from app.reasoning.model_consensus import compute_consensus, get_model_opinions
from app.reporting.news_client import fetch_company_news, fetch_market_news

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


class PortfolioRequest(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    buy_date: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("quantity", "avg_cost")
    @classmethod
    def _must_be_positive(cls, v: float) -> float:
        # Self-reported entry, not a trade -- but zero/negative shares
        # or cost basis can't correspond to any real holding, so reject
        # it the same way SignupRequest rejects a too-short password:
        # a field_validator ValueError, which FastAPI turns into a 422.
        if v <= 0:
            raise ValueError("Must be greater than zero.")
        return v

    @field_validator("buy_date")
    @classmethod
    def _valid_iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        try:
            date.fromisoformat(v.strip())
        except ValueError:
            raise ValueError("buy_date must be an ISO date (YYYY-MM-DD).")
        return v.strip()


class OrderRequest(BaseModel):
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: float

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("quantity")
    @classmethod
    def _must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be greater than zero.")
        return v


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
    window_start = time.time() - RATE_LIMIT_WINDOW_SECONDS
    jobs_used_today = db.count_recent_jobs(current_user, window_start)
    # authorization is guaranteed well-formed here -- Depends(auth.get_current_user)
    # above already validated it (raised 401 otherwise) before this
    # body ever runs, so extract_bearer_token can't itself fail.
    session_expires_at = db.get_session_expiry(auth.extract_bearer_token(authorization))
    # The oldest job counted in the current rate-limit window is the
    # next one to age out -- it does so exactly RATE_LIMIT_WINDOW_SECONDS
    # after it started, which is the moment jobs_used_today next ticks
    # down by one. None (no countdown) if nothing's counted right now.
    oldest = db.oldest_job_time_in_window(current_user, window_start)
    reset_at = (oldest + RATE_LIMIT_WINDOW_SECONDS) if oldest is not None else None
    return {
        "user_id": current_user,
        "email": user["email"] if user else None,
        "created_at": user["created_at"] if user else None,
        "jobs_used_today": jobs_used_today,
        "daily_limit": DAILY_JOB_LIMIT,
        "reset_at": reset_at,
        "total_reports": db.count_all_jobs(current_user),
        "session_expires_at": session_expires_at,
        # NULL on a fresh/pre-migration row defaults to "Moderate" here
        # (read time), not backfilled in the DB -- same nullable
        # pattern the jobs-table migration above already uses.
        "risk_tolerance": user["risk_tolerance"] if user and user.get("risk_tolerance") else "Moderate",
        "waitlist_features": db.list_waitlist_features(current_user),
        # None when unset -- the frontend's own displayNameFromEmail
        # heuristic is the fallback (see page.tsx), not duplicated here.
        "display_name": user["display_name"] if user else None,
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


class DisplayNameRequest(BaseModel):
    display_name: str | None = None

    @field_validator("display_name")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None  # empty/whitespace-only clears back to the email heuristic
        if len(v) > 40:
            raise ValueError("Display name must be 40 characters or fewer.")
        return v


@app.patch("/v1/auth/display-name")
def update_display_name(body: DisplayNameRequest, current_user: str = Depends(auth.get_current_user)):
    db.set_display_name(current_user, body.display_name)
    return {"status": "ok", "display_name": body.display_name}


class WaitlistRequest(BaseModel):
    feature: str


@app.post("/v1/waitlist")
def join_waitlist(body: WaitlistRequest, current_user: str = Depends(auth.get_current_user)):
    # No allowlist of valid `feature` slugs -- this is meant to stay
    # generic across whatever not-yet-built feature needs a real
    # "interested" signal next, not just today's Brokerage Sync.
    db.join_waitlist(current_user, body.feature)
    return {"status": "ok"}


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


@app.get("/v1/research/backtest-accuracy")
def get_backtest_accuracy(current_user: str = Depends(auth.get_current_user)):
    # Registered before /v1/research/{job_id} for the same routing
    # reason as /v1/research/recent above -- same static number for
    # every report, not scoped to a specific job_id. Returns 404 if the
    # backtest result files are missing (should never happen in a real
    # deploy, but degrading honestly beats fabricating a number).
    summary = get_backtest_accuracy_summary()
    if summary is None:
        raise errors.backtest_accuracy_unavailable()
    return summary


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


@app.post("/v1/research/{job_id}/model-compare")
def model_compare(job_id: str, current_user: str = Depends(auth.get_current_user)):
    """On-demand second-opinion panel for an already-completed report --
    3 independent models re-interpreting the SAME already-computed
    evidence (not a re-run of SEC/RAG/DCF), so this stays cheap and
    fast. Deliberately not persisted: unlike the report itself, a
    comparison is cheap enough to regenerate on each view, and not
    caching it avoids a new DB table for v1."""
    job = db.get_job(job_id)
    if job is None:
        raise errors.job_not_found(job_id)
    if job["user_id"] != current_user:
        raise errors.forbidden(job_id)
    if job["status"] != db.STATUS_DONE or not job["result"]:
        raise errors.job_not_done(job_id, job["status"])

    report_data = job["result"].get("report_data", {})
    ticker = job["result"].get("ticker", "")
    opinions = get_model_opinions(report_data, ticker)
    return {"models": opinions, "consensus": compute_consensus(opinions)}


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


# Curated, not user-editable -- a fixed carousel of major indices
# rather than an arbitrary-ticker lookup. GIFT NIFTY is deliberately
# excluded: neither "NIFTY_FUT.NS" nor "GIFTNIFTY" resolve on Yahoo
# Finance (confirmed empirically -- both 404 from yfinance), so there's
# no reliable free data source for it.
#
# "region" powers the /indices page's Indian/Global tab split (see
# IndicesCarousel.tsx's "VIEW ALL" link) -- the home carousel itself
# still shows all of them together, unfiltered.
INDEX_LIST = [
    {"name": "S&P 500", "ticker": "^GSPC", "region": "global"},
    {"name": "NASDAQ", "ticker": "^IXIC", "region": "global"},
    {"name": "DOW", "ticker": "^DJI", "region": "global"},
    {"name": "NIFTY 50", "ticker": "^NSEI", "region": "india"},
    {"name": "SENSEX", "ticker": "^BSESN", "region": "india"},
    {"name": "BANK NIFTY", "ticker": "^NSEBANK", "region": "india"},
    {"name": "INDIA VIX", "ticker": "^INDIAVIX", "region": "india"},
]


@app.get("/v1/companies/suggest")
def suggest_companies_endpoint(q: str = Query(default=""), current_user: str = Depends(auth.get_current_user)):
    # Powers the Watchlist add-ticker input's dropdown -- a user typing
    # "tata" should see real candidates to pick from (Tata Motors, Tata
    # Steel, ...) rather than having to already know the exact symbol.
    return {"suggestions": suggest_companies(q, limit=8)}


@app.get("/v1/news/market")
def get_market_news(limit: int = Query(default=15, le=30), current_user: str = Depends(auth.get_current_user)):
    # Reuses the same Finnhub client the research pipeline's per-ticker
    # Risk Analysis/Sentiment sections already depend on -- a separate,
    # general (not-per-company) endpoint and cache, not a new provider.
    return {"articles": fetch_market_news(limit=limit)}


@app.get("/v1/news/my-stocks")
def get_my_stocks_news(limit: int = Query(default=15, le=30), current_user: str = Depends(auth.get_current_user)):
    # "On My Stocks" tab -- reuses fetch_company_news (the SAME
    # per-ticker Finnhub client + 24h cache the research pipeline's own
    # News Selection stage already depends on), just aggregated across
    # every ticker on the watchlist and/or portfolio (same union as the
    # Corporate Actions feed) into one flat, deduped, recency-filtered
    # list rather than shown per-ticker. fetch_company_news's own cache
    # key is per-ticker only (not per `days`), so filtering to the last
    # week happens here rather than passing a narrower `days` argument
    # that could silently collide with an already-cached wider-window
    # fetch from elsewhere in the app.
    watchlist_tickers = {w["ticker"] for w in db.get_watchlist(current_user)}
    portfolio_tickers = {h["ticker"] for h in db.get_portfolio_holdings(current_user)}
    tickers = watchlist_tickers | portfolio_tickers

    cutoff = (date.today() - timedelta(days=7)).isoformat()
    seen_urls = set()
    articles = []
    for ticker in tickers:
        for article in fetch_company_news(ticker):
            if article["date"] < cutoff or article["url"] in seen_urls:
                continue
            seen_urls.add(article["url"])
            articles.append({**article, "ticker": ticker})

    articles.sort(key=lambda a: a["date"], reverse=True)
    return {"articles": articles[:limit]}


@app.get("/v1/market/indices")
def get_market_indices(current_user: str = Depends(auth.get_current_user)):
    indices = []
    for entry in INDEX_LIST:
        try:
            quote = get_quote(entry["ticker"])
        except Exception:
            # Same per-item isolation as the watchlist below -- one
            # index's fetch failure (Yahoo throttling, a transient
            # outage) must not blank out the whole carousel.
            quote = None
        indices.append({
            "name": entry["name"],
            "ticker": entry["ticker"],
            "region": entry["region"],
            "currency": quote.get("currency", "USD") if quote else "USD",
            "price": quote["price"] if quote else None,
            "change_pct": quote["change_pct"] if quote else None,
        })
    return {"indices": indices}


@app.get("/v1/market/fx")
def get_fx_rate(current_user: str = Depends(auth.get_current_user)):
    # USD/INR only -- the one non-USD currency this app's India
    # coverage actually needs (see get_usd_conversion_rate's own
    # comment). "rate" is INR per 1 USD, the same convention Yahoo/
    # Google/XE all quote this pair in.
    try:
        quote = get_quote("INR=X")
    except Exception:
        return {"pair": "USD/INR", "rate": None, "change_pct": None}
    return {"pair": "USD/INR", "rate": quote["price"], "change_pct": quote["change_pct"]}


@app.get("/v1/corporate-actions/feed")
def get_corporate_actions_feed(
    scope: Literal["all", "portfolio"] = Query(default="all"),
    current_user: str = Depends(auth.get_current_user),
):
    # "portfolio" scope is just the self-reported holdings; "all" adds
    # in the watchlist too -- the same two ticker sources Watchlist.tsx
    # and Portfolio.tsx already track separately, just unioned here into
    # one chronological feed rather than shown per-component. A ticker
    # on both lists contributes its events only once (set union, not
    # concatenation).
    portfolio_tickers = {h["ticker"] for h in db.get_portfolio_holdings(current_user)}
    if scope == "portfolio":
        tickers = portfolio_tickers
    else:
        watchlist_tickers = {w["ticker"] for w in db.get_watchlist(current_user)}
        tickers = watchlist_tickers | portfolio_tickers

    events = []
    for ticker in tickers:
        try:
            actions = get_corporate_actions(ticker)
        except Exception:
            # Same per-ticker isolation as everywhere else -- one bad
            # symbol must not blank out the rest of the feed.
            continue
        name = get_company_name(ticker) or ticker
        if actions["next_earnings_date"]:
            events.append({"ticker": ticker, "name": name, "type": "earnings", "date": actions["next_earnings_date"]})
        if actions["next_ex_dividend_date"]:
            events.append({"ticker": ticker, "name": name, "type": "ex_dividend", "date": actions["next_ex_dividend_date"]})

    # ISO date strings sort correctly as plain strings -- soonest
    # upcoming event first, matching what a "what's coming up" feed
    # should lead with.
    events.sort(key=lambda e: e["date"])
    return {"events": events, "scope": scope}


@app.get("/v1/market/movers")
def get_market_movers(limit: int = Query(default=5, le=10), current_user: str = Depends(auth.get_current_user)):
    # "Tracked Universe" (curated backtest tickers + every user's
    # watchlist tickers, unioned globally) -- NOT a full-market scan,
    # see market_movers.py's module docstring. Frontend must label this
    # clearly so it's never read as full-market coverage.
    return get_top_movers(limit=limit)


@app.get("/v1/market/sentiment")
def get_market_sentiment(current_user: str = Depends(auth.get_current_user)):
    # Global across all users' completed research, one vote per
    # distinct ticker at its most recent rating -- same "pick one scope
    # and be consistent" reasoning as Top Movers above, per the spec.
    # Hold is deliberately excluded from the ratio (a Buy-vs-Sell
    # "battery" gauge reads oddly with a third silent category baked
    # into the denominator) but still returned as its own count.
    counts = db.get_global_rating_distribution()
    buy, hold, sell = counts["Buy"], counts["Hold"], counts["Sell"]
    voting_total = buy + sell
    buy_pct = round(buy / voting_total * 100, 1) if voting_total else None
    sell_pct = round(sell / voting_total * 100, 1) if voting_total else None
    return {
        "buy_pct": buy_pct,
        "sell_pct": sell_pct,
        "buy_count": buy,
        "hold_count": hold,
        "sell_count": sell,
        "total_rated": buy + hold + sell,
    }


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
        try:
            corporate_actions = get_corporate_actions(ticker)
        except Exception:
            # Same isolation -- a corporate-actions lookup failure must
            # not blank out the price/rating that already succeeded.
            corporate_actions = None
        items.append({
            "ticker": ticker,
            "price": quote["price"] if quote else None,
            "change_pct": quote["change_pct"] if quote else None,
            "currency": quote.get("currency", "USD") if quote else "USD",
            "rating": db.get_latest_rating_for_ticker(current_user, ticker),
            "added_at": row["added_at"],
            "next_earnings_date": corporate_actions["next_earnings_date"] if corporate_actions else None,
            "next_ex_dividend_date": corporate_actions["next_ex_dividend_date"] if corporate_actions else None,
            "last_dividend_amount": corporate_actions["last_dividend_amount"] if corporate_actions else None,
            "last_split": corporate_actions["last_split"] if corporate_actions else None,
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


# ---------------------------------------------------------------- portfolio (self-reported)
#
# Manual holdings entry + live-quote P&L calculation ONLY. No
# brokerage connection, no order placement, no execution of any kind
# -- this deliberately stops at "what would my P&L be if what I typed
# in is accurate", the same category boundary the rest of this app
# stays inside of everywhere else.

@app.get("/v1/portfolio")
def get_portfolio(current_user: str = Depends(auth.get_current_user)):
    # Each holding displays in its OWN native currency (price/cost_basis/
    # market_value/today_pnl are never converted at the per-holding
    # level) -- but the aggregate summary below needs one common unit
    # to sum across a mixed USD/INR portfolio meaningfully, so summary
    # totals are converted to USD equivalent via a live FX rate
    # (get_usd_conversion_rate). A holding whose currency has no known
    # FX rate is shown correctly on its own row but excluded from the
    # summary totals (mixed_currency_excluded flags this for the UI)
    # rather than silently mixing incompatible units again.
    holdings = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_today_pnl = 0.0
    # Denominator for total_today_pnl_pct -- only the market value of
    # holdings that actually contributed a today_pnl (i.e. had a real
    # previous_close), so a holding with a missing quote doesn't skew
    # the percentage by inflating the denominator without a matching
    # numerator contribution.
    market_value_with_today_pnl = 0.0
    any_non_usd = False
    any_excluded_from_summary = False
    for row in db.get_portfolio_holdings(current_user):
        ticker = row["ticker"]
        quantity = row["quantity"]
        avg_cost = row["avg_cost"]
        try:
            quote = get_quote(ticker)
        except Exception:
            # Same per-ticker isolation as the watchlist -- one
            # bad/delisted holding must not 500 the whole portfolio.
            quote = None

        price = quote["price"] if quote else None
        previous_close = quote.get("previous_close") if quote else None
        currency = quote.get("currency", "USD") if quote else "USD"
        if currency != "USD":
            any_non_usd = True
        cost_basis = quantity * avg_cost
        market_value = price * quantity if price is not None else None
        unrealized_pnl = (market_value - cost_basis) if market_value is not None else None
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if unrealized_pnl is not None and cost_basis else None
        today_pnl = quantity * (price - previous_close) if price is not None and previous_close else None

        usd_rate = get_usd_conversion_rate(currency)
        if usd_rate is None:
            any_excluded_from_summary = True
        else:
            if market_value is not None:
                total_market_value += market_value * usd_rate
            total_cost_basis += cost_basis * usd_rate
            if today_pnl is not None:
                total_today_pnl += today_pnl * usd_rate
                market_value_with_today_pnl += market_value * usd_rate

        holdings.append({
            "ticker": ticker,
            "quantity": quantity,
            "avg_cost": avg_cost,
            "buy_date": row.get("buy_date"),
            "price": price,
            "change_pct": quote["change_pct"] if quote else None,
            "currency": currency,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "today_pnl": today_pnl,
            # Live verdict from this ticker's latest completed report --
            # the SAME lookup the Watchlist already uses, so a holding
            # whose research call has since flipped (e.g. to Sell) shows
            # it here too, not just on the Watchlist card.
            "rating": db.get_latest_rating_for_ticker(current_user, ticker),
            "added_at": row["added_at"],
        })

    total_unrealized_pnl = total_market_value - total_cost_basis if holdings else None
    total_unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_unrealized_pnl and total_cost_basis else None

    return {
        "holdings": holdings,
        "summary": {
            "total_market_value": total_market_value if holdings else None,
            "total_cost_basis": total_cost_basis if holdings else None,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
            "total_today_pnl": total_today_pnl if market_value_with_today_pnl else None,
            # Percentage against YESTERDAY's value of just the holdings
            # that contributed a today_pnl (today's market value minus
            # today's gain) -- not total_market_value, which may include
            # holdings with no previous_close and would otherwise skew
            # the denominator without a matching numerator contribution.
            "total_today_pnl_pct": (total_today_pnl / (market_value_with_today_pnl - total_today_pnl) * 100)
                if market_value_with_today_pnl and (market_value_with_today_pnl - total_today_pnl) else None,
            # Summary totals are always USD-equivalent, regardless of
            # each holding's own native currency -- mixed_currency tells
            # the frontend to label the total "(USD equiv.)" instead of
            # implying every holding actually trades in dollars.
            "currency": "USD",
            "mixed_currency": any_non_usd,
            "excluded_from_summary": any_excluded_from_summary,
        },
    }


@app.post("/v1/portfolio")
def add_or_update_portfolio_holding(body: PortfolioRequest, current_user: str = Depends(auth.get_current_user)):
    # Same fast-path-then-resolve_companies-fallback validation as
    # POST /v1/watchlist, so a fuzzy/Indian company name works here too.
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

    db.upsert_portfolio_holding(current_user, ticker, body.quantity, body.avg_cost, body.buy_date)
    return {"status": "ok"}


@app.delete("/v1/portfolio/{ticker}")
def remove_portfolio_holding(ticker: str, current_user: str = Depends(auth.get_current_user)):
    db.remove_portfolio_holding(current_user, ticker.upper())
    return {"status": "ok"}


@app.get("/v1/portfolio/analysis")
def get_portfolio_analysis(current_user: str = Depends(auth.get_current_user)):
    # Combined portfolio analysis -- aggregates each holding's already-
    # researched rating (the SAME get_latest_rating_for_ticker lookup
    # GET /v1/portfolio's own per-row rating already uses -- this is
    # NOT a new research run, just a rollup of existing reports) into
    # one portfolio-level Buy/Hold/Sell breakdown, both by simple count
    # and by USD-equivalent market-value weight (a $10k Buy-rated
    # holding should count more than a $50 one). Holdings with no
    # completed report yet are tracked separately as "unresearched",
    # never silently dropped or defaulted to a rating.
    holdings = db.get_portfolio_holdings(current_user)
    rating_counts = {"Buy": 0, "Hold": 0, "Sell": 0}
    value_by_rating = {"Buy": 0.0, "Hold": 0.0, "Sell": 0.0}
    total_value_usd = 0.0
    unresearched_tickers = []

    for row in holdings:
        ticker = row["ticker"]
        rating = db.get_latest_rating_for_ticker(current_user, ticker)
        if rating is None or rating not in rating_counts:
            unresearched_tickers.append(ticker)
            continue

        rating_counts[rating] += 1
        try:
            quote = get_quote(ticker)
        except Exception:
            continue
        usd_rate = get_usd_conversion_rate(quote.get("currency", "USD"))
        if usd_rate is None:
            continue
        market_value_usd = quote["price"] * row["quantity"] * usd_rate
        value_by_rating[rating] += market_value_usd
        total_value_usd += market_value_usd

    value_weighted_pct = (
        {r: round(v / total_value_usd * 100, 1) for r, v in value_by_rating.items()}
        if total_value_usd else None
    )

    return {
        "holdings_total": len(holdings),
        "holdings_researched": sum(rating_counts.values()),
        "rating_counts": rating_counts,
        "value_weighted_pct": value_weighted_pct,
        "unresearched_tickers": unresearched_tickers,
    }


# ---------------------------------------------------------------- orders (simulated paper trading)
#
# NO real broker, NO real credentials, NO real money -- a BUY/SELL
# order fills instantly and completely at the live quote price at
# order time, then updates the SAME self-reported portfolio_holdings
# table Portfolio's manual-entry form writes to (see db.execute_order's
# own docstring for the full boundary explanation). This is a
# rehearsal tool for a trade decision against real prices, not a
# brokerage integration -- that stays out of scope regardless.

@app.post("/v1/orders")
def place_order(body: OrderRequest, current_user: str = Depends(auth.get_current_user)):
    # Same fast-path-then-resolve_companies-fallback validation as
    # POST /v1/portfolio/POST /v1/watchlist, so a fuzzy/Indian company
    # name works here too, not just a bare ticker.
    ticker = body.ticker
    try:
        quote = get_quote(ticker)
    except TickerNotFoundError:
        resolved = resolve_companies(body.ticker)
        if not resolved:
            raise errors.ticker_not_found(body.ticker)
        ticker = resolved[0]
        try:
            quote = get_quote(ticker)
        except TickerNotFoundError:
            raise errors.ticker_not_found(body.ticker)

    try:
        result = db.execute_order(
            current_user, ticker, body.side, body.quantity, quote["price"], quote.get("currency", "USD"),
        )
    except ValueError as exc:
        raise errors.insufficient_shares(str(exc))

    return result


@app.get("/v1/orders")
def get_orders(limit: int = Query(default=20, le=50), current_user: str = Depends(auth.get_current_user)):
    return {"orders": db.list_orders(current_user, limit)}
