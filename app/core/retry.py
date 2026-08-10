"""
retry.py

Generic transient-failure retry wrapper, mirroring llm_provider.py's
HostedProvider retry/backoff shape (see that module for the original,
which stays as-is -- this doesn't replace it, it generalizes the same
pattern for callers that aren't the LLM provider) but usable by any
function that makes a network call.

Retries only requests.exceptions.Timeout/ConnectionError (network-level
transient failures) and an HTTPError on a 429 or >=500 status. Never
retries any other exception, or an HTTPError on any other 4xx status --
those are permanent (bad auth, bad request, not found) and retrying them
just delays a failure that was already certain from the first attempt.

On exhaustion, re-raises the LAST real exception as-is rather than
wrapping it in a new type -- sec_edgar_client.py/news_client.py's own
broad `except (requests.RequestException, ValueError)` clauses already
catch exactly that hierarchy, so wiring retry in "inside" those existing
boundaries (see this repo's production-readiness plan) needs no change
to what they catch.
"""

import time
from typing import Callable, Optional, TypeVar

import requests

DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_SECONDS = 1.0

T = TypeVar("T")


def _backoff_delay(attempt: int, response: Optional[requests.Response], base_seconds: float) -> float:
    """Same Retry-After-first, exponential-fallback shape as
    llm_provider.py's HostedProvider._backoff_delay -- honoring a real
    Retry-After header is how you get politely throttled instead of
    rate-limit-banned; the fixed exponential schedule is only a
    fallback for when no header is present."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return base_seconds * (2 ** attempt)


def retry_on_transient_error(
    fn: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
) -> T:
    """
    Calls `fn()` (a zero-argument callable -- wrap the real call in a
    lambda/closure at the call site, e.g.
    `retry_on_transient_error(lambda: requests.get(url, timeout=15))`),
    retrying up to `max_retries` additional times on a transient
    failure. Sleeps between attempts using _backoff_delay above.

    Re-raises the triggering exception unchanged if every attempt is
    exhausted, or immediately for any non-transient exception (no
    retries spent on a failure that was never going to succeed).
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status != 429 and (status is None or status < 500):
                raise
            last_error = e

        if attempt < max_retries:
            response = getattr(last_error, "response", None)
            time.sleep(_backoff_delay(attempt, response, backoff_base_seconds))

    raise last_error
