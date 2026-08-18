"""
llm_provider.py

Pluggable LLM backend behind a single interface (LLMProvider), selected
by the LLM_PROVIDER env var ("hosted" default -- set LLM_PROVIDER=local
to opt back into the on-box llama.cpp model, which stays fully
supported and unchanged by this default flip).
Every caller that used to reach for get_shared_generator() (removed --
see below) now calls get_llm_provider().generate(prompt, max_new_tokens)
instead; the method signature is identical to ReportGenerator.generate()'s,
so each call site's change is genuinely two lines (import name, function
name), not a rewrite. Confirmed by grepping every real call site before
writing this file: app/core/company_resolver.py, app/planner/llm_planner.py,
app/reporting/narrative_builder.py -- all three call `.generate(prompt,
max_new_tokens=N)` with no other parameters.

Two implementations:
- LocalLlamaProvider: wraps the existing ReportGenerator
  (app/rag/report_generator.py) unchanged -- same lazy llama.cpp load,
  same temperature=0.0/repeat_penalty=1.15 internals. Nothing about how
  local inference actually runs is touched by this file.
- HostedProvider: any OpenAI-compatible chat completions API (Groq,
  Together, OpenRouter, OpenAI, ...) via LLM_BASE_URL/LLM_API_KEY/LLM_MODEL.

get_shared_generator() is removed, not kept alongside the new interface --
every current call site moves to get_llm_provider(), and two parallel ways
to reach "the LLM" would be exactly the kind of duplication this codebase
avoids everywhere else.

The prompt-format risk (read before relying on LLM_PROVIDER=hosted, the default)
------------------------------------------------------------------------
Every prompt in this codebase (report_generator.py's own docstring says
this explicitly) is a raw TEXT-CONTINUATION prompt aimed at llama.cpp's
completion API, not a chat-formatted one -- no caller here has ever used
a chat template. HostedProvider necessarily wraps that same prompt text
as a single `user` message, because "OpenAI-compatible chat completions
API" is what every mainstream hosted provider actually exposes. A chat
model receiving that text will tend to respond AS an assistant answering
it, not continue it the way the local completion API does -- for the
~700-token narrative call this can mean different framing, unwanted
preamble, or an outright refusal to "just continue." This is not
something stubbed unit tests can catch, and it is not fixed by this
file -- see scripts/compare_local_vs_hosted_narrative.py, a required
manual (human-read) comparison before LLM_PROVIDER=hosted is used for
anything real.

Concurrency note
-----------------
is_local_provider() is read by app/api/jobs.py to decide max_workers.
Local mode is hard-pinned to 1 there regardless of any env var (llama.cpp's
shared context isn't documented as thread-safe for concurrent generate()
calls) -- hosted mode ALSO defaults to 1 in this pass, since removing the
LLM as a concurrency blocker doesn't mean the rest of the pipeline
(SEC EDGAR's rate limit, Finnhub/News free-tier caps, the shared FinBERT
model, the shared ChromaDB client, the narrative Redis cache's
check-then-set path) has been audited for concurrent access. See jobs.py.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import requests

from app.core.paths import LLM_LOG_DIR

LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


def _merge_usage(a: dict, b: dict, combine) -> dict:
    """Recursively combines two usage dicts leaf-by-leaf via `combine(a_val,
    b_val)`. Needed because some hosted models (Groq's gpt-oss/qwen
    reasoning models) nest extra fields under completion_tokens_details
    (e.g. {"reasoning_tokens": N}) instead of keeping every usage field a
    flat int -- a naive top-level `a.get(k, 0) + b.get(k, 0)` raises
    TypeError: unsupported operand type(s) for +/-: 'dict' and 'int' on
    those, which used to take the whole call down with it (this ran in
    logged_generate's `finally`, after the real response had already been
    produced, so the exception it raised replaced a perfectly good result)."""
    result = {}
    for k in set(a) | set(b):
        av, bv = a.get(k), b.get(k)
        if isinstance(av, dict) or isinstance(bv, dict):
            result[k] = _merge_usage(av or {}, bv or {}, combine)
        else:
            result[k] = combine(av or 0, bv or 0)
    return result


class LLMProviderError(Exception):
    """Raised when a provider fails after exhausting retries, is
    unreachable, is misconfigured, or returns a non-retryable error.
    Caught explicitly in app/api/jobs.py's _run_job and surfaced as the
    LLM_PROVIDER_ERROR job error code."""


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> Optional[dict]:
        """{"prompt_tokens", "completion_tokens", "total_tokens"} accumulated
        since the last reset_usage() call, or None if nothing has been
        generated yet (or the provider never reported usage)."""
        raise NotImplementedError

    @abstractmethod
    def reset_usage(self) -> None:
        raise NotImplementedError


class LocalLlamaProvider(LLMProvider):

    def __init__(self):
        self._report_generator = None
        self._usage = None

    @property
    def report_generator(self):
        if self._report_generator is None:
            from app.rag.report_generator import ReportGenerator
            self._report_generator = ReportGenerator()
        return self._report_generator

    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        try:
            text, usage = self.report_generator.generate_with_usage(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            raise LLMProviderError(f"Local inference failed: {e}") from e

        if usage:
            self._usage = self._accumulate(self._usage, usage)
        return text

    def get_last_usage(self) -> Optional[dict]:
        return self._usage

    def reset_usage(self) -> None:
        self._usage = None

    @staticmethod
    def _accumulate(existing: Optional[dict], new: dict) -> dict:
        if not existing:
            return dict(new)
        return _merge_usage(existing, new, lambda a, b: a + b)


class HostedProvider(LLMProvider):

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = LLM_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL")
        if not self.base_url or not self.api_key or not self.model:
            raise LLMProviderError(
                "HostedProvider requires LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL "
                "(env vars or constructor args) to all be set."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self._usage = None

    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        last_error = None
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }
        if self.model.startswith("openai/gpt-oss"):
            # gpt-oss is a reasoning model: unset, its hidden chain-of-thought
            # (returned in a separate `reasoning` field, billed out of the
            # same max_tokens budget) scales to fill whatever budget it's
            # given rather than costing a fixed tax -- confirmed live: at
            # default effort it burned all of a 200-token budget on
            # reasoning alone and never reached the answer. "low" bounds it
            # (~15-35 reasoning tokens on this codebase's short prompts)
            # so callers with tight max_new_tokens (company_resolver.py's
            # 8-40, chat_router.py's 20) still get real content back.
            # Sending this field to a non-reasoning model (e.g. allam-2-7b)
            # is a hard 400 "not supported with this model", not a no-op --
            # confirmed live -- so it's gated to the one model family that
            # accepts and needs it.
            payload["reasoning_effort"] = "low"

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(self._backoff_delay(attempt))
                    continue
                raise LLMProviderError(
                    f"Hosted provider unreachable after {self.max_retries} retries: {last_error}"
                ) from e

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage")
                if usage:
                    self._usage = self._accumulate(self._usage, usage)
                return data["choices"][0]["message"]["content"]

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < self.max_retries:
                    time.sleep(self._backoff_delay(attempt, resp))
                    continue
                raise LLMProviderError(
                    f"Hosted provider failed after {self.max_retries} retries: {last_error}"
                )

            # Any other 4xx (401 bad key, 400 bad request, 404 unknown
            # model, ...) is never transient -- retrying a bad API key
            # 3 times with backoff just delays a failure that was
            # certain from the first response.
            raise LLMProviderError(
                f"Hosted provider returned HTTP {resp.status_code} (non-retryable): {resp.text[:200]}"
            )

        raise LLMProviderError(f"Hosted provider failed: {last_error}")

    def _backoff_delay(self, attempt: int, resp: Optional[requests.Response] = None) -> float:
        """Honors a real Retry-After header (seconds form) when the
        provider sends one -- Groq and most others do on 429. Ignoring
        it is how you get rate-limit-BANNED instead of politely
        throttled; the fixed exponential schedule is only a fallback
        for when no header is present, or it's in a form (e.g. an
        HTTP-date) this doesn't parse."""
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return _BACKOFF_BASE_SECONDS * (2 ** attempt)

    def get_last_usage(self) -> Optional[dict]:
        return self._usage

    def reset_usage(self) -> None:
        self._usage = None

    @staticmethod
    def _accumulate(existing: Optional[dict], new: dict) -> dict:
        if not existing:
            return dict(new)
        return _merge_usage(existing, new, lambda a, b: a + b)


_provider = None

# Default ON, deliberately -- this is training-data collection for a
# future distillation pass (every real prompt/response pair this
# deployment ever sees), not a debug-only nice-to-have someone has to
# remember to enable. Set LOG_LLM_CALLS=false to disable.
_LOG_LLM_CALLS_ENABLED = os.environ.get("LOG_LLM_CALLS", "true").strip().lower() not in ("false", "0", "no")


def _log_llm_call(provider: str, model: Optional[str], prompt: str, response: Optional[str],
                   usage: Optional[dict], latency_seconds: float, error: Optional[str] = None) -> None:
    """
    One JSON object per line, one file per UTC day, under LLM_LOG_DIR
    (app/core/paths.py -- the persistent volume in production, not the
    container filesystem, so this survives a redeploy the same way
    jobs.db does). Deliberately the full prompt and full response, not
    a summary or a truncated preview -- a distillation pass needs the
    exact text a real call actually saw and produced, and a record
    that's missing either half of the pair isn't replayable at all.

    Best-effort only: logging must never be the reason a real request
    fails. A write failure here (disk full, volume not mounted, a
    permissions issue) is swallowed, not raised -- the caller already
    has (or is about to raise) the actual LLM result/error, and that's
    what matters to the request in flight.
    """
    if not _LOG_LLM_CALLS_ENABLED:
        return
    try:
        LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "response": response,
            "usage": usage,
            "latency_seconds": round(latency_seconds, 3),
            "error": error,
        }
        log_path = LLM_LOG_DIR / f"llm_calls_{now:%Y-%m-%d}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _install_call_logging(provider: LLMProvider) -> None:
    """
    Wraps `provider.generate` with an instance-level attribute, not a
    wrapping class -- `isinstance(get_llm_provider(), LocalLlamaProvider)`
    (used elsewhere, including app/tests/test_llm_provider.py) still
    holds true, since this is still the real, unwrapped provider
    object; only its bound `.generate` is shadowed by an instance
    attribute of the same name, which every call site (LLMPlanner,
    narrative_builder.py, company_resolver.py) picks up transparently
    since none of them cache the method reference across calls.

    Usage is logged as the DELTA this specific call added, not
    get_last_usage()'s running total -- get_last_usage() accumulates
    across calls until reset_usage() (see LocalLlamaProvider/
    HostedProvider's own docstrings), so reading it straight after one
    call would double-count every call after the first.
    """
    original_generate = provider.generate

    def logged_generate(prompt: str, max_new_tokens: int = 700) -> str:
        usage_before = provider.get_last_usage() or {}
        start = time.time()
        response = None
        error = None
        try:
            response = original_generate(prompt, max_new_tokens=max_new_tokens)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            # The whole block is best-effort, matching _log_llm_call's own
            # "logging must never be the reason a real request fails"
            # contract -- this runs in a `finally` after `try` already hit
            # `return response`, so an exception raised here doesn't just
            # fail to log, it REPLACES the real (already-successful)
            # response the caller was about to receive. Confirmed the hard
            # way: the usage-delta math used to assume every usage field
            # was a flat int, which crashed here (and silently ate every
            # otherwise-successful response) the moment a model's usage
            # payload nested a nested dict field under it (Groq's gpt-oss/
            # qwen reasoning models' completion_tokens_details).
            try:
                latency = time.time() - start
                usage_after = provider.get_last_usage() or {}
                call_usage = _merge_usage(usage_before, usage_after, lambda a, b: b - a) or None
                _log_llm_call(
                    provider=type(provider).__name__,
                    model=getattr(provider, "model", None) or "local-llama",
                    prompt=prompt,
                    response=response,
                    usage=call_usage,
                    latency_seconds=latency,
                    error=error,
                )
            except Exception:
                pass

    provider.generate = logged_generate


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        provider_name = os.environ.get("LLM_PROVIDER", "hosted")
        _provider = LocalLlamaProvider() if provider_name == "local" else HostedProvider()
        if _LOG_LLM_CALLS_ENABLED:
            _install_call_logging(_provider)
    return _provider


def is_local_provider() -> bool:
    """Single source of truth for "are we local or hosted" -- read by
    app/api/jobs.py's concurrency policy instead of re-checking
    LLM_PROVIDER itself, so the two can't drift out of sync."""
    return os.environ.get("LLM_PROVIDER", "hosted") == "local"
