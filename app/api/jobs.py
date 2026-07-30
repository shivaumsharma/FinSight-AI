"""
jobs.py

Runs research jobs in a background worker and writes their outcome to
SQLite (app/api/db.py).

Concurrency: max_workers is resolved dynamically (see
_resolve_max_workers), not a fixed 1. In local mode it's hard-pinned to
1 in code regardless of any env var -- `llama-cpp-python`'s `Llama`
context (app/rag/report_generator.py) is shared as a process-wide
singleton and not documented as safe for concurrent generate() calls
from multiple threads; a bigger pool there wouldn't add real
concurrency, it would add a real risk of corrupting/crashing concurrent
local-model calls. In hosted mode the LLM itself is no longer the
blocker, but MAX_CONCURRENT_JOBS still *defaults* to 1 -- raising it is
a deliberate, separate step that requires auditing the rest of the
pipeline for concurrent-access safety first (SEC EDGAR's hard rate
limit, Finnhub/News free-tier caps, the shared FinBERT model, the
shared ChromaDB client, the narrative Redis cache's check-then-set
path -- none of that is audited by this change). See
_resolve_max_workers()'s own docstring.

Writes the PDF to reports/{job_id}.pdf on completion and stores the
*path*, not the bytes, in SQLite -- see db.py's module docstring for
why persistence exists at all (surviving a restart), which a
still-in-memory PDF blob would have quietly defeated.
"""

import os
from typing import Callable, Optional

from app.api import db
from app.api.serialization import context_to_api_dict
from app.core.llm_provider import LLMProviderError, get_llm_provider, is_local_provider
from app.core.logger import ResearchLogger
from app.core.paths import REPORTS_DIR, RESEARCH_LOG_DIR  # noqa: F401 -- re-exported; see app/core/paths.py
from concurrent.futures import ThreadPoolExecutor


def _resolve_max_workers() -> int:
    """
    Local mode: hard-pinned to 1, ignoring MAX_CONCURRENT_JOBS entirely
    -- "enforced in code, not left to config" means a misconfigured env
    var (MAX_CONCURRENT_JOBS=99 with LLM_PROVIDER=local) must not be
    able to raise this, not just that the default is safe.

    Hosted mode: MAX_CONCURRENT_JOBS env var, defaulting to 1 -- see
    this module's docstring for why the default doesn't jump to a
    bigger number just because the LLM stopped being the blocker.
    Explicit opt-in (setting the env var) still works once the rest of
    the pipeline has actually been audited for concurrent access.
    """
    if is_local_provider():
        return 1
    return int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))


_executor = ThreadPoolExecutor(max_workers=_resolve_max_workers())


def _hand_rolled_factory():
    from app.agents.research_agent import ResearchAgent
    return ResearchAgent()


def _langgraph_factory():
    from app.agents.langgraph_agent import LangGraphResearchAgent
    return LangGraphResearchAgent()


# Factory functions, not the classes directly -- ResearchAgent's own
# import chain pulls in ToolRegistry's full model/network stack (torch,
# transformers, chromadb, llama-cpp-python) via all 9 tool classes at
# import time. Deferring the import into these factories means
# `import app.api.jobs` (and anything that imports it, like
# app.api.main) stays free of that stack until a job actually runs and
# a real orchestrator is genuinely needed -- app/tests/test_api.py
# overrides these entries with lightweight stub agents and never
# triggers the real imports at all, matching the same "defer the
# expensive import" pattern already used in langgraph_agent.py.
ORCHESTRATORS = {
    "hand_rolled": _hand_rolled_factory,
    "langgraph": _langgraph_factory,
}


def _run_job(job_id: str, question: str, orchestrator_name: str,
             agent_factory: Optional[Callable] = None) -> None:
    """
    Runs entirely inside a ThreadPoolExecutor worker thread with no
    caller ever calling .result() on the submitted Future -- so ANY
    exception that escapes this function, including a bug in this
    function's OWN error-handling code, is silently discarded by
    concurrent.futures with no log, no trace, nothing. The job's row
    would then sit at RUNNING until the timeout sweep in db.py catches
    it minutes later. The outer try/except below exists specifically
    as a last-resort safety net against that -- it wraps the more
    specific handling, not the other way around, so a bug in the
    specific handlers still gets caught and reported instead of
    propagating past this function silently. (Found the hard way: an
    earlier version of this function had exactly this bug -- a typo'd
    reference inside the `except TickerNotFoundError` clause raised its
    own NameError while handling the original exception, and every job
    in the test suite hung at RUNNING forever until that was traced
    down to a missing import.)
    """
    db.mark_running(job_id)

    # Reset before the run, read after -- safe as a shared-singleton
    # reset specifically BECAUSE _resolve_max_workers() defaults to a
    # single worker even in hosted mode (see this module's docstring).
    # Concurrent jobs would need per-job usage tracking, which this
    # does not attempt -- a documented limitation of today's default,
    # not an oversight; raising MAX_CONCURRENT_JOBS without also fixing
    # this would make usage numbers meaningless across overlapping jobs.
    provider = get_llm_provider()
    provider.reset_usage()

    try:
        try:
            from app.data.market_data import TickerNotFoundError
            agent = (agent_factory or ORCHESTRATORS[orchestrator_name])()
            context = agent.run(question)

            # Moved here from streamlit_app.py: this needs to happen for
            # every job regardless of which client submitted it (Streamlit,
            # a future mobile client, curl), not just Streamlit-originated
            # ones -- was previously called client-side, which only worked
            # because Streamlit ran the pipeline in-process and had the raw
            # ResearchContext to hand. See app/core/logger.py's own
            # docstring for why this logging exists at all.
            try:
                ResearchLogger(log_directory=str(RESEARCH_LOG_DIR)).save(context)
            except Exception as e:
                print(f"[job {job_id}] ResearchLogger.save failed (non-fatal): {e}")

            pdf_path = None
            if context.pdf_bytes:
                REPORTS_DIR.mkdir(exist_ok=True)
                pdf_path = str(REPORTS_DIR / f"{job_id}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(context.pdf_bytes)

            result = context_to_api_dict(context)
            usage = provider.get_last_usage()
            result["llm_usage"] = usage
            if usage:
                print(f"[job {job_id}] {context.ticker}: llm_usage={usage}")
            db.mark_done(job_id, result, pdf_path)

        except TickerNotFoundError as e:
            db.mark_error(job_id, "TICKER_NOT_FOUND", str(e))
        except LLMProviderError as e:
            # The structured error the task asks for -- without this,
            # a hosted-provider outage falls into the generic
            # PIPELINE_ERROR catch-all below, losing the distinction
            # between "the pipeline has a bug" and "the LLM backend is
            # down," which a real client needs to tell apart (retry
            # later vs. report a bug).
            db.mark_error(job_id, "LLM_PROVIDER_ERROR", str(e))
        except Exception as e:
            db.mark_error(job_id, "PIPELINE_ERROR", str(e))

    except Exception as e:
        # The last-resort net described above -- a failure in the
        # handling above itself. Best-effort: db.mark_error could
        # theoretically fail too (e.g. disk full), in which case there
        # is genuinely nothing left to do but let the timeout sweep
        # catch it eventually.
        try:
            db.mark_error(job_id, "INTERNAL_ERROR", f"Unhandled error in job runner: {e}")
        except Exception:
            pass


def submit_job(
    question: str, orchestrator: str = "hand_rolled", user_id: str = "anonymous",
    agent_factory: Optional[Callable] = None,
) -> str:
    """
    Company-name validation (resolve_companies/NoCompanyDetectedError)
    happens in main.py's endpoint handler BEFORE this is called, not
    here -- a request that fails that check should never create a row
    or occupy the single worker slot at all.
    """
    job_id = db.create_job(question=question, orchestrator=orchestrator, user_id=user_id)
    _executor.submit(_run_job, job_id, question, orchestrator, agent_factory)
    return job_id
