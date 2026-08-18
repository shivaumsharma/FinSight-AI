"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiErrorBody, JobProgress, JobResponse, ResearchResult } from "./types";

const POLL_INTERVAL_MS = 3000;
const LAST_REPORT_KEY = "finsight_last_report";

type Status = "idle" | "submitting" | "running" | "done" | "error";

interface State {
  status: Status;
  jobId: string | null;
  result: ResearchResult | null;
  errorMessage: string | null;
  latencySeconds: number | null;
  // True only when `result` was restored from localStorage on an
  // offline page load (see the mount effect below), not from a real
  // just-completed job -- ReportView/page.tsx use this to show an
  // "offline, showing your last report" banner instead of presenting
  // stale data as if it were fresh.
  fromCache: boolean;
  // Real per-step progress from the polled JobResponse (see
  // JobProgress's own comment in types.ts) -- null while queued, once
  // done/error, or whenever the backend itself has no real signal to
  // report yet (early setup, or the langgraph orchestrator). Same
  // lifecycle as jobId: reset alongside it, updated on every poll tick.
  progress: JobProgress | null;
}

const IDLE_STATE: State = {
  status: "idle",
  jobId: null,
  result: null,
  errorMessage: null,
  latencySeconds: null,
  fromCache: false,
  progress: null,
};

function persistLastReport(jobId: string, result: ResearchResult) {
  try {
    localStorage.setItem(LAST_REPORT_KEY, JSON.stringify({ jobId, result }));
  } catch {
    // localStorage can throw (Safari private mode, quota exceeded) --
    // losing offline-restore capability silently is fine, it must
    // never break a just-completed, already-successful job from
    // rendering normally.
  }
}

export function useResearch() {
  // Always starts idle, on both server and client render -- reading
  // navigator.onLine/localStorage here (in the useState initializer)
  // would run during SSR too, where neither exists, producing a
  // server/client markup mismatch the moment a client actually is
  // offline. The mount effect below runs client-only, after
  // hydration, and is where the offline-restore decision belongs
  // instead (same pattern useAuth.ts's checkSession already uses for
  // its own post-mount check).
  const [state, setState] = useState<State>(IDLE_STATE);
  const cancelledRef = useRef(false);

  useEffect(() => {
    if (typeof navigator === "undefined" || navigator.onLine) return;
    try {
      const raw = localStorage.getItem(LAST_REPORT_KEY);
      if (!raw) return;
      const { jobId, result } = JSON.parse(raw);
      if (jobId && result) {
        setState({ status: "done", jobId, result, errorMessage: null, latencySeconds: null, fromCache: true, progress: null });
      }
    } catch {
      // Malformed/missing cache entry -- stay idle, same as a normal
      // first visit.
    }
  }, []);

  const submit = useCallback(async (question: string) => {
    cancelledRef.current = false;
    setState({ ...IDLE_STATE, status: "submitting" });
    const start = Date.now();

    let submitResp: Response;
    try {
      submitResp = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
    } catch {
      setState((s) => ({ ...s, status: "error", errorMessage: "Couldn't reach the research service. Check your connection and try again." }));
      return;
    }

    if (!submitResp.ok) {
      const body: ApiErrorBody = await submitResp.json().catch(() => ({ code: "UNKNOWN", message: "Something went wrong." }));
      const message =
        body.code === "NO_COMPANY_DETECTED"
          ? "FinSight AI specializes in company and investment research. No publicly listed company was detected in your query. Please provide a company name to begin the analysis."
          : body.message || "Something went wrong while researching this.";
      setState((s) => ({ ...s, status: "error", errorMessage: message }));
      return;
    }

    const { job_id: jobId } = await submitResp.json();
    setState((s) => ({ ...s, status: "running", jobId }));

    // Polling loop -- mirrors streamlit_app.py's _poll_job(), just as a
    // client-side loop instead of a server-side blocking one, since a
    // browser tab can't hold a Python-style synchronous wait.
    while (!cancelledRef.current) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      if (cancelledRef.current) return;

      let pollResp: Response;
      try {
        pollResp = await fetch(`/api/research/${jobId}`, { cache: "no-store" });
      } catch {
        setState((s) => ({ ...s, status: "error", errorMessage: "Lost connection to the research service while waiting for results." }));
        return;
      }

      const data: JobResponse = await pollResp.json();

      if (data.status === "done" && data.result) {
        persistLastReport(jobId, data.result);
        setState((s) => ({
          ...s,
          status: "done",
          result: data.result!,
          latencySeconds: (Date.now() - start) / 1000,
          fromCache: false,
          progress: null,
        }));
        return;
      }

      if (data.status === "error") {
        const message =
          data.error_code === "TICKER_NOT_FOUND"
            ? `Couldn't find market data for this company: ${data.error_message || ""} The company may be delisted, foreign-listed, or the name didn't resolve to a real ticker -- please check the spelling and try again.`
            : data.error_message || "Something went wrong while researching this.";
        setState((s) => ({ ...s, status: "error", errorMessage: message, progress: null }));
        return;
      }
      // status is "queued" or "running" -- keep polling. data.progress is
      // only ever populated once status === "running" (see main.py's GET
      // /v1/research/{job_id}), undefined/null otherwise -- normalized to
      // null here so ResearchProgress.tsx has one falsy shape to check,
      // not "undefined vs null" depending on which status this tick was.
      setState((s) => ({ ...s, progress: data.progress ?? null }));
    }
  }, []);

  // For redisplaying an already-completed report (a Recent Reports row
  // click) -- one GET, no polling loop, no new POST /v1/research. Only
  // handles the "done" case: a Recent Reports row is only ever shown
  // for a job that already completed (see db.list_recent_jobs's own
  // status='done' filter), so a queued/running/error response here
  // would mean the row itself was stale, not something this needs to
  // render a progress UI for.
  const loadJob = useCallback(async (jobId: string) => {
    cancelledRef.current = true;
    let resp: Response;
    try {
      resp = await fetch(`/api/research/${jobId}`, { cache: "no-store" });
    } catch {
      setState((s) => ({ ...s, status: "error", errorMessage: "Couldn't reach the research service. Check your connection and try again." }));
      return;
    }
    const data: JobResponse = await resp.json();
    if (data.status === "done" && data.result) {
      setState({
        status: "done",
        jobId,
        result: data.result,
        errorMessage: null,
        latencySeconds: null,
        fromCache: false,
        progress: null,
      });
    }
  }, []);

  const reset = useCallback(() => {
    cancelledRef.current = true;
    // Only clears the current view, not the localStorage entry itself
    // -- the cached last report should still be there to restore on
    // the next offline visit, even after the user starts a fresh
    // search in this session.
    setState(IDLE_STATE);
  }, []);

  return { ...state, submit, loadJob, reset };
}
