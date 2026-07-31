"use client";

import { useCallback, useRef, useState } from "react";
import type { ApiErrorBody, JobResponse, ResearchResult } from "./types";

const POLL_INTERVAL_MS = 3000;

type Status = "idle" | "submitting" | "running" | "done" | "error";

interface State {
  status: Status;
  jobId: string | null;
  result: ResearchResult | null;
  errorMessage: string | null;
  latencySeconds: number | null;
}

export function useResearch() {
  const [state, setState] = useState<State>({
    status: "idle",
    jobId: null,
    result: null,
    errorMessage: null,
    latencySeconds: null,
  });
  const cancelledRef = useRef(false);

  const submit = useCallback(async (question: string) => {
    cancelledRef.current = false;
    setState({ status: "submitting", jobId: null, result: null, errorMessage: null, latencySeconds: null });
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
        setState((s) => ({
          ...s,
          status: "done",
          result: data.result!,
          latencySeconds: (Date.now() - start) / 1000,
        }));
        return;
      }

      if (data.status === "error") {
        const message =
          data.error_code === "TICKER_NOT_FOUND"
            ? `Couldn't find market data for this company: ${data.error_message || ""} The company may be delisted, foreign-listed, or the name didn't resolve to a real ticker -- please check the spelling and try again.`
            : data.error_message || "Something went wrong while researching this.";
        setState((s) => ({ ...s, status: "error", errorMessage: message }));
        return;
      }
      // status is "queued" or "running" -- keep polling
    }
  }, []);

  const reset = useCallback(() => {
    cancelledRef.current = true;
    setState({ status: "idle", jobId: null, result: null, errorMessage: null, latencySeconds: null });
  }, []);

  return { ...state, submit, reset };
}
