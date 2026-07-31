"use client";

import { useState } from "react";
import { useResearch } from "@/lib/useResearch";
import ReportView from "@/components/ReportView";
import ResearchProgress from "@/components/ResearchProgress";
import InstallPrompt from "@/components/InstallPrompt";

const QUICK_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT"];

export default function Home() {
  const [query, setQuery] = useState("");
  const { status, jobId, result, errorMessage, latencySeconds, submit } = useResearch();

  const isBusy = status === "submitting" || status === "running";

  function run(q: string) {
    if (q.trim() && !isBusy) submit(q.trim());
  }

  return (
    <div className="min-h-screen bg-bg">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <InstallPrompt />

        {/* Header */}
        <div className="flex items-center justify-between border-b border-border pb-3">
          <span className="font-mono text-base font-bold tracking-wide text-text">FINSIGHT</span>
          <span className="flex h-7 w-7 items-center justify-center rounded-full border border-border font-mono text-[11px] text-muted">
            AI
          </span>
        </div>

        {!isBusy && status !== "done" && (
          <p className="mt-4 text-sm text-muted">
            Autonomous Financial Intelligence Platform for institutional-style equity research. Not a
            general-purpose financial chatbot — name a publicly listed company to begin.
          </p>
        )}

        {/* Search */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run(query);
          }}
          className="mt-5"
        >
          <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3.5 py-3">
            <span className="font-mono text-sm font-bold text-accent">&gt;</span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="ask about a ticker or thesis..."
              disabled={isBusy}
              className="flex-1 bg-transparent font-mono text-sm text-text placeholder:text-muted focus:outline-none disabled:opacity-60"
            />
          </div>
          <div className="mt-2.5 flex flex-wrap gap-2">
            {QUICK_TICKERS.map((t) => (
              <button
                key={t}
                type="button"
                disabled={isBusy}
                onClick={() => {
                  const q = `Should I buy ${t}?`;
                  setQuery(q);
                  run(q);
                }}
                className="rounded border border-border bg-card px-2.5 py-1 font-mono text-[11px] font-semibold text-muted hover:border-accent hover:text-accent disabled:opacity-50"
              >
                {t}
              </button>
            ))}
          </div>
          <button
            type="submit"
            disabled={isBusy || !query.trim()}
            className="mt-3 w-full rounded-lg bg-accent py-2.5 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isBusy ? "RUNNING..." : "RUN RESEARCH AGENT"}
          </button>
        </form>

        {isBusy && <ResearchProgress question={query} />}

        {status === "error" && errorMessage && (
          <div className="mt-6 rounded-lg border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-danger">
            {errorMessage}
          </div>
        )}

        {status === "done" && result && jobId && (
          <ReportView result={result} jobId={jobId} latencySeconds={latencySeconds} />
        )}

        <p className="mt-16 text-center font-mono text-[10px] text-dim">
          LLM Planner + RAG over live SEC filings, ChromaDB, FinBERT and DCF valuation tools
        </p>
      </div>
    </div>
  );
}
