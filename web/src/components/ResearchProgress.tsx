"use client";

import { useEffect, useState } from "react";

// Real pipeline stages (app/api/jobs.py's ORCHESTRATORS -> ResearchAgent
// -> tool_trace order: market_data_tool, valuation_tool, rag_tool,
// sentiment_tool, institutional_consensus_tool, news_tool, report_tool,
// evaluation_tool). The backend only reports queued/running/done/error,
// no sub-step signal -- this list advances on a fixed clock as an
// honest approximation of typical timing (observed ~90-190s total this
// session), not a claim of live per-step status. It never reaches the
// final step until the real result actually comes back.
export const STEPS = [
  "Resolving ticker & company",
  "Market data & financial statements",
  "SEC filings & RAG retrieval",
  "FinBERT sentiment scoring",
  "Institutional consensus",
  "News selection & sentiment",
  "WACC · FCFF · DCF valuation",
  "Narrative synthesis & report generation",
];

const STEP_INTERVAL_MS = 14000;

export default function ResearchProgress({ ticker, question }: { ticker?: string; question: string }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const stepTimer = setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, STEPS.length - 1));
    }, STEP_INTERVAL_MS);
    const clock = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => {
      clearInterval(stepTimer);
      clearInterval(clock);
    };
  }, []);

  return (
    <div className="mt-6 rounded-lg border border-border bg-card p-6">
      <div className="flex items-center gap-3">
        <div>
          <div className="font-mono text-sm font-bold tracking-wide text-text">
            RESEARCHING{ticker ? ` ${ticker.toUpperCase()}` : ""}
          </div>
          <div className="text-xs text-muted">&quot;{question}&quot;</div>
        </div>
      </div>

      <div className="mt-6 flex flex-col items-center">
        <div className="relative flex h-20 w-20 items-center justify-center rounded-full border-[3px] border-border">
          <span className="absolute inset-0 animate-spin rounded-full border-[3px] border-transparent border-t-accent" />
          <span className="font-mono text-sm font-bold text-accent">
            {stepIndex + 1}/{STEPS.length}
          </span>
        </div>
        <div className="mt-2 font-mono text-[11px] text-muted">
          STEP {stepIndex + 1} OF {STEPS.length} · ~{elapsed}S ELAPSED (estimated)
        </div>
      </div>

      <div className="mt-5 flex flex-col">
        {STEPS.map((label, i) => {
          const done = i < stepIndex;
          const active = i === stepIndex;
          return (
            <div key={label} className="flex items-center gap-3 py-2">
              <span
                className={`font-mono text-sm ${done ? "text-accent" : active ? "text-accent" : "text-dim"}`}
              >
                {done ? "✓" : active ? "●" : "○"}
              </span>
              <span className={`font-mono text-xs ${active ? "font-semibold text-text" : done ? "text-muted" : "text-dim"}`}>
                {label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
