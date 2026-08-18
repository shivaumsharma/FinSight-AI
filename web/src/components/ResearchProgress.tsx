"use client";

import { useEffect, useState } from "react";
import type { JobProgress } from "@/lib/types";

// Fallback (simulated-timer) stages -- shown when `progress` is null,
// i.e. the job hasn't reached the real backend signal yet (early
// setup), or is running under the langgraph orchestrator, which does
// NOT wire up on_step (see app/api/jobs.py's _run_job -- a deliberate,
// documented scope limit, not an oversight; LangGraph's structurally
// different execution model made real per-step signal out of scope for
// this pass). This list advances on a fixed clock as an honest
// approximation of typical timing (observed ~90-190s total this
// session), not a claim of live per-step status, and it never reaches
// the final step until the real result actually comes back.
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

// Human labels for every real tool name the planner can choose (see
// app/tools/tool_registry.py's registered tools and
// app/agents/agent_constants.py's TRAILING_TOOLS -- together the full
// set ResearchAgent.run()'s plan can ever contain). Anything not listed
// here (a future tool added to the registry without a matching label
// update here) falls back to a title-cased version of the raw
// tool_name below, rather than crashing or rendering blank.
const TOOL_LABELS: Record<string, string> = {
  market_data_tool: "Market data & financial statements",
  valuation_tool: "WACC · FCFF · DCF valuation",
  rag_tool: "SEC filings & RAG retrieval",
  sentiment_tool: "FinBERT sentiment scoring",
  comparison_tool: "Peer comparison",
  institutional_consensus_tool: "Institutional consensus",
  news_tool: "News selection & sentiment",
  report_tool: "Narrative synthesis & report generation",
  evaluation_tool: "Evaluation & confidence scoring",
};

function labelForTool(toolName: string): string {
  if (TOOL_LABELS[toolName]) return TOOL_LABELS[toolName];
  // Title-cased fallback for an unmapped tool_name (e.g.
  // "some_new_tool" -> "Some New Tool") -- never crashes, never
  // renders a raw snake_case string to the user.
  return toolName
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function SimulatedProgress({ ticker, question }: { ticker?: string; question: string }) {
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

function RealProgress({
  ticker,
  question,
  progress,
}: {
  ticker?: string;
  question: string;
  progress: JobProgress;
}) {
  const { plan, completed, current } = progress;
  const completedSet = new Set(completed);
  const currentIndex = current ? plan.indexOf(current) : -1;
  // A step counts as "done" once the backend has reported it complete
  // OR (defensively) it sits earlier in the plan than the current step
  // -- the latter only matters if a step were ever skipped without an
  // explicit completed entry, which _on_step's own bookkeeping in
  // jobs.py doesn't do today, but this keeps the display honest even if
  // that ever changes.
  const doneCount = plan.filter(
    (name, i) => completedSet.has(name) || (currentIndex >= 0 && i < currentIndex)
  ).length;
  const displayStep = current ? Math.min(doneCount + 1, plan.length) : Math.max(doneCount, 1);

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
            {displayStep}/{plan.length}
          </span>
        </div>
        <div className="mt-2 font-mono text-[11px] text-muted">
          STEP {displayStep} OF {plan.length} · LIVE
        </div>
      </div>

      <div className="mt-5 flex flex-col">
        {plan.map((toolName, i) => {
          const done = completedSet.has(toolName) || (currentIndex >= 0 && i < currentIndex);
          const active = toolName === current;
          return (
            <div key={`${toolName}-${i}`} className="flex items-center gap-3 py-2">
              <span
                className={`font-mono text-sm ${done ? "text-accent" : active ? "text-accent" : "text-dim"}`}
              >
                {done ? "✓" : active ? "●" : "○"}
              </span>
              <span className={`font-mono text-xs ${active ? "font-semibold text-text" : done ? "text-muted" : "text-dim"}`}>
                {labelForTool(toolName)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ResearchProgress({
  ticker,
  question,
  progress,
}: {
  ticker?: string;
  question: string;
  progress?: JobProgress | null;
}) {
  // Real signal wins whenever the backend has actually reported a
  // non-empty plan (hand_rolled orchestrator, past the planning stage);
  // otherwise this falls back to the exact previous simulated-timer
  // behavior unchanged -- covers a job that's still queued/in early
  // setup, and the langgraph orchestrator, which never populates
  // progress at all (see this file's own TOOL_LABELS comment above).
  if (progress && progress.plan.length > 0) {
    return <RealProgress ticker={ticker} question={question} progress={progress} />;
  }
  return <SimulatedProgress ticker={ticker} question={question} />;
}
