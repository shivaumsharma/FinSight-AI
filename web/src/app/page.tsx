"use client";

import { useState } from "react";
import { useResearch } from "@/lib/useResearch";
import { usePushNotifications } from "@/lib/usePushNotifications";
import ReportView from "@/components/ReportView";
import ResearchProgress from "@/components/ResearchProgress";
import InstallPrompt from "@/components/InstallPrompt";
import AuthGate from "@/components/AuthGate";

const QUICK_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT"];

export default function Home() {
  return <AuthGate>{({ userId, logout }) => <ResearchPage userId={userId} onLogout={logout} />}</AuthGate>;
}

// Purely additive to the header -- not gating anything, not shown at
// all when the browser doesn't support push (still "checking" is
// treated the same as unsupported for a beat, avoiding a flash of the
// button before the initial support/permission check resolves).
function NotificationToggle() {
  const { status, subscribe, unsubscribe } = usePushNotifications();

  if (status === "unsupported" || status === "checking") return null;

  if (status === "denied") {
    return (
      <span
        className="font-mono text-[10px] text-dim"
        title="Notifications are blocked in your browser settings for this site"
      >
        NOTIFICATIONS BLOCKED
      </span>
    );
  }

  if (status === "subscribed") {
    return (
      <button
        type="button"
        onClick={unsubscribe}
        title="Notifications on -- click to turn off"
        className="font-mono text-[10px] font-bold text-accent hover:text-muted"
      >
        NOTIFICATIONS ON
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={subscribe}
      className="font-mono text-[10px] font-bold text-muted hover:text-accent"
    >
      ENABLE NOTIFICATIONS
    </button>
  );
}

function ResearchPage({ userId, onLogout }: { userId: string; onLogout: () => void }) {
  const [query, setQuery] = useState("");
  const { status, jobId, result, errorMessage, latencySeconds, fromCache, submit } = useResearch();

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
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] text-dim" title={userId}>
              {userId.slice(0, 8)}
            </span>
            <NotificationToggle />
            <button
              type="button"
              onClick={onLogout}
              className="font-mono text-[10px] font-bold text-muted hover:text-danger"
            >
              LOGOUT
            </button>
            <span className="flex h-7 w-7 items-center justify-center rounded-full border border-border font-mono text-[11px] text-muted">
              AI
            </span>
          </div>
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

        {status === "done" && fromCache && (
          <div className="mt-6 rounded-lg border border-amber-900/60 bg-amber-950/40 px-4 py-3 text-xs text-warn">
            You&apos;re offline -- showing your last report. Reconnect to run new research.
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
