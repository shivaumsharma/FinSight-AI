"use client";

import { useState } from "react";
import Link from "next/link";
import { useResearch } from "@/lib/useResearch";
import { usePushNotifications } from "@/lib/usePushNotifications";
import ReportView from "@/components/ReportView";
import ResearchProgress, { STEPS } from "@/components/ResearchProgress";
import InstallPrompt from "@/components/InstallPrompt";
import AuthGate from "@/components/AuthGate";
import Watchlist from "@/components/Watchlist";
import RecentReports from "@/components/RecentReports";

const QUICK_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT"];

export default function Home() {
  return (
    <AuthGate>
      {({ userId, email, logout }) => <ResearchPage userId={userId} email={email} onLogout={logout} />}
    </AuthGate>
  );
}

// First two letters of the email's local-part, uppercased -- there's
// no real "name" field on a user (see db.py's users table), so this is
// the closest thing to initials this app can show without adding one.
function initialsFromEmail(email: string | null): string {
  if (!email) return "AI";
  return email.split("@")[0].slice(0, 2).toUpperCase();
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

// Idle-state preview of the same 8 stages ResearchProgress lights up
// once a job is actually running -- same STEPS list, same icon
// language (just all "○", nothing active yet), so a first-time visitor
// sees exactly what they're about to trigger before they trigger it,
// and the transition into a real run feels continuous rather than
// like a different UI taking over.
function PipelinePreview() {
  return (
    <div className="mt-6 rounded-lg border border-border-subtle bg-card/60 p-5">
      <p className="font-mono text-[10px] tracking-wide text-dim">WHAT RUNNING THIS DOES</p>
      <div className="mt-3 flex flex-col">
        {STEPS.map((label, i) => (
          <div key={label} className="flex items-center gap-3 py-1.5">
            <span className="font-mono text-sm text-dim">○</span>
            <span className="font-mono text-xs text-muted">
              {String(i + 1).padStart(2, "0")} · {label}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-dim">
        Every step above hits real data -- live market quotes, actual SEC filings, a FinBERT sentiment
        model, and a full WACC/FCFF/DCF valuation -- then an LLM synthesizes the result into a sourced
        report, typically in 1–3 minutes.
      </p>
    </div>
  );
}

function ResearchPage({
  userId,
  email,
  onLogout,
}: {
  userId: string;
  email: string | null;
  onLogout: () => void;
}) {
  const [query, setQuery] = useState("");
  const { status, jobId, result, errorMessage, latencySeconds, fromCache, submit, loadJob } = useResearch();

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
          <div className="flex items-baseline gap-2.5">
            <span className="font-mono text-base font-bold tracking-wide text-text">FINSIGHT</span>
            <span className="hidden font-mono text-[10px] tracking-wide text-dim sm:inline">
              AUTONOMOUS EQUITY RESEARCH
            </span>
          </div>
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
            <Link
              href="/profile"
              title="Profile"
              className="flex h-7 w-7 items-center justify-center rounded-full border border-border font-mono text-[11px] text-muted hover:border-accent hover:text-accent"
            >
              {initialsFromEmail(email)}
            </Link>
          </div>
        </div>

        {!isBusy && status !== "done" && (
          <p className="mt-4 text-sm text-muted">
            An LLM planning agent that builds institutional-style equity research on demand. Not a
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

        {!isBusy && status !== "done" && status !== "error" && (
          <>
            <Watchlist />
            <RecentReports onSelectReport={loadJob} />
            <PipelinePreview />
          </>
        )}

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
