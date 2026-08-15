"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useResearch } from "@/lib/useResearch";
import ReportView from "@/components/ReportView";
import ResearchProgress, { STEPS } from "@/components/ResearchProgress";
import InstallPrompt from "@/components/InstallPrompt";
import OnboardingTour from "@/components/OnboardingTour";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import SearchOverlay from "@/components/SearchOverlay";
import NotificationsPanel from "@/components/NotificationsPanel";
import IndicesCarousel from "@/components/IndicesCarousel";
import CurrencyTracker from "@/components/CurrencyTracker";
import Portfolio from "@/components/Portfolio";
import OrderTicket from "@/components/OrderTicket";
import MarketNews from "@/components/MarketNews";
import MarketMovers from "@/components/MarketMovers";
import ScoreboardHomeCard from "@/components/ScoreboardHomeCard";
import SentimentGauge from "@/components/SentimentGauge";
import VoiceInputButton from "@/components/VoiceInputButton";
import HomeAssistant from "@/components/HomeAssistant";

const QUICK_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT"];

export default function Home() {
  return (
    <AuthGate>
      {({ email, displayName }) => (
        // useSearchParams requires a Suspense boundary around whatever
        // reads it, or the production build bails out of static
        // optimization with a hard error -- this page is fully client
        // rendered already (every child below uses hooks), so the
        // fallback is never actually visible in practice.
        <Suspense fallback={null}>
          <ResearchPage email={email} displayName={displayName} />
        </Suspense>
      )}
    </AuthGate>
  );
}

// Prefers the user's own editable display_name (Profile page) over a
// guess -- that guess (capitalize up to the first separator in the
// email's local-part) reads as a typo for any email that doesn't
// cleanly split into a real name, e.g. "sshivaum@..." -> "Sshivaum".
// Falls back to the raw local-part if no separator is found.
function greetingName(email: string | null, displayName: string | null): string {
  if (displayName) return displayName;
  if (!email) return "there";
  const local = email.split("@")[0];
  const first = local.split(/[._-]/)[0];
  return first.charAt(0).toUpperCase() + first.slice(1);
}

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="M20 20l-4.8-4.8" strokeLinecap="round" />
    </svg>
  );
}

function greetingForHour(hour: number): string {
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
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

function ResearchPage({ email, displayName }: { email: string | null; displayName: string | null }) {
  const [query, setQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const { status, jobId, result, errorMessage, latencySeconds, fromCache, submit, loadJob, reset } = useResearch();
  const searchParams = useSearchParams();

  // Cross-page navigation target: the /reports page links here as
  // /?job=<id> when you click a past report, since useResearch's state
  // lives in this component instance, not anywhere global -- the URL
  // is the one thing both pages can actually share. Runs once per
  // distinct job id (not on every render), and deliberately does NOT
  // clear the query param afterward -- a reload of this exact URL
  // should keep showing the same report, not silently drop back to
  // the search form.
  const jobParam = searchParams.get("job");
  useEffect(() => {
    if (jobParam) loadJob(jobParam);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobParam]);

  // Landing target for SearchOverlay's "Research this" fallback
  // (/?q=<query> when nothing matched an existing report or watchlist
  // ticker) -- prefills and immediately runs it, since typing into the
  // overlay and hitting that button is already an explicit act of
  // intent, not something that needs a second confirmation here.
  const qParam = searchParams.get("q");
  useEffect(() => {
    if (qParam) {
      setQuery(qParam);
      submit(qParam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qParam]);

  const isBusy = status === "submitting" || status === "running";

  function run(q: string) {
    if (q.trim() && !isBusy) submit(q.trim());
  }

  const hour = new Date().getHours();

  return (
    <div className="min-h-screen bg-bg pb-20">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <InstallPrompt />
        <OnboardingTour />

        {/* Header */}
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] font-bold tracking-wide text-dim">FINSIGHT</span>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setShowSearch(true)}
              aria-label="Search reports and watchlist"
              className="text-muted hover:text-accent"
            >
              <SearchIcon />
            </button>
            <NotificationsPanel />
          </div>
        </div>

        {showSearch && <SearchOverlay onClose={() => setShowSearch(false)} />}

        <h1 className="mt-3 font-mono text-xl font-bold text-text">
          {greetingForHour(hour)}, {greetingName(email, displayName)}
        </h1>

        {/* RAG research agent -- the app's core interaction, kept above
            the supplementary market widgets below rather than buried
            under them. */}
        {!isBusy && status !== "done" && (
          <p className="mt-2 text-sm text-muted">
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
            <VoiceInputButton onTranscript={setQuery} disabled={isBusy} />
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

        {!isBusy && status !== "done" && status !== "error" && <PipelinePreview />}

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
          <>
            <button
              type="button"
              onClick={() => {
                setQuery("");
                reset();
              }}
              className="mt-6 font-mono text-xs font-bold text-muted hover:text-accent"
            >
              &larr; BACK TO SEARCH
            </button>
            <ReportView result={result} jobId={jobId} latencySeconds={latencySeconds} />
          </>
        )}

        {/* Conversational assistant -- active the instant the app
            opens, not buried behind the separate Chat tab. Same
            hook/session-loop as the full-page /chat view. */}
        <HomeAssistant userName={greetingName(email, displayName)} />

        <ScoreboardHomeCard />

        <IndicesCarousel />

        <CurrencyTracker />

        <Portfolio />

        <OrderTicket />

        <MarketMovers />

        <SentimentGauge />

        <MarketNews />

        <p className="mt-16 text-center font-mono text-[10px] text-dim">
          LLM Planner + RAG over live SEC filings, ChromaDB, FinBERT and DCF valuation tools
        </p>
      </div>

      <BottomNav />
    </div>
  );
}
