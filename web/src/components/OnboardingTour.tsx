"use client";

import { useEffect, useState } from "react";

const SEEN_KEY = "finsight:onboarding-seen-v1";

const STEPS = [
  {
    title: "Research any company",
    body: "Name a ticker or ask a thesis and the agent runs a full institutional-style pipeline -- live financials, SEC filings, sentiment, and a DCF valuation -- typically in 1-3 minutes.",
  },
  {
    title: "Track what matters",
    body: "Watchlist and Portfolio keep live prices, ratings, and corporate actions on the tickers you care about, right on your home screen.",
  },
  {
    title: "Practice without risk",
    body: "Trade (Simulated) lets you place paper BUY/SELL orders that fill instantly at the live quote -- no real broker, no real money, ever.",
  },
  {
    title: "Stay on top of the market",
    body: "Top Movers, Research Sentiment, and Market News give you a full market pulse alongside your own holdings.",
  },
] as const;

// Shown once per browser (localStorage-gated, same pattern as the
// existing offline-report cache) after a user's first successful
// login -- introduces the four home-dashboard sections before they've
// had to guess what any of them do. Skippable at any step; finishing
// or skipping both permanently dismiss it.
export default function OnboardingTour() {
  const [show, setShow] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!window.localStorage.getItem(SEEN_KEY)) setShow(true);
  }, []);

  function dismiss() {
    window.localStorage.setItem(SEEN_KEY, "1");
    setShow(false);
  }

  if (!show) return null;

  const isLast = step === STEPS.length - 1;
  const current = STEPS[step];

  return (
    <div
      className="fixed inset-0 z-30 flex items-end justify-center bg-bg/80 backdrop-blur-sm sm:items-center"
      onClick={(e) => {
        if (e.target === e.currentTarget) dismiss();
      }}
    >
      <div className="w-full max-w-md rounded-t-2xl border border-border bg-card p-5 sm:rounded-2xl">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] tracking-wide text-dim">
            {step + 1} OF {STEPS.length}
          </span>
          <button type="button" onClick={dismiss} className="font-mono text-[10px] font-bold text-muted hover:text-accent">
            SKIP
          </button>
        </div>

        <h2 className="mt-4 font-mono text-base font-bold text-text">{current.title}</h2>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">{current.body}</p>

        <div className="mt-5 flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <span key={i} className={`h-1.5 w-1.5 rounded-full ${i === step ? "bg-accent" : "bg-border"}`} />
            ))}
          </div>
          <button
            type="button"
            onClick={() => (isLast ? dismiss() : setStep((s) => s + 1))}
            className="rounded-lg bg-accent px-4 py-2 font-mono text-xs font-bold text-bg"
          >
            {isLast ? "GET STARTED" : "NEXT"}
          </button>
        </div>
      </div>
    </div>
  );
}
