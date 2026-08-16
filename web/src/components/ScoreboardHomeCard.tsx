"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { ScoreboardResponse } from "@/lib/types";

// Prominent, not buried -- this is the app's core credibility feature
// (does the model actually beat doing nothing?), so it gets a card at
// the top of Home, same visual weight as Portfolio/Market Movers,
// driving into the full /scoreboard page. Uses the 30-day window as
// the headline number (7-day is noisy, 90-day is slow to fill) --
// full breakdown across all three windows lives on the page itself.
export default function ScoreboardHomeCard() {
  const [data, setData] = useState<ScoreboardResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/scoreboard")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, []);

  const window30 = data?.windows["30"];
  const hasData = window30 && window30.sample_size > 0;
  const bestBaseline = hasData
    ? Math.max(window30.always_buy_accuracy_pct ?? -Infinity, window30.always_hold_accuracy_pct ?? -Infinity)
    : null;
  const modelColor =
    !hasData || bestBaseline === null
      ? "text-dim"
      : window30.model_accuracy_pct! > bestBaseline
        ? "text-accent"
        : window30.model_accuracy_pct! < bestBaseline
          ? "text-danger"
          : "text-warn";

  return (
    <Link
      href="/scoreboard"
      className="mt-4 block rounded-lg border border-border bg-card px-3.5 py-3 hover:border-accent"
    >
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">MODEL VS. NAIVE BASELINE</p>
        <span className="font-mono text-[10px] font-bold text-accent">VIEW SCOREBOARD &rarr;</span>
      </div>
      {error ? (
        <p className="mt-1.5 font-mono text-xs text-danger">Couldn&apos;t load the scoreboard. Try again.</p>
      ) : hasData ? (
        <p className="mt-1.5 font-mono text-xs text-text">
          30-day: <span className={`font-bold ${modelColor}`}>{window30!.model_accuracy_pct!.toFixed(1)}%</span> model
          vs <span className="text-muted">{bestBaseline!.toFixed(1)}%</span> best naive baseline (n=
          {window30!.sample_size})
        </p>
      ) : (
        <p className="mt-1.5 font-mono text-xs text-dim">Tracking real calls against naive baselines -- not enough data yet.</p>
      )}
    </Link>
  );
}
