"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { AccuracyTearsheetResponse } from "@/lib/types";

// Same "compact summary card on Home, driving into the full page"
// pattern as ScoreboardHomeCard -- see that component's own comment.
// Distinct data source though: Scoreboard is live production calls in
// recent windows, this is the one canonical BACKTESTED number (a
// broad, non-cherry-picked 1,000+ ticker historical evaluation -- see
// app/reporting/accuracy_tearsheet.py's own module docstring).
export default function AccuracyTearsheetHomeCard() {
  const [data, setData] = useState<AccuracyTearsheetResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/accuracy-tearsheet")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, []);

  const canonical = data?.canonical;
  const modelColor = !canonical
    ? "text-dim"
    : canonical.beats_baseline
      ? "text-accent"
      : "text-danger";

  return (
    <Link
      href="/accuracy-tearsheet"
      className="mt-4 block rounded-lg border border-border bg-card px-3.5 py-3 hover:border-accent"
    >
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">BACKTESTED ACCURACY (12-MONTH)</p>
        <span className="font-mono text-[10px] font-bold text-accent">VIEW TEARSHEET &rarr;</span>
      </div>
      {error ? (
        <p className="mt-1.5 font-mono text-xs text-danger">Couldn&apos;t load the tearsheet. Try again.</p>
      ) : data && !data.available ? (
        <p className="mt-1.5 font-mono text-xs text-dim">Backtest not yet run for this deployment.</p>
      ) : canonical ? (
        <p className="mt-1.5 font-mono text-xs text-text">
          <span className={`font-bold ${modelColor}`}>{canonical.model_accuracy_pct.toFixed(1)}%</span> model vs{" "}
          <span className="text-muted">{canonical.always_buy_baseline_pct.toFixed(1)}%</span> Always-Buy baseline
          (n={canonical.n})
        </p>
      ) : null}
    </Link>
  );
}
