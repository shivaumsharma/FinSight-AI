"use client";

import { useEffect, useState } from "react";
import type { BacktestAccuracySummary } from "@/lib/types";

// Mandatory badge on every recommendation card -- this app's own
// measured Buy/Hold/Sell accuracy against real historical outcomes,
// not a per-ticker or per-report number (see backtest_stats.py's
// docstring: most researched tickers aren't in the 79-ticker curated
// backtest universe, so there's no honest way to compute one just for
// THIS ticker). The title attribute surfaces the older, lower-scoring
// cohort too -- shown, not hidden.
export default function BacktestBadge() {
  const [data, setData] = useState<BacktestAccuracySummary | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetch("/api/research/backtest-accuracy")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  if (failed || data === null) return null;

  const tooltip = data.secondary
    ? `${data.correct}/${data.scored} correct, ${data.window_label}. Older cohort (${data.secondary.window_label}): ${data.secondary.accuracy_pct}% (${data.secondary.correct}/${data.secondary.scored}).`
    : `${data.correct}/${data.scored} correct, ${data.window_label}.`;

  return (
    <span
      title={tooltip}
      className="inline-flex items-center gap-1 rounded-full border border-border-subtle bg-bg px-2 py-0.5 font-mono text-[9.5px] font-bold text-muted"
    >
      BACKTESTED {data.accuracy_pct}% ACCURATE
      <span className="text-dim">ⓘ</span>
    </span>
  );
}
