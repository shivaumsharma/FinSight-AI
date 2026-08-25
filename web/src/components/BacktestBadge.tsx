"use client";

import type { TrackRecord } from "@/lib/types";

// Mandatory badge on every recommendation card -- this app's own
// measured Buy/Hold/Sell accuracy against real historical outcomes.
//
// Deliberately reads the SAME `trackRecord` prop TrackRecordBlock below
// uses (report_data_builder.py's canonical_accuracy.py-backed number),
// not its own separate fetch to /v1/research/backtest-accuracy. It used
// to: that endpoint computes a DIFFERENT statistic (the curated
// 79-ticker cohort, e.g. "48.0%") from what TrackRecordBlock shows
// (the canonical broad-universe number, e.g. "36.4%") -- two real,
// honestly-computed numbers, but displayed as if they were the same
// claim on the same report, which just reads as a contradiction to
// anyone comparing them. canonical_accuracy.py's own docstring is
// explicit that it's meant to be "FinSight's ONE reported accuracy
// number, not a table of sub-metrics" -- this badge now honors that by
// showing that one number everywhere, instead of a second, older one.
export default function BacktestBadge({ trackRecord }: { trackRecord: TrackRecord | null | undefined }) {
  if (!trackRecord) return null;

  const tooltip = `${trackRecord.summary_line}. ${trackRecord.methodology}`;

  return (
    <span
      title={tooltip}
      className="inline-flex items-center gap-1 rounded-full border border-border-subtle bg-bg px-2 py-0.5 font-mono text-[9.5px] font-bold text-muted"
    >
      BACKTESTED {trackRecord.model_accuracy_pct}% ACCURATE
      <span className="text-dim">ⓘ</span>
    </span>
  );
}
