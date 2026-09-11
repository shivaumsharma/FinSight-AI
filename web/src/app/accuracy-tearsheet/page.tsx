"use client";

import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import type { AccuracyTearsheetResponse } from "@/lib/types";

export default function AccuracyTearsheetPage() {
  return (
    <AuthGate>
      {() => <AccuracyTearsheetContent />}
    </AuthGate>
  );
}

function AccuracyTearsheetContent() {
  const [data, setData] = useState<AccuracyTearsheetResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/accuracy-tearsheet")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, []);

  return (
    <div className="min-h-screen bg-bg pb-safe-20">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <h1 className="font-mono text-lg font-bold text-text">Accuracy Tearsheet</h1>
        <p className="mt-1.5 text-xs text-muted">
          FinSight&apos;s one canonical backtested-accuracy number -- every Buy/Hold/Sell call the real production
          decision path made on the broad, non-cherry-picked S&amp;P 500+400+600(partial) universe, scored 12 months
          later. Different from the Scoreboard: that tracks recent live calls; this is the full historical backtest.
        </p>

        {error && (
          <p className="mt-6 font-mono text-xs text-danger">Couldn&apos;t load the accuracy tearsheet. Try again.</p>
        )}

        {!error && !data && (
          <div className="mt-6 flex flex-col gap-4">
            <div className="h-[140px] animate-pulse rounded-lg border border-border bg-card/60" />
          </div>
        )}

        {data && !data.available && (
          <p className="mt-6 font-mono text-xs text-dim">
            The canonical backtest hasn&apos;t been run for this deployment yet.
          </p>
        )}

        {data?.canonical && (
          <div className="mt-6 flex flex-col gap-5">
            <HeadlineCard canonical={data.canonical} />
            {data.live_breakdowns && <LiveBreakdowns breakdowns={data.live_breakdowns} />}
            <MethodologyCard canonical={data.canonical} />
          </div>
        )}
      </div>
      <BottomNav />
    </div>
  );
}

function HeadlineCard({ canonical }: { canonical: NonNullable<AccuracyTearsheetResponse["canonical"]> }) {
  const modelColor = canonical.beats_baseline ? "text-accent" : "text-danger";
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3">
      <p className="font-mono text-[10px] tracking-wide text-dim">{canonical.metric.toUpperCase()}</p>
      <div className="mt-3 grid grid-cols-2 gap-2 text-center">
        <div>
          <p className={`font-mono text-2xl font-bold ${modelColor}`}>{canonical.model_accuracy_pct.toFixed(1)}%</p>
          <p className="font-mono text-[9px] tracking-wide text-dim">
            MODEL (95% CI {canonical.model_ci_95[0].toFixed(1)}-{canonical.model_ci_95[1].toFixed(1)}%)
          </p>
        </div>
        <div>
          <p className="font-mono text-2xl font-bold text-muted">{canonical.always_buy_baseline_pct.toFixed(1)}%</p>
          <p className="font-mono text-[9px] tracking-wide text-dim">
            ALWAYS-BUY BASELINE (95% CI {canonical.always_buy_ci_95[0].toFixed(1)}-
            {canonical.always_buy_ci_95[1].toFixed(1)}%)
          </p>
        </div>
      </div>
      <p className="mt-3 border-t border-border-subtle pt-2 font-mono text-[10px] text-dim">
        n={canonical.n} &middot; model {canonical.beats_baseline ? "beats" : "loses to"} the naive baseline
      </p>
    </div>
  );
}

function MethodologyCard({ canonical }: { canonical: NonNullable<AccuracyTearsheetResponse["canonical"]> }) {
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3">
      <p className="font-mono text-[10px] tracking-wide text-dim">METHODOLOGY</p>
      <p className="mt-1.5 text-xs text-muted">{canonical.methodology}</p>
      <div className="mt-3 border-t border-border-subtle pt-2">
        <p className="font-mono text-[9px] tracking-wide text-dim">SOURCES</p>
        <div className="mt-1.5 flex flex-col gap-1">
          {canonical.sources.map((s) => (
            <div key={s.file} className="font-mono text-[11px] text-muted">
              {s.file}: n={s.n_scored}
              {s.as_of_range && ` (${s.as_of_range[0]} to ${s.as_of_range[1]})`}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const BREAKDOWN_LABELS: Record<string, string> = {
  precision_by_sector: "Precision by sector/universe",
  buy_vs_sell_precision: "Buy vs. Sell precision",
  accuracy_trend_by_run_date: "Accuracy trend by run date",
};

function LiveBreakdowns({ breakdowns }: { breakdowns: NonNullable<AccuracyTearsheetResponse["live_breakdowns"]> }) {
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3">
      <p className="font-mono text-[10px] tracking-wide text-dim">LIVE BREAKDOWNS (SNOWFLAKE)</p>
      <div className="mt-2 flex flex-col gap-4">
        {Object.entries(breakdowns).map(([name, rows]) => (
          <div key={name}>
            <p className="font-mono text-[9px] tracking-wide text-dim">
              {(BREAKDOWN_LABELS[name] ?? name).toUpperCase()}
            </p>
            <div className="mt-1.5 flex flex-col gap-1">
              {rows.map((row, i) => (
                <div key={i} className="flex items-center justify-between font-mono text-[11px] text-muted">
                  <span>{String(row[0])}</span>
                  <span className="text-text">{row.slice(1).map((v) => String(v)).join(" / ")}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
