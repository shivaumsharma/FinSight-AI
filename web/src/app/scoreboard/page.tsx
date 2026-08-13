"use client";

import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import type { ScoreboardResponse, ScoreboardWindow } from "@/lib/types";

const WINDOWS: { key: "7" | "30" | "90"; label: string }[] = [
  { key: "7", label: "7 DAYS" },
  { key: "30", label: "30 DAYS" },
  { key: "90", label: "90 DAYS" },
];

export default function ScoreboardPage() {
  return (
    <AuthGate>
      {() => <ScoreboardContent />}
    </AuthGate>
  );
}

function ScoreboardContent() {
  const [data, setData] = useState<ScoreboardResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/scoreboard")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, []);

  return (
    <div className="min-h-screen bg-bg pb-20">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <h1 className="font-mono text-lg font-bold text-text">Scoreboard</h1>
        <p className="mt-1.5 text-xs text-muted">
          Does FinSight&apos;s Buy/Hold/Sell call actually beat doing nothing? Every real call this app has made is
          tracked and checked back on -- shown here honestly, whether the model is ahead or behind.
        </p>

        {error && <p className="mt-6 font-mono text-xs text-danger">Couldn&apos;t load the scoreboard. Try again.</p>}

        {!error && !data && (
          <div className="mt-6 flex flex-col gap-4">
            {WINDOWS.map((w) => (
              <div key={w.key} className="h-[180px] animate-pulse rounded-lg border border-border bg-card/60" />
            ))}
          </div>
        )}

        {data && (
          <div className="mt-6 flex flex-col gap-5">
            {WINDOWS.map((w) => (
              <WindowCard key={w.key} label={w.label} window={data.windows[w.key]} />
            ))}
          </div>
        )}
      </div>
      <BottomNav />
    </div>
  );
}

function fmtPct(v: number | null): string {
  return v === null ? "N/A" : `${v.toFixed(1)}%`;
}

// Visually blunt on purpose: the model's own number is red whenever it
// trails the better of the two naive baselines, accent (green) only
// when it's genuinely ahead of both -- never reframed or softened.
function modelColor(window: ScoreboardWindow): string {
  const { model_accuracy_pct, always_buy_accuracy_pct, always_hold_accuracy_pct } = window;
  if (model_accuracy_pct === null) return "text-dim";
  const bestBaseline = Math.max(always_buy_accuracy_pct ?? -Infinity, always_hold_accuracy_pct ?? -Infinity);
  if (bestBaseline === -Infinity) return "text-text";
  if (model_accuracy_pct > bestBaseline) return "text-accent";
  if (model_accuracy_pct < bestBaseline) return "text-danger";
  return "text-warn";
}

function WindowCard({ label, window }: { label: string; window: ScoreboardWindow }) {
  const noData = window.sample_size === 0;

  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">{label}</p>
        <p className="font-mono text-[10px] text-dim">
          n={window.sample_size}
          {window.pending_count > 0 && `, ${window.pending_count} pending`}
        </p>
      </div>

      {noData ? (
        <p className="mt-3 font-mono text-xs text-dim">
          Not enough data yet{window.pending_count > 0 ? ` -- ${window.pending_count} call(s) still tracking toward this checkpoint.` : "."}
        </p>
      ) : (
        <>
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            <StatTile label="MODEL" value={fmtPct(window.model_accuracy_pct)} colorClass={modelColor(window)} />
            <StatTile label="ALWAYS BUY" value={fmtPct(window.always_buy_accuracy_pct)} colorClass="text-text" />
            <StatTile label="ALWAYS HOLD" value={fmtPct(window.always_hold_accuracy_pct)} colorClass="text-text" />
          </div>

          <div className="mt-3 border-t border-border-subtle pt-2">
            <p className="font-mono text-[9px] tracking-wide text-dim">BUY / SELL / HOLD PRECISION</p>
            <div className="mt-1.5 flex flex-col gap-1">
              {(["buy", "sell", "hold"] as const).map((key) => {
                const b = window.breakdown[key];
                return (
                  <div key={key} className="flex items-center justify-between font-mono text-[11px]">
                    <span className="uppercase text-muted">{key}</span>
                    <span className="text-text">
                      {b.precision_pct === null ? "N/A" : `${b.precision_pct.toFixed(1)}%`}
                      <span className="ml-1.5 text-dim">
                        ({b.correct}/{b.total})
                      </span>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatTile({ label, value, colorClass }: { label: string; value: string; colorClass: string }) {
  return (
    <div>
      <p className={`font-mono text-lg font-bold ${colorClass}`}>{value}</p>
      <p className="font-mono text-[9px] tracking-wide text-dim">{label}</p>
    </div>
  );
}
