"use client";

import { useEffect, useState } from "react";

interface GrowthData {
  revenue_cagr: Record<string, number | null>;
  eps_cagr: Record<string, number | null>;
  book_value_cagr: Record<string, number | null>;
  fcf_cagr: Record<string, number | null>;
}

const ROWS: { key: keyof GrowthData; label: string }[] = [
  { key: "revenue_cagr", label: "Revenue CAGR" },
  { key: "eps_cagr", label: "EPS CAGR" },
  { key: "book_value_cagr", label: "Book Value CAGR" },
  { key: "fcf_cagr", label: "FCF CAGR" },
];

const BUCKETS = ["1y", "3y", "5y"] as const;

function fmtCagr(v: number | null | undefined): string {
  if (v === null || v === undefined) return "N/A";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

// Independent fetch (own /financials call, same endpoint
// FinancialPerformanceChart uses at its default "yearly" period) --
// growth is always computed from annual statements server-side
// regardless of period, so this never needs to react to a
// quarterly/yearly toggle. Matches this page's other cards' own
// self-fetching convention rather than threading data down from a
// shared parent fetch.
export default function GrowthMetrics({ ticker }: { ticker: string }) {
  const [growth, setGrowth] = useState<GrowthData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setGrowth(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/financials`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => setGrowth(data.growth))
      .catch(() => setError(true));
  }, [ticker]);

  if (error) return null;

  const hasAnyValue =
    growth && Object.values(growth).some((bucket) => Object.values(bucket).some((v) => v !== null));

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">GROWTH</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        {!growth ? (
          <div className="h-[90px] animate-pulse rounded bg-card/60" />
        ) : !hasAnyValue ? (
          <p className="py-2 text-center font-mono text-[11px] text-dim">Not enough annual history to compute growth rates.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[320px] font-mono text-xs">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-dim">
                  <th className="pb-2 text-left font-normal"></th>
                  {BUCKETS.map((b) => (
                    <th key={b} className="pb-2 text-right font-normal">
                      {b}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ROWS.map((row) => {
                  const bucket = growth[row.key];
                  return (
                    <tr key={row.key} className="border-t border-border-subtle">
                      <td className="py-1.5 text-dim">{row.label}</td>
                      {BUCKETS.map((b) => {
                        const v = bucket?.[b];
                        return (
                          <td
                            key={b}
                            className={`py-1.5 text-right ${
                              v === null || v === undefined ? "text-dim" : v >= 0 ? "text-accent" : "text-danger"
                            }`}
                          >
                            {fmtCagr(v)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
