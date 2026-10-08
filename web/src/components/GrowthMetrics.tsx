"use client";

import { useEffect, useState } from "react";
import { formatPercent } from "@/lib/numberFormat";
import { Table } from "./ui";

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

const fmtCagr = (v: number | null | undefined): string => formatPercent(v, { decimals: 1, signed: true });

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
          <Table
            caption="Compound annual growth rates"
            columns={["", ...BUCKETS]}
            rows={ROWS.map((row) => ({
              key: row.key,
              label: row.label,
              cells: BUCKETS.map((b) => {
                const v = growth[row.key]?.[b];
                return { value: fmtCagr(v), tone: v === null || v === undefined ? "muted" : v >= 0 ? "gain" : "loss" } as const;
              }),
            }))}
          />
        )}
      </div>
    </div>
  );
}
