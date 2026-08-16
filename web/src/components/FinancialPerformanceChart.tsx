"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactNumber } from "@/lib/stockFormat";

interface FinancialPeriodRow {
  period_end: string;
  revenue: number | null;
  net_income: number | null;
  ebit: number | null;
  net_margin_pct: number | null;
  operating_margin_pct: number | null;
}

interface FinancialsResponse {
  period: "yearly" | "quarterly";
  series: FinancialPeriodRow[];
  growth: {
    revenue_cagr: Record<string, number | null>;
    eps_cagr: Record<string, number | null>;
    book_value_cagr: Record<string, number | null>;
    fcf_cagr: Record<string, number | null>;
  };
}

// Colors read from this app's own CSS variables (set globally in
// globals.css) rather than hardcoded hex, matching the rest of the
// app's theme-driven styling -- recharts needs a resolved string, not
// a CSS var reference, so these are read from computed styles once on
// mount.
function useThemeColors() {
  const [colors, setColors] = useState({ accent: "#c9a227", danger: "#e05252", dim: "#7a7a7a" });
  useEffect(() => {
    const style = getComputedStyle(document.documentElement);
    setColors({
      accent: style.getPropertyValue("--accent").trim() || "#c9a227",
      danger: style.getPropertyValue("--danger").trim() || "#e05252",
      dim: style.getPropertyValue("--dim").trim() || "#7a7a7a",
    });
  }, []);
  return colors;
}

function ChartTooltip({
  active,
  payload,
  label,
  symbol,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string;
  symbol: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 font-mono text-[10px] shadow-lg">
      <div className="mb-1 font-bold text-text">{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {p.name.includes("Margin") ? `${p.value.toFixed(2)}%` : `${symbol}${fmtCompactNumber(p.value)}`}
        </div>
      ))}
    </div>
  );
}

// Fetches its own data (rather than receiving it as a prop) since this
// is the one card on the stock detail page whose data depends on a
// second axis (quarterly vs yearly) the user toggles in place -- a
// parent-fetched prop would mean re-fetching from the parent on every
// toggle anyway.
export default function FinancialPerformanceChart({ ticker, currency }: { ticker: string; currency: string }) {
  const [period, setPeriod] = useState<"yearly" | "quarterly">("yearly");
  const [data, setData] = useState<FinancialsResponse | null>(null);
  const [error, setError] = useState(false);
  const colors = useThemeColors();
  const symbol = currencySymbol(currency);

  useEffect(() => {
    setData(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/financials?period=${period}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, [ticker, period]);

  const rows = data?.series ?? [];
  const hasData = rows.some((r) => r.revenue !== null || r.net_income !== null);

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">FINANCIAL PERFORMANCE</p>
        <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
          {(["yearly", "quarterly"] as const).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPeriod(p)}
              className={`rounded px-2.5 py-1 font-mono text-[10px] font-bold ${
                period === p ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {p === "yearly" ? "YEARLY" : "QUARTERLY"}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-3">
        {error ? (
          <p className="py-6 text-center font-mono text-[11px] text-dim">Couldn&apos;t load financial statement data for this ticker.</p>
        ) : !data ? (
          <div className="h-[220px] animate-pulse rounded bg-card/60" />
        ) : !hasData ? (
          <p className="py-6 text-center font-mono text-[11px] text-dim">No {period} statement history available.</p>
        ) : (
          <div className="h-[240px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis dataKey="period_end" tick={{ fontSize: 9, fill: colors.dim }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
                <YAxis
                  yAxisId="amount"
                  tick={{ fontSize: 9, fill: colors.dim }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => fmtCompactNumber(v)}
                  width={40}
                />
                <YAxis yAxisId="margin" orientation="right" tick={{ fontSize: 9, fill: colors.dim }} tickLine={false} axisLine={false} width={36} />
                <Tooltip content={<ChartTooltip symbol={symbol} />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />
                <Bar yAxisId="amount" dataKey="revenue" name="Revenue" fill={colors.accent} radius={[2, 2, 0, 0]} />
                <Bar yAxisId="amount" dataKey="net_income" name="Net Income" fill={colors.dim} radius={[2, 2, 0, 0]} />
                <Line yAxisId="margin" type="monotone" dataKey="net_margin_pct" name="Net Margin %" stroke={colors.danger} dot={false} strokeWidth={1.5} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
