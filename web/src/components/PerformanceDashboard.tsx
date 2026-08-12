"use client";

import { useEffect, useState } from "react";

interface StockInsights {
  rating: string;
  overall_score: number | null;
  category_scores: {
    valuation: number | null;
    financial_health: number | null;
    growth: number | null;
    profitability: number | null;
    momentum: number | null;
    risk: number | null;
  };
}

const CATEGORY_LABELS: { key: keyof StockInsights["category_scores"]; label: string }[] = [
  { key: "valuation", label: "Valuation" },
  { key: "financial_health", label: "Financial Health" },
  { key: "growth", label: "Growth" },
  { key: "profitability", label: "Profitability" },
  { key: "momentum", label: "Momentum" },
  { key: "risk", label: "Risk (higher = safer)" },
];

function StarRow({ value }: { value: number | null }) {
  if (value === null) return <span className="font-mono text-xs text-dim">N/A</span>;
  return (
    <span className="font-mono text-sm tracking-wide text-accent">
      {"★".repeat(value)}
      <span className="text-dim">{"★".repeat(5 - value)}</span>
    </span>
  );
}

function scoreColor(score: number | null): string {
  if (score === null) return "text-dim";
  if (score >= 70) return "text-accent";
  if (score >= 40) return "text-warn";
  return "text-danger";
}

// Star-rating summary card, sits near the header on the Overview tab --
// same self-fetching convention as every other stock-detail card.
// Deliberately excludes an "Ownership" category (no shareholding-data
// source exists anywhere in this app) rather than showing a fake one.
export default function PerformanceDashboard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<StockInsights | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setData(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/insights`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, [ticker]);

  if (error) return null;

  if (!data) {
    return (
      <div className="mt-4">
        <div className="h-[220px] animate-pulse rounded-lg border border-border bg-card/60" />
      </div>
    );
  }

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">PERFORMANCE DASHBOARD</p>
        {data.overall_score !== null && (
          <span className={`font-mono text-sm font-bold ${scoreColor(data.overall_score)}`}>{data.overall_score}/100</span>
        )}
      </div>
      <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-3">
        <div className="flex flex-col gap-2">
          {CATEGORY_LABELS.map((c) => (
            <div key={c.key} className="flex items-center justify-between">
              <span className="font-mono text-xs text-muted">{c.label}</span>
              <StarRow value={data.category_scores[c.key]} />
            </div>
          ))}
        </div>
        <p className="mt-3 border-t border-border-subtle pt-2 font-mono text-[9px] text-dim">
          Algorithmically derived from real valuation/financial/momentum signals -- not a substitute for the Full
          Research Report.
        </p>
      </div>
    </div>
  );
}
