"use client";

import { useEffect, useState } from "react";
import RatingBadge from "./RatingBadge";
import { currencySymbol } from "@/lib/currency";
import { fmtPrice } from "@/lib/stockFormat";

interface StockInsights {
  rating: string;
  fair_value_estimate: number | null;
  current_price: number | null;
  upside_percent: number | null;
  consensus: { score: number; label: string; agreeing_count: number; total_count: number } | null;
  flags: string[];
}

// "AI Insights" here means flags algorithmically derived from disclosed
// thresholds on real valuation/financial/momentum signals (see
// app/reasoning/stock_score.py) -- deliberately NOT an LLM-generated
// paragraph, so this card loads instantly and costs nothing per view,
// unlike the Full Research Report's own LLM-written Investment Thesis.
export default function AIInsightsCard({ ticker, currency }: { ticker: string; currency: string }) {
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
        <div className="h-[100px] animate-pulse rounded-lg border border-border bg-card/60" />
      </div>
    );
  }

  const symbol = currencySymbol(currency);

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">AI INSIGHTS</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <div className="flex items-center justify-between">
          <RatingBadge rating={data.rating} />
          {data.fair_value_estimate !== null && (
            <div className="text-right">
              <div className="font-mono text-[10px] text-dim">FAIR VALUE ESTIMATE</div>
              <div className="font-mono text-sm font-bold text-text">{fmtPrice(data.fair_value_estimate, symbol)}</div>
              {data.upside_percent !== null && (
                <div className={`font-mono text-[10px] font-bold ${data.upside_percent >= 0 ? "text-accent" : "text-danger"}`}>
                  {data.upside_percent >= 0 ? "+" : ""}
                  {data.upside_percent.toFixed(1)}% vs current price
                </div>
              )}
            </div>
          )}
        </div>

        {data.flags.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {data.flags.map((flag) => (
              <span key={flag} className="rounded border border-border-subtle px-2 py-0.5 font-mono text-[10px] text-muted">
                {flag}
              </span>
            ))}
          </div>
        )}

        {data.consensus && (
          <p className="mt-3 border-t border-border-subtle pt-2 font-mono text-[10px] text-dim">
            Institutional consensus: {data.consensus.label} ({data.consensus.agreeing_count}/{data.consensus.total_count}{" "}
            analysts agree).
          </p>
        )}
      </div>
    </div>
  );
}
