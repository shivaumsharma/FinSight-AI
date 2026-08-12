"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { currencySymbol } from "@/lib/currency";

interface SimilarStock {
  ticker: string;
  name: string;
  price: number;
  change_pct: number | null;
  currency: string;
}

// Explicitly labeled "From your tracked universe" -- this app has no
// peer-multiple/industry data source (see relative_valuation.py's own
// docstring) and no comprehensive sector index, so this is a real
// same-sector lookup scoped to the same curated+watchlist universe Top
// Movers already uses, never a market-wide scan.
export default function SimilarStocks({ ticker }: { ticker: string }) {
  const [similar, setSimilar] = useState<SimilarStock[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setSimilar(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/similar`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => setSimilar(data.similar))
      .catch(() => setError(true));
  }, [ticker]);

  if (error) return null;
  if (!similar) {
    return (
      <div className="mt-4">
        <div className="h-[80px] animate-pulse rounded-lg border border-border bg-card/60" />
      </div>
    );
  }
  if (similar.length === 0) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">SIMILAR STOCKS (FROM YOUR TRACKED UNIVERSE)</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {similar.map((s) => (
          <Link
            key={s.ticker}
            href={`/stock/${s.ticker}`}
            className="flex w-[130px] shrink-0 flex-col gap-1 rounded-lg border border-border bg-card px-3 py-2.5 hover:border-accent"
          >
            <span className="truncate font-mono text-xs font-bold text-text">{s.ticker}</span>
            <span className="truncate font-mono text-[9px] text-dim">{s.name}</span>
            <span className="font-mono text-xs text-text">
              {currencySymbol(s.currency)}
              {s.price.toFixed(2)}
            </span>
            {s.change_pct !== null && (
              <span className={`font-mono text-[10px] font-bold ${s.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                {s.change_pct >= 0 ? "+" : ""}
                {s.change_pct.toFixed(2)}%
              </span>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}
