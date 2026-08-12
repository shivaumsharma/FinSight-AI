"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SectionSkeleton from "./SectionSkeleton";
import { currencySymbol } from "@/lib/currency";
import type { MarketMoversData, MoverItem } from "@/lib/types";

function BookmarkIcon({ filled }: { filled: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.8}>
      <path d="M6 3.5h12a.5.5 0 0 1 .5.5v17l-6.5-4-6.5 4V4a.5.5 0 0 1 .5-.5z" strokeLinejoin="round" />
    </svg>
  );
}

function AddToWatchlistButton({ ticker }: { ticker: string }) {
  const [state, setState] = useState<"idle" | "adding" | "added" | "error">("idle");

  async function handleAdd() {
    setState("adding");
    try {
      const resp = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      if (!resp.ok) throw new Error("watchlist add failed");
      setState("added");
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 2000);
    }
  }

  return (
    <button
      type="button"
      onClick={handleAdd}
      disabled={state === "adding" || state === "added"}
      title={state === "added" ? "On watchlist" : "Add to watchlist"}
      className={`shrink-0 ${state === "added" ? "text-accent" : "text-dim hover:text-accent"} ${state === "error" ? "text-danger" : ""}`}
    >
      <BookmarkIcon filled={state === "added"} />
    </button>
  );
}

function MoverRow({ item }: { item: MoverItem }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5">
      <Link href={`/stock/${item.ticker}`} className="flex min-w-0 flex-1 items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="font-mono text-sm font-bold text-text hover:text-accent">{item.ticker}</div>
          <div className="truncate font-mono text-[10px] text-dim">{item.name}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-sm text-text">
            {currencySymbol(item.currency)}
            {item.price.toFixed(2)}
          </div>
          <div className={`font-mono text-[10px] font-bold ${item.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
            {item.change_pct >= 0 ? "+" : ""}
            {item.change_pct.toFixed(2)}%
          </div>
        </div>
      </Link>
      <div className="ml-3">
        <AddToWatchlistButton ticker={item.ticker} />
      </div>
    </div>
  );
}

// "Top Movers (Tracked Universe)" -- deliberately NOT a full-market
// scan (see app/reasoning/market_movers.py's docstring): the curated
// backtest universe plus every user's watchlist tickers, unioned
// globally. The label below must stay explicit about that scope.
export default function MarketMovers() {
  const [data, setData] = useState<MarketMoversData | null>(null);
  const [tab, setTab] = useState<"gainers" | "losers">("gainers");

  useEffect(() => {
    fetch("/api/market/movers?limit=5")
      .then((r) => (r.ok ? r.json() : { gainers: [], losers: [] }))
      .then(setData)
      .catch(() => setData({ gainers: [], losers: [] }));
  }, []);

  if (data === null) return <SectionSkeleton label="TOP MOVERS (TRACKED UNIVERSE)" rows={3} />;

  const rows = tab === "gainers" ? data.gainers : data.losers;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">TOP MOVERS (TRACKED UNIVERSE)</p>
        <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
          {(["gainers", "losers"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded px-2.5 py-1 font-mono text-[10px] font-bold ${
                tab === t ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {t === "gainers" ? "GAINERS" : "LOSERS"}
            </button>
          ))}
        </div>
      </div>

      {rows.length > 0 ? (
        <div className="mt-2 flex flex-col gap-2">
          {rows.map((item) => (
            <MoverRow key={item.ticker} item={item} />
          ))}
        </div>
      ) : (
        <p className="mt-2 font-mono text-[11px] text-dim">No movers to show right now.</p>
      )}
    </div>
  );
}
