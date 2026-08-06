"use client";

import { useEffect, useState } from "react";
import RatingBadge from "./RatingBadge";
import type { WatchlistItem } from "@/lib/types";

function formatShortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Always renders, even with zero items -- the add-ticker input is the
// primary way a user grows this list, so it needs to stay visible
// rather than disappearing until something's already on it.
export default function Watchlist() {
  const [items, setItems] = useState<WatchlistItem[] | null>(null);
  const [ticker, setTicker] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    fetch("/api/watchlist")
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => setItems(data.items))
      .catch(() => setItems([]));
  }

  useEffect(refresh, []);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!ticker.trim() || adding) return;
    setAdding(true);
    setError(null);
    try {
      const resp = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim() }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ message: "Couldn't add that ticker." }));
        setError(body.message || "Couldn't add that ticker.");
        return;
      }
      setTicker("");
      refresh();
    } finally {
      setAdding(false);
    }
  }

  async function handleRemove(t: string) {
    // Optimistic removal -- the DELETE endpoint is idempotent and
    // near-instant, no need to wait for the round-trip before the
    // card disappears.
    setItems((prev) => (prev ? prev.filter((i) => i.ticker !== t) : prev));
    await fetch(`/api/watchlist/${t}`, { method: "DELETE" }).catch(() => {});
  }

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">WATCHLIST</p>

      {items && items.length > 0 && (
        <div className="mt-2 flex flex-col gap-2">
          {items.map((item) => {
            const corporateActions = [
              item.next_earnings_date && `Earnings ${formatShortDate(item.next_earnings_date)}`,
              item.next_ex_dividend_date && `Ex-div ${formatShortDate(item.next_ex_dividend_date)}`,
            ].filter(Boolean);

            return (
              <div key={item.ticker} className="rounded-lg border border-border bg-card px-3.5 py-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-sm font-bold text-text">{item.ticker}</span>
                    {item.rating ? (
                      <RatingBadge rating={item.rating} size="sm" />
                    ) : (
                      <span className="font-mono text-[10px] text-dim">not yet researched</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {item.price !== null && (
                      <div className="text-right">
                        <div className="font-mono text-sm text-text">{item.price.toFixed(2)}</div>
                        {item.change_pct !== null && (
                          <div className={`font-mono text-[10px] ${item.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                            {item.change_pct >= 0 ? "+" : ""}
                            {item.change_pct.toFixed(2)}%
                          </div>
                        )}
                      </div>
                    )}
                    <button
                      type="button"
                      onClick={() => handleRemove(item.ticker)}
                      title="Remove from watchlist"
                      className="font-mono text-xs text-dim hover:text-danger"
                    >
                      &times;
                    </button>
                  </div>
                </div>
                {corporateActions.length > 0 && (
                  <p className="mt-1.5 font-mono text-[10px] text-dim">{corporateActions.join(" · ")}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      <form onSubmit={handleAdd} className="mt-2 flex gap-2">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="add ticker..."
          disabled={adding}
          className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={adding || !ticker.trim()}
          className="rounded-lg border border-border bg-card px-3.5 py-2 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent disabled:opacity-50"
        >
          {adding ? "..." : "ADD"}
        </button>
      </form>
      {error && <p className="mt-1.5 font-mono text-[10px] text-danger">{error}</p>}
    </div>
  );
}
