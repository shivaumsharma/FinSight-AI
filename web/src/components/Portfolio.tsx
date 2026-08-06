"use client";

import { useEffect, useState } from "react";
import type { PortfolioHolding, PortfolioSummary } from "@/lib/types";

function fmt(n: number): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// A home-dashboard card, not a dedicated nav-linked page like
// Watchlist/Reports/Profile -- see BottomNav.tsx's own comment on
// deliberately staying at 4 tabs. Self-reported only: quantity/avg_cost
// are whatever the user types in here, never synced from a real
// brokerage, and this never places or executes anything -- purely a
// manual P&L calculator against live quotes.
export default function Portfolio() {
  const [holdings, setHoldings] = useState<PortfolioHolding[] | null>(null);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("");
  const [avgCost, setAvgCost] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  function refresh() {
    fetch("/api/portfolio")
      .then((r) => (r.ok ? r.json() : { holdings: [], summary: null }))
      .then((data) => {
        setHoldings(data.holdings);
        setSummary(data.summary);
      })
      .catch(() => {
        setHoldings([]);
        setSummary(null);
      });
  }

  useEffect(refresh, []);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    const q = parseFloat(quantity);
    const c = parseFloat(avgCost);
    if (!ticker.trim() || !(q > 0) || !(c > 0) || adding) return;
    setAdding(true);
    setError(null);
    try {
      const resp = await fetch("/api/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim(), quantity: q, avg_cost: c }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ message: "Couldn't add that holding." }));
        setError(body.message || "Couldn't add that holding.");
        return;
      }
      setTicker("");
      setQuantity("");
      setAvgCost("");
      setShowForm(false);
      refresh();
    } finally {
      setAdding(false);
    }
  }

  async function handleRemove(t: string) {
    setHoldings((prev) => (prev ? prev.filter((h) => h.ticker !== t) : prev));
    await fetch(`/api/portfolio/${t}`, { method: "DELETE" }).catch(() => {});
    refresh();
  }

  if (holdings === null) return null;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">PORTFOLIO</p>
        <button
          type="button"
          onClick={() => setShowForm((s) => !s)}
          className="font-mono text-[10px] font-bold text-muted hover:text-accent"
        >
          {showForm ? "CANCEL" : "+ ADD HOLDING"}
        </button>
      </div>

      {holdings.length > 0 && summary && summary.total_market_value !== null && (
        <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-2.5">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-lg font-bold text-text">${fmt(summary.total_market_value)}</span>
            {summary.total_unrealized_pnl !== null && (
              <span className={`font-mono text-xs font-bold ${summary.total_unrealized_pnl >= 0 ? "text-accent" : "text-danger"}`}>
                {summary.total_unrealized_pnl >= 0 ? "+" : ""}
                ${fmt(summary.total_unrealized_pnl)}
                {summary.total_unrealized_pnl_pct !== null && ` (${summary.total_unrealized_pnl_pct.toFixed(2)}%)`}
              </span>
            )}
          </div>
          <p className="mt-0.5 font-mono text-[10px] text-dim">
            Self-reported holdings, not connected to a brokerage -- unrealized P&amp;L only, no orders placed here.
          </p>
        </div>
      )}

      {holdings.length > 0 && (
        <div className="mt-2 flex flex-col gap-2">
          {holdings.map((h) => (
            <div key={h.ticker} className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5">
              <div>
                <div className="font-mono text-sm font-bold text-text">{h.ticker}</div>
                <div className="font-mono text-[10px] text-dim">
                  {h.quantity} sh @ ${fmt(h.avg_cost)}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {h.market_value !== null && h.unrealized_pnl !== null ? (
                  <div className="text-right">
                    <div className="font-mono text-sm text-text">${fmt(h.market_value)}</div>
                    <div className={`font-mono text-[10px] ${h.unrealized_pnl >= 0 ? "text-accent" : "text-danger"}`}>
                      {h.unrealized_pnl >= 0 ? "+" : ""}
                      ${fmt(h.unrealized_pnl)}
                      {h.unrealized_pnl_pct !== null && ` (${h.unrealized_pnl_pct.toFixed(1)}%)`}
                    </div>
                  </div>
                ) : (
                  <span className="font-mono text-[10px] text-dim">quote unavailable</span>
                )}
                <button
                  type="button"
                  onClick={() => handleRemove(h.ticker)}
                  title="Remove holding"
                  className="font-mono text-xs text-dim hover:text-danger"
                >
                  &times;
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {holdings.length === 0 && !showForm && (
        <p className="mt-2 font-mono text-[11px] text-dim">
          No holdings yet -- add one to track its live value against what you paid.
        </p>
      )}

      {showForm && (
        <form onSubmit={handleAdd} className="mt-2 flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="ticker (e.g. AAPL)"
            disabled={adding}
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <div className="flex gap-2">
            <input
              type="number"
              step="any"
              min="0"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="quantity"
              disabled={adding}
              className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
            />
            <input
              type="number"
              step="any"
              min="0"
              value={avgCost}
              onChange={(e) => setAvgCost(e.target.value)}
              placeholder="avg cost / share"
              disabled={adding}
              className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
            />
          </div>
          <button
            type="submit"
            disabled={adding || !ticker.trim() || !quantity || !avgCost}
            className="rounded-lg bg-accent py-2 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {adding ? "ADDING..." : "ADD HOLDING"}
          </button>
          {error && <p className="font-mono text-[10px] text-danger">{error}</p>}
        </form>
      )}
    </div>
  );
}
