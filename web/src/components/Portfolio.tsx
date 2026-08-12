"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import RatingBadge from "./RatingBadge";
import SectionSkeleton from "./SectionSkeleton";
import { currencySymbol } from "@/lib/currency";
import { PORTFOLIO_UPDATED_EVENT, notifyPortfolioUpdated } from "@/lib/portfolioEvents";
import type { CompanySuggestion, PortfolioAnalysis, PortfolioHolding, PortfolioSummary } from "@/lib/types";

function fmt(n: number): string {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
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
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null);
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("");
  const [avgCost, setAvgCost] = useState("");
  const [buyDate, setBuyDate] = useState(todayIso());
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [seeding, setSeeding] = useState(false);

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
    // Combined portfolio analysis (rating rollup across already-
    // researched holdings) -- a separate, independent fetch from the
    // holdings list above, never blocking it if this one fails.
    fetch("/api/portfolio/analysis")
      .then((r) => (r.ok ? r.json() : null))
      .then(setAnalysis)
      .catch(() => setAnalysis(null));
  }

  useEffect(refresh, []);

  // A simulated order placed in OrderTicket (a sibling component on
  // this same page) updates the backend's holdings correctly, but
  // this component has no other way to know its own already-fetched
  // data just went stale -- see lib/portfolioEvents.ts's own comment.
  useEffect(() => {
    window.addEventListener(PORTFOLIO_UPDATED_EVENT, refresh);
    return () => window.removeEventListener(PORTFOLIO_UPDATED_EVENT, refresh);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Same debounced autocomplete as Watchlist's add-ticker input --
  // reuses the same /api/companies/suggest endpoint, not a second
  // ticker-resolution path.
  useEffect(() => {
    const q = ticker.trim();
    if (q.length < 2) {
      setSuggestions([]);
      return;
    }
    const timeout = setTimeout(() => {
      fetch(`/api/companies/suggest?q=${encodeURIComponent(q)}`)
        .then((r) => (r.ok ? r.json() : { suggestions: [] }))
        .then((data) => setSuggestions(data.suggestions))
        .catch(() => setSuggestions([]));
    }, 250);
    return () => clearTimeout(timeout);
  }, [ticker]);

  async function submitHolding(rawTicker: string) {
    const q = parseFloat(quantity);
    const c = parseFloat(avgCost);
    if (!rawTicker.trim() || !(q > 0) || !(c > 0) || adding) return;
    setAdding(true);
    setError(null);
    try {
      const resp = await fetch("/api/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: rawTicker.trim(), quantity: q, avg_cost: c, buy_date: buyDate || null }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ message: "Couldn't add that holding." }));
        setError(body.message || "Couldn't add that holding.");
        return;
      }
      setTicker("");
      setQuantity("");
      setAvgCost("");
      setBuyDate(todayIso());
      setShowForm(false);
      setSuggestions([]);
      refresh();
    } finally {
      setAdding(false);
    }
  }

  function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    submitHolding(ticker);
  }

  function selectSuggestion(s: CompanySuggestion) {
    setShowSuggestions(false);
    setTicker(s.ticker);
  }

  async function handleRemove(t: string) {
    setHoldings((prev) => (prev ? prev.filter((h) => h.ticker !== t) : prev));
    await fetch(`/api/portfolio/${t}`, { method: "DELETE" }).catch(() => {});
    refresh();
  }

  // Empty-state helper for a brand new account -- seeds a few
  // realistic watchlist tickers and simulated BUY orders using the
  // exact same endpoints the manual add-ticker/OrderTicket forms call,
  // so a first-time user immediately sees a populated dashboard rather
  // than a wall of "no holdings yet" placeholders. Reversible: every
  // seeded row uses the same remove/× controls as anything added by
  // hand.
  async function seedSampleData() {
    if (seeding) return;
    setSeeding(true);
    try {
      await Promise.allSettled([
        fetch("/api/watchlist", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: "TSLA" }),
        }),
        fetch("/api/watchlist", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: "RELIANCE.NS" }),
        }),
        fetch("/api/orders", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: "AAPL", side: "BUY", quantity: 5 }),
        }),
        fetch("/api/orders", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: "MSFT", side: "BUY", quantity: 3 }),
        }),
        fetch("/api/orders", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: "NVDA", side: "BUY", quantity: 2 }),
        }),
      ]);
      refresh();
      notifyPortfolioUpdated();
    } finally {
      setSeeding(false);
    }
  }

  if (holdings === null) return <SectionSkeleton label="PORTFOLIO" rows={2} />;

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
            <div>
              <div className="font-mono text-[10px] text-dim">
                PORTFOLIO VALUE{summary.mixed_currency && " (USD EQUIV.)"}
              </div>
              <span className="font-mono text-lg font-bold text-text">
                {currencySymbol(summary.currency)}
                {fmt(summary.total_market_value)}
              </span>
            </div>
            {summary.total_unrealized_pnl !== null && (
              <div className="text-right">
                <div className="font-mono text-[10px] text-dim">OVERALL P&amp;L</div>
                <span className={`font-mono text-xs font-bold ${summary.total_unrealized_pnl >= 0 ? "text-accent" : "text-danger"}`}>
                  {summary.total_unrealized_pnl >= 0 ? "+" : ""}
                  {currencySymbol(summary.currency)}
                  {fmt(summary.total_unrealized_pnl)}
                  {summary.total_unrealized_pnl_pct !== null && ` (${summary.total_unrealized_pnl_pct.toFixed(2)}%)`}
                </span>
              </div>
            )}
          </div>
          <div className="mt-2 flex items-baseline justify-between border-t border-border-subtle pt-2">
            {summary.total_cost_basis !== null && (
              <div>
                <div className="font-mono text-[10px] text-dim">INVESTED</div>
                <span className="font-mono text-xs text-muted">
                  {currencySymbol(summary.currency)}
                  {fmt(summary.total_cost_basis)}
                </span>
              </div>
            )}
            {summary.total_today_pnl !== null && (
              <div className="text-right">
                <div className="font-mono text-[10px] text-dim">TODAY&apos;S P&amp;L</div>
                <span className={`font-mono text-xs font-bold ${summary.total_today_pnl >= 0 ? "text-accent" : "text-danger"}`}>
                  {summary.total_today_pnl >= 0 ? "+" : ""}
                  {currencySymbol(summary.currency)}
                  {fmt(summary.total_today_pnl)}
                  {summary.total_today_pnl_pct !== null && ` (${summary.total_today_pnl_pct.toFixed(2)}%)`}
                </span>
              </div>
            )}
          </div>
          <p className="mt-2 font-mono text-[10px] text-dim">
            Self-reported holdings, not connected to a brokerage -- unrealized P&amp;L only, no orders placed here.
            {summary.mixed_currency && " Totals above are converted to USD at the live exchange rate."}
            {summary.excluded_from_summary && " A holding in an unsupported currency is excluded from the totals."}
          </p>
        </div>
      )}

      {holdings.length > 0 && analysis && analysis.holdings_researched > 0 && (
        <div className="mt-2 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-2">
          <div className="flex items-center justify-between font-mono text-[10px]">
            <span className="text-dim">
              {analysis.holdings_researched}/{analysis.holdings_total} HOLDINGS RESEARCHED
            </span>
            {analysis.value_weighted_pct && (
              <span className="font-bold">
                <span className="text-accent">{analysis.value_weighted_pct.Buy.toFixed(0)}% Buy</span>
                {" · "}
                <span className="text-danger">{analysis.value_weighted_pct.Sell.toFixed(0)}% Sell</span>
                <span className="text-dim"> (by value)</span>
              </span>
            )}
          </div>
        </div>
      )}

      {holdings.length > 0 && (
        <div className="mt-2 flex flex-col gap-2">
          {holdings.map((h) => (
            <div
              key={h.ticker}
              className={`flex items-center justify-between rounded-lg border bg-card px-3.5 py-2.5 ${
                h.rating === "Sell" ? "border-danger" : "border-border"
              }`}
            >
              <div>
                <div className="flex items-center gap-2">
                  <Link href={`/stock/${h.ticker}`} className="font-mono text-sm font-bold text-text hover:text-accent">
                    {h.ticker}
                  </Link>
                  {h.rating ? (
                    <RatingBadge rating={h.rating} size="sm" />
                  ) : (
                    <Link
                      href={`/?q=${encodeURIComponent(`Should I invest in ${h.ticker}?`)}`}
                      className="font-mono text-[10px] font-bold text-muted hover:text-accent"
                    >
                      RESEARCH &rarr;
                    </Link>
                  )}
                </div>
                <div className="font-mono text-[10px] text-dim">
                  {h.quantity} sh @ {currencySymbol(h.currency)}
                  {fmt(h.avg_cost)}
                  {h.buy_date && ` · bought ${h.buy_date}`}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {h.market_value !== null && h.unrealized_pnl !== null ? (
                  <div className="text-right">
                    <div className="font-mono text-sm text-text">
                      {currencySymbol(h.currency)}
                      {fmt(h.market_value)}
                    </div>
                    <div className={`font-mono text-[10px] ${h.unrealized_pnl >= 0 ? "text-accent" : "text-danger"}`}>
                      {h.unrealized_pnl >= 0 ? "+" : ""}
                      {currencySymbol(h.currency)}
                      {fmt(h.unrealized_pnl)}
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
        <div className="mt-2 flex items-center justify-between gap-3 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-2.5">
          <p className="font-mono text-[11px] text-dim">
            No holdings yet -- add one, or try it with sample data.
          </p>
          <button
            type="button"
            onClick={seedSampleData}
            disabled={seeding}
            className="shrink-0 font-mono text-[10px] font-bold text-muted hover:text-accent disabled:opacity-50"
          >
            {seeding ? "LOADING..." : "TRY SAMPLE DATA"}
          </button>
        </div>
      )}

      {showForm && (
        <form onSubmit={handleAdd} className="relative mt-2 flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => {
              setTicker(e.target.value);
              setShowSuggestions(true);
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setShowSuggestions(false)}
            onKeyDown={(e) => {
              if (e.key === "Escape") setShowSuggestions(false);
            }}
            placeholder="ticker or company name..."
            disabled={adding}
            autoComplete="off"
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />

          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute left-3 right-3 top-[42px] z-10 overflow-hidden rounded-lg border border-border bg-card shadow-lg">
              {suggestions.map((s) => (
                <button
                  key={s.ticker}
                  type="button"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    selectSuggestion(s);
                  }}
                  className="flex w-full items-center justify-between border-b border-border-subtle px-3 py-2 text-left last:border-b-0 hover:bg-bg/60"
                >
                  <span className="truncate font-mono text-xs text-text">{s.name}</span>
                  <span className="ml-2 flex-shrink-0 font-mono text-[10px] text-dim">{s.ticker}</span>
                </button>
              ))}
            </div>
          )}

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
          <label className="flex items-center gap-2 font-mono text-[10px] text-dim">
            Buy date
            <input
              type="date"
              value={buyDate}
              onChange={(e) => setBuyDate(e.target.value)}
              disabled={adding}
              className="flex-1 rounded-lg border border-border bg-bg px-3 py-1.5 font-mono text-xs text-text focus:outline-none focus:border-accent disabled:opacity-60"
            />
          </label>
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
