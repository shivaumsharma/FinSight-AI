"use client";

import { useEffect, useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { notifyPortfolioUpdated } from "@/lib/portfolioEvents";
import type { CompanySuggestion, Order } from "@/lib/types";

type Side = "BUY" | "SELL";

// Simulated paper trading -- NO real broker, NO real credentials, NO
// real money ever changes hands. A BUY/SELL fills instantly at the
// live quote price and updates the SAME self-reported Portfolio this
// app already shows (see db.execute_order's own docstring for the
// full boundary explanation). This is a rehearsal tool for a trade
// decision against real prices, not a brokerage integration.
export default function OrderTicket() {
  const [orders, setOrders] = useState<Order[] | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [side, setSide] = useState<Side>("BUY");
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  function refresh() {
    fetch("/api/orders?limit=5")
      .then((r) => (r.ok ? r.json() : { orders: [] }))
      .then((data) => setOrders(data.orders))
      .catch(() => setOrders([]));
  }

  useEffect(refresh, []);

  // Same debounced autocomplete as Portfolio/Watchlist's own ticker
  // inputs -- reuses the same /api/companies/suggest endpoint.
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

  async function submitOrder(rawTicker: string) {
    const q = parseFloat(quantity);
    if (!rawTicker.trim() || !(q > 0) || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const resp = await fetch("/api/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: rawTicker.trim(), side, quantity: q }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ message: "Couldn't place that order." }));
        setError(body.message || "Couldn't place that order.");
        return;
      }
      setTicker("");
      setQuantity("");
      setShowForm(false);
      setSuggestions([]);
      refresh();
      notifyPortfolioUpdated();
    } finally {
      setSubmitting(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    submitOrder(ticker);
  }

  function selectSuggestion(s: CompanySuggestion) {
    setShowSuggestions(false);
    setTicker(s.ticker);
  }

  if (orders === null) return null;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">TRADE (SIMULATED)</p>
        <button
          type="button"
          onClick={() => setShowForm((s) => !s)}
          className="font-mono text-[10px] font-bold text-muted hover:text-accent"
        >
          {showForm ? "CANCEL" : "+ NEW ORDER"}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleSubmit} className="relative mt-2 flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
          <div className="flex gap-1 rounded-lg border border-border bg-bg p-0.5">
            {(["BUY", "SELL"] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSide(s)}
                className={`flex-1 rounded px-2.5 py-1.5 font-mono text-[10px] font-bold ${
                  side === s ? (s === "BUY" ? "bg-accent text-bg" : "bg-danger text-bg") : "text-muted hover:text-text"
                }`}
              >
                {s}
              </button>
            ))}
          </div>

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
            disabled={submitting}
            autoComplete="off"
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />

          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute left-3 right-3 top-[86px] z-10 overflow-hidden rounded-lg border border-border bg-card shadow-lg">
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

          <input
            type="number"
            step="any"
            min="0"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            placeholder="quantity"
            disabled={submitting}
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />

          <button
            type="submit"
            disabled={submitting || !ticker.trim() || !quantity}
            className={`rounded-lg py-2 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40 ${
              side === "BUY" ? "bg-accent" : "bg-danger"
            }`}
          >
            {submitting ? "PLACING..." : `PLACE ${side} ORDER (SIMULATED)`}
          </button>
          {error && <p className="font-mono text-[10px] text-danger">{error}</p>}
          <p className="font-mono text-[9.5px] text-dim">
            Fills instantly at the live quote price -- no real broker, no real money.
          </p>
        </form>
      )}

      {orders.length > 0 && (
        <div className="mt-2 flex flex-col gap-1.5">
          {orders.map((o) => (
            <div
              key={o.order_id}
              className="flex items-center justify-between rounded-lg border border-border-subtle bg-card/60 px-3 py-1.5"
            >
              <span className="font-mono text-[11px] text-text">
                <span className={`font-bold ${o.side === "BUY" ? "text-accent" : "text-danger"}`}>{o.side}</span>{" "}
                {o.quantity} {o.ticker}
              </span>
              <span className="font-mono text-[10px] text-dim">
                {currencySymbol(o.currency)}
                {o.execution_price.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
