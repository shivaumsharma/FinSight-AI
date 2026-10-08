"use client";

import { useEffect, useState } from "react";
import SectionSkeleton from "./SectionSkeleton";
import { Button } from "./ui";
import { currencySymbol } from "@/lib/currency";
import { PORTFOLIO_UPDATED_EVENT, notifyPortfolioUpdated } from "@/lib/portfolioEvents";
import type { CompanySuggestion, Order } from "@/lib/types";
import { formatPrice, formatQuantity } from "@/lib/numberFormat";
import { orderErrorMessage, validateOrder } from "@/lib/orderValidation";

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
  const [step, setStep] = useState<"edit" | "review">("edit");
  const [touched, setTouched] = useState({ ticker: false, quantity: false });
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  function refresh() {
    fetch("/api/orders?limit=5")
      .then((r) => (r.ok ? r.json() : { orders: [] }))
      .then((data) => setOrders(data.orders))
      .catch(() => setOrders([]));
  }

  useEffect(refresh, []);

  // A holding can also change via a path that never goes through this
  // component's own submitOrder (e.g. Portfolio's "try sample data"
  // seeding) -- listen for the same cross-component event Portfolio.tsx
  // already relies on so this order history never goes stale either.
  useEffect(() => {
    window.addEventListener(PORTFOLIO_UPDATED_EVENT, refresh);
    return () => window.removeEventListener(PORTFOLIO_UPDATED_EVENT, refresh);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  const validation = validateOrder({ ticker, quantity });

  async function submitOrder() {
    if (!validation.ok || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const resp = await fetch("/api/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim(), side, quantity: validation.quantity }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        setError(orderErrorMessage(resp.status, body));
        setStep("edit");
        return;
      }
      setTicker("");
      setQuantity("");
      setTouched({ ticker: false, quantity: false });
      setStep("edit");
      setShowForm(false);
      setSuggestions([]);
      refresh();
      notifyPortfolioUpdated();
    } catch {
      setError(orderErrorMessage(503, null));
      setStep("edit");
    } finally {
      setSubmitting(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (step === "edit") {
      setTouched({ ticker: true, quantity: true });
      if (validation.ok) {
        setError(null);
        setStep("review");
      }
      return;
    }
    submitOrder();
  }

  function selectSuggestion(s: CompanySuggestion) {
    setShowSuggestions(false);
    setTicker(s.ticker);
  }

  if (orders === null) return <SectionSkeleton label="TRADE (SIMULATED)" rows={1} />;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">TRADE (SIMULATED)</p>
        <button
          type="button"
          onClick={() => setShowForm((s) => !s)}
          className="-my-2 inline-block py-2 font-mono text-[10px] font-bold text-muted hover:text-accent"
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
                onClick={() => {
                  setSide(s);
                  setStep("edit");
                }}
                className={`flex-1 rounded px-2.5 py-1.5 font-mono text-[10px] font-bold ${
                  side === s ? (s === "BUY" ? "bg-accent text-bg" : "bg-danger text-bg") : "text-muted hover:text-text"
                }`}
              >
                {s}
              </button>
            ))}
          </div>

          <label htmlFor="order-ticker" className="font-mono text-[10px] text-muted">
            Ticker or company
          </label>
          <input
            id="order-ticker"
            type="text"
            value={ticker}
            aria-invalid={touched.ticker && !!validation.errors.ticker}
            aria-describedby={touched.ticker && validation.errors.ticker ? "order-ticker-error" : undefined}
            onChange={(e) => {
              setTicker(e.target.value);
              setShowSuggestions(true);
              setStep("edit");
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => {
              setShowSuggestions(false);
              setTouched((t) => ({ ...t, ticker: true }));
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") setShowSuggestions(false);
            }}
            placeholder="e.g. AAPL or Apple"
            disabled={submitting}
            autoComplete="off"
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />

          {touched.ticker && validation.errors.ticker && (
            <p id="order-ticker-error" role="alert" className="font-mono text-[10px] text-danger">
              {validation.errors.ticker}
            </p>
          )}

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

          <label htmlFor="order-quantity" className="font-mono text-[10px] text-muted">
            Quantity (shares)
          </label>
          <input
            id="order-quantity"
            type="text"
            inputMode="decimal"
            value={quantity}
            aria-invalid={touched.quantity && !!validation.errors.quantity}
            aria-describedby={touched.quantity && validation.errors.quantity ? "order-quantity-error" : undefined}
            onChange={(e) => {
              setQuantity(e.target.value);
              setStep("edit");
            }}
            onBlur={() => setTouched((t) => ({ ...t, quantity: true }))}
            placeholder="e.g. 10"
            disabled={submitting}
            className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />

          {touched.quantity && validation.errors.quantity && (
            <p id="order-quantity-error" role="alert" className="font-mono text-[10px] text-danger">
              {validation.errors.quantity}
            </p>
          )}

          {step === "review" && validation.ok && (
            <div className="rounded-lg border border-border bg-bg px-3 py-2 font-mono text-[11px] text-text" data-testid="order-review">
              <span className={`font-bold ${side === "BUY" ? "text-accent" : "text-danger"}`}>{side}</span>{" "}
              {formatQuantity(validation.quantity)} {ticker.trim().toUpperCase()} at the live quote when you confirm.
              <span className="block text-[9.5px] text-dim">Simulated: no real broker, no real money.</span>
            </div>
          )}

          <div className="flex gap-2">
            {step === "review" && (
              <Button onClick={() => setStep("edit")} disabled={submitting}>
                Edit order
              </Button>
            )}
            <Button type="submit" variant={side === "BUY" ? "primary" : "danger"} loading={submitting} className="flex-1">
              {submitting ? "Placing order..." : step === "review" ? `Confirm ${side.toLowerCase()} (simulated)` : "Review order"}
            </Button>
          </div>
          {error && (
            <p role="alert" className="font-mono text-[10px] text-danger">
              {error}
            </p>
          )}
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
              className="flex flex-col gap-0.5 rounded-lg border border-border-subtle bg-card/60 px-3 py-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] text-text">
                  <span className={`font-bold ${o.side === "BUY" ? "text-accent" : "text-danger"}`}>{o.side}</span>{" "}
                  {formatQuantity(o.quantity)} {o.ticker}
                </span>
                <span className="font-mono text-[10px] text-dim">
                  {formatPrice(o.execution_price, currencySymbol(o.currency))}
                </span>
              </div>
              {o.rationale && <p className="font-mono text-[9.5px] text-dim">{o.rationale}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
