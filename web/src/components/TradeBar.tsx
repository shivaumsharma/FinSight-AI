"use client";

import { useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { notifyPortfolioUpdated } from "@/lib/portfolioEvents";

// Sticky Buy/Sell bar, always visible on the stock detail page (unlike
// OrderTicket's own collapsed-behind-a-toggle form on the home page) --
// reuses the exact same POST /v1/orders endpoint and
// notifyPortfolioUpdated() cross-component event OrderTicket already
// does, just scoped to this page's ticker with no ticker input needed.
// Simulated only: fills instantly at the live quote price, no real
// broker, no real money -- same boundary as everywhere else this app
// trades (see db.execute_order's own docstring).
export default function TradeBar({ ticker, currency }: { ticker: string; currency: string }) {
  const [quantity, setQuantity] = useState("");
  const [submitting, setSubmitting] = useState<"BUY" | "SELL" | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function submit(side: "BUY" | "SELL") {
    const q = parseFloat(quantity);
    if (!(q > 0) || submitting) return;
    setSubmitting(side);
    setMessage(null);
    try {
      const resp = await fetch("/api/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, side, quantity: q }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        setMessage(body.message || "Couldn't place that order.");
        return;
      }
      setMessage(`${side} ${q} ${ticker} filled at ${currencySymbol(currency)}${body.execution_price?.toFixed(2) ?? "?"} (simulated).`);
      setQuantity("");
      notifyPortfolioUpdated();
    } finally {
      setSubmitting(null);
    }
  }

  return (
    <div
      className="fixed inset-x-0 z-10 border-t border-border bg-bg/95 px-4 py-2.5 backdrop-blur"
      // bottom-16 (4rem) was tuned to sit exactly above BottomNav's own
      // ~64px height -- now that BottomNav adds env(safe-area-inset-bottom)
      // on top of that, this needs the same inset added or TradeBar
      // sits too low and hides partly behind the now-taller nav on any
      // notched/gesture-bar device.
      style={{ bottom: "calc(4rem + env(safe-area-inset-bottom))" }}
    >
      <div className="mx-auto flex max-w-2xl items-center gap-2">
        <input
          type="number"
          step="any"
          min="0"
          value={quantity}
          onChange={(e) => setQuantity(e.target.value)}
          placeholder="qty"
          disabled={submitting !== null}
          className="w-20 rounded-lg border border-border bg-card px-2.5 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
        />
        <button
          type="button"
          onClick={() => submit("BUY")}
          disabled={submitting !== null || !quantity}
          className="flex-1 rounded-lg bg-accent py-2 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting === "BUY" ? "..." : "BUY (SIM)"}
        </button>
        <button
          type="button"
          onClick={() => submit("SELL")}
          disabled={submitting !== null || !quantity}
          className="flex-1 rounded-lg bg-danger py-2 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting === "SELL" ? "..." : "SELL (SIM)"}
        </button>
      </div>
      {message && <p className="mx-auto mt-1 max-w-2xl font-mono text-[9.5px] text-dim">{message}</p>}
    </div>
  );
}
