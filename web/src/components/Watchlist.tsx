"use client";

import { useCallback, useEffect, useState } from "react";
import WatchlistRow from "./WatchlistRow";
import LoadError from "./LoadError";
import ListSkeleton from "./ListSkeleton";
import type { CompanySuggestion, WatchlistItem } from "@/lib/types";

// Static, clearly-labeled sample rows shown only when a user's real
// watchlist is empty -- gives new users a sense of what the feature
// looks like populated, instead of a blank input on first login.
// Fake price/rating data, never sent to the backend; tapping one still
// navigates to that ticker's real /stock page since the ticker itself
// is real.
const DEMO_ITEMS: WatchlistItem[] = [
  {
    ticker: "AAPL",
    price: 233.14,
    change_pct: 1.42,
    currency: "USD",
    rating: "Buy",
    added_at: 0,
    next_earnings_date: null,
    next_ex_dividend_date: null,
    last_dividend_amount: null,
    last_split: null,
  },
  {
    ticker: "NVDA",
    price: 178.9,
    change_pct: -0.86,
    currency: "USD",
    rating: "Hold",
    added_at: 0,
    next_earnings_date: null,
    next_ex_dividend_date: null,
    last_dividend_amount: null,
    last_split: null,
  },
  {
    ticker: "TSLA",
    price: 312.55,
    change_pct: 2.07,
    currency: "USD",
    rating: "Sell",
    added_at: 0,
    next_earnings_date: null,
    next_ex_dividend_date: null,
    last_dividend_amount: null,
    last_split: null,
  },
];

// Always renders, even with zero items -- the add-ticker input is the
// primary way a user grows this list, so it needs to stay visible
// rather than disappearing until something's already on it.
export default function Watchlist() {
  const [items, setItems] = useState<WatchlistItem[] | null>(null);
  const [ticker, setTicker] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const refresh = useCallback(() => {
    fetch("/api/watchlist")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => {
        setItems(data.items);
        setLoadFailed(false);
      })
      // Keep whatever was already on screen on a failed re-fetch; only a
      // failed FIRST load shows the error block (never the empty state).
      .catch(() => setLoadFailed(true));
  }, []);

  useEffect(refresh, [refresh]);

  // Debounced -- fetching on every keystroke would fire a request per
  // character typed. 2-char minimum keeps a single keypress from
  // querying the full ~10k-company index for nothing useful.
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

  async function submitTicker(rawTicker: string) {
    if (!rawTicker.trim() || adding) return;
    setAdding(true);
    setError(null);
    try {
      const resp = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: rawTicker.trim() }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ message: "Couldn't add that ticker." }));
        setError(body.message || "Couldn't add that ticker.");
        return;
      }
      setTicker("");
      setSuggestions([]);
      setShowSuggestions(false);
      refresh();
    } finally {
      setAdding(false);
    }
  }

  function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    submitTicker(ticker);
  }

  // onMouseDown, not onClick -- fires before the input's onBlur, so
  // selecting a suggestion doesn't race the dropdown closing before
  // the click registers.
  function selectSuggestion(s: CompanySuggestion) {
    setShowSuggestions(false);
    submitTicker(s.ticker);
  }

  const handleRemove = useCallback(async (t: string) => {
    // Optimistic removal -- the DELETE endpoint is idempotent and
    // near-instant, no need to wait for the round-trip before the
    // card disappears. refresh() afterwards re-syncs from the server
    // either way, so a failed delete doesn't leave the ticker
    // permanently (and incorrectly) missing from the UI -- same
    // approach as Portfolio.tsx's handleRemove.
    setItems((prev) => (prev ? prev.filter((i) => i.ticker !== t) : prev));
    await fetch(`/api/watchlist/${t}`, { method: "DELETE" }).catch(() => {});
    refresh();
  }, [refresh]);

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">WATCHLIST</p>

      {loadFailed && items === null && <LoadError what="your watchlist" onRetry={refresh} className="mt-2" />}

      {items === null && !loadFailed && <ListSkeleton rows={3} className="mt-2" />}

      {items && items.length > 0 && (
        <div className="mt-2 flex flex-col gap-2">
          {items.map((item) => (
            <WatchlistRow key={item.ticker} item={item} onRemove={handleRemove} />
          ))}
        </div>
      )}

      {items && items.length === 0 && (
        <div className="mt-2">
          <p className="font-mono text-[10px] text-dim">
            Empty for now -- here&apos;s what it looks like once you add a stock:
          </p>
          <div className="mt-2 flex flex-col gap-2 opacity-70">
            {DEMO_ITEMS.map((item) => (
              <WatchlistRow key={item.ticker} item={item} isDemo />
            ))}
          </div>
        </div>
      )}

      <form onSubmit={handleAdd} className="relative mt-2 flex gap-2">
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
          placeholder="add ticker or company name..."
          disabled={adding}
          autoComplete="off"
          className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={adding || !ticker.trim()}
          className="rounded-lg border border-border bg-card px-3.5 py-2 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent disabled:opacity-50"
        >
          {adding ? "..." : "ADD"}
        </button>

        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute left-0 right-[68px] top-full z-10 mt-1 overflow-hidden rounded-lg border border-border bg-card shadow-lg">
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
      </form>
      {error && <p className="mt-1.5 font-mono text-[10px] text-danger">{error}</p>}
    </div>
  );
}
