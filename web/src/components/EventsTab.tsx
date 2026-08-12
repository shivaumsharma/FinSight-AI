"use client";

import { useEffect, useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { formatShortDate } from "@/lib/format";

interface EventsData {
  next_earnings_date: string | null;
  next_ex_dividend_date: string | null;
  dividends: { date: string; amount: number }[];
  splits: { date: string; ratio: number }[];
}

interface TimelineRow {
  date: string;
  label: string;
  detail: string;
}

// No bonus/rights/buyback/ESOP rows here -- yfinance has no such
// concepts at all (see app/data/market_data.py's
// get_corporate_actions_history docstring), only real dividend/split
// history plus the same upcoming earnings/ex-dividend dates the
// Watchlist already shows.
export default function EventsTab({ ticker, currency }: { ticker: string; currency: string }) {
  const [data, setData] = useState<EventsData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setData(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/events`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, [ticker]);

  if (error) {
    return <p className="mt-4 py-6 text-center font-mono text-[11px] text-dim">Couldn&apos;t load events for this ticker.</p>;
  }
  if (!data) {
    return <div className="mt-4 h-[200px] animate-pulse rounded-lg border border-border bg-card/60" />;
  }

  const symbol = currencySymbol(currency);
  const rows: TimelineRow[] = [];

  if (data.next_earnings_date) {
    rows.push({ date: data.next_earnings_date, label: "Upcoming Earnings", detail: "Next reporting date" });
  }
  if (data.next_ex_dividend_date) {
    rows.push({ date: data.next_ex_dividend_date, label: "Upcoming Ex-Dividend", detail: "Next ex-dividend date" });
  }
  for (const d of data.dividends.slice().reverse()) {
    rows.push({ date: d.date, label: "Dividend", detail: `${symbol}${d.amount.toFixed(2)} per share` });
  }
  for (const s of data.splits.slice().reverse()) {
    rows.push({ date: s.date, label: "Stock Split", detail: `${s.ratio.toFixed(2)}:1` });
  }

  if (rows.length === 0) {
    return (
      <div className="mt-4 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-6 text-center">
        <p className="font-mono text-[11px] text-dim">No corporate action history available for this ticker.</p>
      </div>
    );
  }

  return (
    <div className="mt-4 flex flex-col gap-2">
      {rows.map((row, i) => (
        <div key={`${row.label}-${row.date}-${i}`} className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5">
          <div>
            <div className="font-mono text-xs font-bold text-text">{row.label}</div>
            <div className="font-mono text-[10px] text-dim">{row.detail}</div>
          </div>
          <span className="font-mono text-[10px] text-muted">{formatShortDate(row.date)}</span>
        </div>
      ))}
    </div>
  );
}
