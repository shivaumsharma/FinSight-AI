"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AddToWatchlistButton from "./AddToWatchlistButton";
import { currencySymbol } from "@/lib/currency";
import { formatCompact } from "@/lib/format";
import type { ScreenerData, ScreenerFilters, ScreenerRow } from "@/lib/types";

const SORT_OPTIONS: { value: string; label: string }[] = [
  { value: "market_cap", label: "Market Cap" },
  { value: "change_pct", label: "Day Change" },
  { value: "pe_ratio", label: "P/E" },
  { value: "pb_ratio", label: "P/B" },
  { value: "dividend_yield", label: "Dividend Yield" },
  { value: "volume", label: "Volume" },
];

type FilterField = keyof ScreenerFilters;

const FILTER_FIELDS: { field: FilterField; label: string; placeholder: string }[] = [
  { field: "market_cap_min", label: "MARKET CAP MIN", placeholder: "e.g. 1000000000" },
  { field: "market_cap_max", label: "MARKET CAP MAX", placeholder: "no limit" },
  { field: "pe_max", label: "P/E MAX", placeholder: "e.g. 25" },
  { field: "pb_max", label: "P/B MAX", placeholder: "e.g. 5" },
  { field: "dividend_yield_min", label: "DIV YIELD MIN %", placeholder: "e.g. 1" },
  { field: "change_pct_min", label: "DAY CHANGE MIN %", placeholder: "e.g. -5" },
  { field: "change_pct_max", label: "DAY CHANGE MAX %", placeholder: "e.g. 5" },
  { field: "volume_min", label: "VOLUME MIN", placeholder: "e.g. 1000000" },
];

function buildQuery(filters: ScreenerFilters, sortBy: string, sortDesc: boolean): string {
  const params = new URLSearchParams();
  (Object.keys(filters) as FilterField[]).forEach((key) => {
    const value = filters[key];
    if (value !== undefined && value !== null && !Number.isNaN(value)) {
      params.set(key, String(value));
    }
  });
  params.set("sort_by", sortBy);
  params.set("sort_desc", String(sortDesc));
  return params.toString();
}

function ScreenerRowCard({ row }: { row: ScreenerRow }) {
  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-2.5">
      <div className="flex items-center justify-between">
        <Link href={`/stock/${row.ticker}`} className="flex min-w-0 flex-1 items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="font-mono text-sm font-bold text-text hover:text-accent">{row.ticker}</div>
            <div className="truncate font-mono text-[10px] text-dim">{row.name}</div>
          </div>
          <div className="text-right">
            <div className="font-mono text-sm text-text">
              {currencySymbol(row.currency)}
              {row.price.toFixed(2)}
            </div>
            {row.change_pct !== null && (
              <div className={`font-mono text-[10px] font-bold ${row.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                {row.change_pct >= 0 ? "+" : ""}
                {row.change_pct.toFixed(2)}%
              </div>
            )}
          </div>
        </Link>
        <div className="ml-3">
          <AddToWatchlistButton ticker={row.ticker} />
        </div>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-dim">
        <span>MCAP {row.market_cap !== null ? formatCompact(row.market_cap) : "--"}</span>
        <span>P/E {row.pe_ratio !== null ? row.pe_ratio.toFixed(1) : "--"}</span>
        <span>P/B {row.pb_ratio !== null ? row.pb_ratio.toFixed(1) : "--"}</span>
        <span>DIV {row.dividend_yield !== null ? `${row.dividend_yield.toFixed(2)}%` : "--"}</span>
        <span>VOL {row.volume !== null ? formatCompact(row.volume) : "--"}</span>
      </div>
    </div>
  );
}

// "Screener (Tracked Universe)" -- same scope as MarketMovers/
// SentimentGauge (curated backtest tickers + every user's watchlist
// tickers, unioned globally, see app/reasoning/market_movers.py's
// docstring), NOT a full-market scan. Filters/sort are forwarded
// straight to GET /v1/market/screener as query params; the backend owns
// defaults and the None-sinks-last sort rule.
export default function Screener() {
  const [filters, setFilters] = useState<ScreenerFilters>({});
  const [sortBy, setSortBy] = useState("market_cap");
  const [sortDesc, setSortDesc] = useState(true);
  const [data, setData] = useState<ScreenerData | null>(null);
  const [showFilters, setShowFilters] = useState(true);

  useEffect(() => {
    setData(null);
    const query = buildQuery(filters, sortBy, sortDesc);
    fetch(`/api/market/screener?${query}`)
      .then((r) => (r.ok ? r.json() : { results: [], total_matched: 0, universe_size: 0 }))
      .then(setData)
      .catch(() => setData({ results: [], total_matched: 0, universe_size: 0 }));
    // Re-runs whenever a filter, sort field, or sort direction changes --
    // no debounce needed since these are number inputs committed on
    // blur/change, not a per-keystroke search box.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, sortBy, sortDesc]);

  function updateFilter(field: FilterField, raw: string) {
    setFilters((prev) => {
      const next = { ...prev };
      if (raw.trim() === "") {
        delete next[field];
      } else {
        const parsed = Number(raw);
        if (!Number.isNaN(parsed)) next[field] = parsed;
      }
      return next;
    });
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">
          {data ? `${data.total_matched} OF ${data.universe_size} TRACKED` : "LOADING..."}
        </p>
        <button
          type="button"
          onClick={() => setShowFilters((v) => !v)}
          className="font-mono text-[10px] font-bold text-muted hover:text-accent"
        >
          {showFilters ? "HIDE FILTERS" : "SHOW FILTERS"}
        </button>
      </div>

      {showFilters && (
        <div className="mt-2 grid grid-cols-2 gap-2 rounded-lg border border-border bg-card p-3">
          {FILTER_FIELDS.map(({ field, label, placeholder }) => (
            <label key={field} className="flex flex-col gap-1">
              <span className="font-mono text-[9px] tracking-wide text-dim">{label}</span>
              <input
                type="number"
                inputMode="decimal"
                placeholder={placeholder}
                value={filters[field] ?? ""}
                onChange={(e) => updateFilter(field, e.target.value)}
                className="rounded border border-border bg-bg px-2 py-1.5 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent"
              />
            </label>
          ))}
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="flex-1 rounded-lg border border-border bg-card px-2.5 py-1.5 font-mono text-[11px] text-text focus:outline-none focus:border-accent"
        >
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              Sort: {opt.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setSortDesc((v) => !v)}
          title={sortDesc ? "Descending" : "Ascending"}
          className="rounded-lg border border-border bg-card px-3 py-1.5 font-mono text-[11px] font-bold text-muted hover:border-accent hover:text-accent"
        >
          {sortDesc ? "↓" : "↑"}
        </button>
      </div>

      {data === null ? (
        <p className="mt-3 font-mono text-[11px] text-dim">Loading...</p>
      ) : data.results.length > 0 ? (
        <div className="mt-3 flex flex-col gap-2">
          {data.results.map((row) => (
            <ScreenerRowCard key={row.ticker} row={row} />
          ))}
        </div>
      ) : (
        <p className="mt-3 font-mono text-[11px] text-dim">No tracked tickers match these filters.</p>
      )}
    </div>
  );
}
