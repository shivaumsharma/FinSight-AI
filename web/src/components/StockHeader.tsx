"use client";

import Link from "next/link";
import { currencySymbol } from "@/lib/currency";
import { fmtPrice } from "@/lib/stockFormat";
import type { StockOverview } from "@/lib/types";

function BackIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <path d="M15 5l-7 7 7 7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Header for the stock detail page -- back link, name/ticker, live
// price + absolute/percent change with the same green/red convention
// as every other price tile in this app (Watchlist/MarketMovers/
// Portfolio). No market-open/closed/pre-open state here: yfinance's
// `.info` doesn't expose a reliable market-state field for both US and
// NSE tickers, and guessing from wall-clock time zones is worse than
// just not claiming it.
export default function StockHeader({ overview }: { overview: StockOverview }) {
  const symbol = currencySymbol(overview.currency);
  const changeAbs =
    overview.change_pct !== null && overview.previous_close !== null
      ? (overview.change_pct / 100) * overview.previous_close
      : null;
  const positive = (overview.change_pct ?? 0) >= 0;

  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <Link href="/" className="mb-2 inline-flex items-center gap-1 font-mono text-[10px] font-bold text-dim hover:text-accent">
          <BackIcon />
          BACK
        </Link>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-lg font-bold text-text">{overview.ticker}</span>
        </div>
        {overview.company_name && <div className="truncate font-mono text-xs text-muted">{overview.company_name}</div>}
      </div>
      <div className="shrink-0 text-right">
        <div className="font-mono text-xl font-bold text-text">{fmtPrice(overview.price, symbol)}</div>
        {overview.change_pct !== null && (
          <div className={`font-mono text-xs font-bold ${positive ? "text-accent" : "text-danger"}`}>
            {changeAbs !== null && `${positive ? "+" : "-"}${symbol}${Math.abs(changeAbs).toFixed(2)} `}
            ({positive ? "+" : ""}
            {overview.change_pct.toFixed(2)}%)
          </div>
        )}
      </div>
    </div>
  );
}
