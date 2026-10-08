"use client";

import { memo, useState } from "react";
import Link from "next/link";
import RatingBadge from "./RatingBadge";
import { formatShortDate } from "@/lib/format";
import { currencySymbol } from "@/lib/currency";
import { formatPercent, formatPrice } from "@/lib/numberFormat";
import type { LiveQuote } from "@/lib/livePrices";
import type { WatchlistItem } from "@/lib/types";

// Render counter used only by the benchmark page (src/app/bench); null in normal use.
export const renderProbe: { onRender: ((ticker: string) => void) | null } = { onRender: null };

interface Props {
  item: WatchlistItem;
  // Latest streamed quote; its object only changes when the price does, so memo skips unchanged rows.
  live?: LiveQuote;
  isDemo?: boolean;
  onRemove?: (ticker: string) => void;
}

// One watchlist line. Memoized so a re-render of the list only repaints rows whose item or live quote changed.
function WatchlistRow({ item, live, isDemo = false, onRemove }: Props) {
  renderProbe.onRender?.(item.ticker);
  const price = live?.price ?? item.price;
  const changePct = live ? live.changePct : item.change_pct;

  // Flash green/red when the price moves. The counter is the key of the price element, which restarts the CSS animation.
  const [lastPrice, setLastPrice] = useState(price);
  const [flash, setFlash] = useState<{ dir: "up" | "down"; n: number } | null>(null);
  if (price !== lastPrice) {
    setLastPrice(price);
    if (price !== null && lastPrice !== null && price !== lastPrice) {
      setFlash((f) => ({ dir: price > lastPrice ? "up" : "down", n: (f?.n ?? 0) + 1 }));
    }
  }

  const corporateActions = [
    item.next_earnings_date && `Earnings ${formatShortDate(item.next_earnings_date)}`,
    item.next_ex_dividend_date && `Ex-div ${formatShortDate(item.next_ex_dividend_date)}`,
  ].filter(Boolean);

  return (
    <div
      data-testid={`watchlist-row-${item.ticker}`}
      className={`rounded-lg border px-3.5 py-2.5 ${isDemo ? "border-dashed border-border-subtle bg-card/50" : "border-border bg-card"}`}
    >
      <div className="flex items-center justify-between">
        <Link href={`/stock/${item.ticker}`} className="flex flex-1 items-center justify-between gap-2 min-w-0">
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-sm font-bold text-text hover:text-accent">{item.ticker}</span>
            {item.rating ? (
              <RatingBadge rating={item.rating} size="sm" />
            ) : (
              <span className="font-mono text-[10px] text-dim">not yet researched</span>
            )}
          </div>
          {price !== null && (
            <div className="text-right">
              <div
                key={flash?.n ?? 0}
                data-flash={flash ? flash.dir : undefined}
                className={`rounded px-1 font-mono text-sm text-text ${flash ? (flash.dir === "up" ? "flash-up" : "flash-down") : ""}`}
              >
                {formatPrice(price, currencySymbol(live?.currency ?? item.currency))}
              </div>
              {changePct !== null && (
                <div className={`font-mono text-[10px] ${changePct >= 0 ? "text-accent" : "text-danger"}`}>
                  {formatPercent(changePct, { signed: true })}
                </div>
              )}
            </div>
          )}
        </Link>
        {!isDemo && (
          <button
            type="button"
            onClick={() => onRemove?.(item.ticker)}
            title="Remove from watchlist"
            aria-label={`Remove ${item.ticker} from watchlist`}
            className="-my-2 -mr-2 ml-1 flex h-9 w-9 items-center justify-center font-mono text-xs text-dim hover:text-danger"
          >
            &times;
          </button>
        )}
      </div>
      {corporateActions.length > 0 && (
        <p className="mt-1.5 font-mono text-[10px] text-dim">{corporateActions.join(" · ")}</p>
      )}
    </div>
  );
}

// The unmemoized body is exported for the benchmark, which compares it with the memoized default.
export { WatchlistRow as WatchlistRowUnmemoized };
export default memo(WatchlistRow);
