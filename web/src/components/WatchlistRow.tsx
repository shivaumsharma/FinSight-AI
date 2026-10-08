"use client";

import { memo } from "react";
import Link from "next/link";
import RatingBadge from "./RatingBadge";
import { formatShortDate } from "@/lib/format";
import { currencySymbol } from "@/lib/currency";
import { formatPercent, formatPrice } from "@/lib/numberFormat";
import type { WatchlistItem } from "@/lib/types";

interface Props {
  item: WatchlistItem;
  isDemo?: boolean;
  onRemove?: (ticker: string) => void;
}

// One watchlist line. Memoized so a re-render of the list only repaints rows whose item changed.
function WatchlistRow({ item, isDemo = false, onRemove }: Props) {
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
          {item.price !== null && (
            <div className="text-right">
              <div className="font-mono text-sm text-text">
                {formatPrice(item.price, currencySymbol(item.currency))}
              </div>
              {item.change_pct !== null && (
                <div className={`font-mono text-[10px] ${item.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                  {formatPercent(item.change_pct, { signed: true })}
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

export default memo(WatchlistRow);
