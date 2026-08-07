"use client";

import { useEffect, useState } from "react";
import type { MarketSentiment } from "@/lib/types";

// "FinSight Research Sentiment" -- the Buy/Sell share of every
// distinct ticker's latest completed rating across ALL users, NOT a
// market-wide indicator (that would need a real institutional data
// feed this app doesn't have -- see the earlier FII/DII research).
// Hold is excluded from the ratio (see main.py's get_market_sentiment
// docstring for why) but still shown as its own count below the bar.
export default function SentimentGauge() {
  const [data, setData] = useState<MarketSentiment | null>(null);

  useEffect(() => {
    fetch("/api/market/sentiment")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (data === null) return null;

  const hasVotes = data.buy_pct !== null && data.sell_pct !== null;

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">FINSIGHT RESEARCH SENTIMENT</p>

      <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-3">
        {hasVotes ? (
          <>
            <div className="flex items-baseline justify-between font-mono text-[11px] font-bold">
              <span className="text-accent">{data.buy_pct!.toFixed(0)}% BUY</span>
              <span className="text-danger">{data.sell_pct!.toFixed(0)}% SELL</span>
            </div>
            <div className="mt-1.5 flex h-3 w-full overflow-hidden rounded-full border border-border-subtle bg-bg">
              <div className="h-full bg-accent" style={{ width: `${data.buy_pct}%` }} />
              <div className="h-full bg-danger" style={{ width: `${data.sell_pct}%` }} />
            </div>
            <p className="mt-2 font-mono text-[10px] text-dim">
              {data.buy_count} Buy · {data.hold_count} Hold · {data.sell_count} Sell -- across all researched
              tickers, Hold excluded from the ratio above.
            </p>
          </>
        ) : (
          <p className="font-mono text-[11px] text-dim">Not enough completed research yet to show a sentiment split.</p>
        )}
      </div>
    </div>
  );
}
