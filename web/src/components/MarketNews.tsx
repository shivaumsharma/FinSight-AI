"use client";

import { useEffect, useState } from "react";
import { formatShortDate } from "@/lib/format";
import type { NewsArticle } from "@/lib/types";

// TRENDING = general market headlines (unchanged /v1/news/market feed).
// ON MY STOCKS = last week's company-specific news for every ticker on
// the watchlist and/or portfolio (/v1/news/my-stocks), reusing the
// SAME per-ticker Finnhub client the research pipeline's own News
// Selection stage depends on -- not a new provider.
export default function MarketNews() {
  const [tab, setTab] = useState<"trending" | "mine">("trending");
  const [trending, setTrending] = useState<NewsArticle[] | null>(null);
  const [mine, setMine] = useState<NewsArticle[] | null>(null);

  useEffect(() => {
    fetch("/api/news/market?limit=8")
      .then((r) => (r.ok ? r.json() : { articles: [] }))
      .then((data) => setTrending(data.articles))
      .catch(() => setTrending([]));
  }, []);

  useEffect(() => {
    if (tab !== "mine" || mine !== null) return;
    fetch("/api/news/my-stocks?limit=10")
      .then((r) => (r.ok ? r.json() : { articles: [] }))
      .then((data) => setMine(data.articles))
      .catch(() => setMine([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  if (trending === null) return null;
  // Nothing to show at all -- no trending feed configured and the
  // user hasn't opened "On My Stocks" yet (or it's also empty).
  if (trending.length === 0 && tab === "trending") return null;

  const articles = tab === "trending" ? trending : mine;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">MARKET NEWS</p>
        <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
          {(["trending", "mine"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded px-2.5 py-1 font-mono text-[10px] font-bold ${
                tab === t ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {t === "trending" ? "TRENDING" : "ON MY STOCKS"}
            </button>
          ))}
        </div>
      </div>

      {articles === null ? (
        <p className="mt-2 font-mono text-[11px] text-dim">Loading...</p>
      ) : articles.length > 0 ? (
        <div className="mt-2 flex flex-col gap-2">
          {articles.map((a) => (
            <a
              key={a.url}
              href={a.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-lg border border-border bg-card px-3.5 py-2.5 hover:border-accent"
            >
              <p className="font-mono text-xs text-text">{a.headline}</p>
              <p className="mt-1 font-mono text-[10px] text-dim">
                {a.ticker && <span className="font-bold text-muted">{a.ticker} · </span>}
                {a.source} &middot; {formatShortDate(a.date)}
              </p>
            </a>
          ))}
        </div>
      ) : (
        <p className="mt-2 font-mono text-[11px] text-dim">
          No recent news for tickers on your watchlist or portfolio.
        </p>
      )}
    </div>
  );
}
