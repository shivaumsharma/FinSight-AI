"use client";

import { useEffect, useState } from "react";
import { formatShortDate } from "@/lib/format";
import type { NewsArticle } from "@/lib/types";

// General market headlines, not tied to any one company -- distinct
// from the per-ticker news a research report already cites. Renders
// nothing while loading or on a genuinely empty feed (no FINNHUB_API_KEY
// configured, or a transient fetch failure) rather than showing an
// empty "MARKET NEWS" section header with nothing under it.
export default function MarketNews() {
  const [articles, setArticles] = useState<NewsArticle[] | null>(null);

  useEffect(() => {
    fetch("/api/news/market?limit=8")
      .then((r) => (r.ok ? r.json() : { articles: [] }))
      .then((data) => setArticles(data.articles))
      .catch(() => setArticles([]));
  }, []);

  if (!articles || articles.length === 0) return null;

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">MARKET NEWS</p>
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
              {a.source} &middot; {formatShortDate(a.date)}
            </p>
          </a>
        ))}
      </div>
    </div>
  );
}
