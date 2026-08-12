"use client";

import { useEffect, useState } from "react";
import type { NewsArticle } from "@/lib/types";

// No per-article sentiment score here -- FinBERT sentiment in this app
// is only ever computed over retrieved filing/news TEXT for a full
// research run, never as a per-headline classifier on demand. The
// category chips (litigation/regulatory/competitive/macro/other) are
// real, keyword-based (see app/reporting/news_client.py), not guessed.
export default function NewsTab({ ticker }: { ticker: string }) {
  const [articles, setArticles] = useState<NewsArticle[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setArticles(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/news`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => setArticles(data.articles))
      .catch(() => setError(true));
  }, [ticker]);

  if (error) {
    return <p className="mt-4 py-6 text-center font-mono text-[11px] text-dim">Couldn&apos;t load news for this ticker.</p>;
  }
  if (!articles) {
    return (
      <div className="mt-4 flex flex-col gap-2">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-[68px] animate-pulse rounded-lg border border-border-subtle bg-card/60" />
        ))}
      </div>
    );
  }
  if (articles.length === 0) {
    return (
      <div className="mt-4 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-6 text-center">
        <p className="font-mono text-[11px] text-dim">No recent news coverage found for this ticker.</p>
      </div>
    );
  }

  return (
    <div className="mt-4 flex flex-col gap-2">
      {articles.map((article) => (
        <a
          key={article.url}
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block rounded-lg border border-border bg-card px-3.5 py-2.5 hover:border-accent"
        >
          <div className="font-mono text-xs font-bold text-text">{article.headline}</div>
          <div className="mt-1 flex items-center justify-between">
            <span className="font-mono text-[10px] text-dim">
              {article.source} &middot; {article.date}
            </span>
            {article.categories && article.categories[0] !== "other" && (
              <div className="flex gap-1">
                {article.categories.map((c) => (
                  <span key={c} className="rounded border border-border-subtle px-1.5 py-0.5 font-mono text-[9px] uppercase text-muted">
                    {c}
                  </span>
                ))}
              </div>
            )}
          </div>
        </a>
      ))}
    </div>
  );
}
