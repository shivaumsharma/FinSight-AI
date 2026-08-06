"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import RatingBadge from "./RatingBadge";
import type { ReportSummary, WatchlistItem } from "@/lib/types";

// Scoped down from the original spec's "search stocks/ETFs/mutual
// funds/futures/options/indices/news/orders" -- this app has no data
// for any of those beyond stocks, and the home page's own "ask about a
// ticker or thesis" bar already covers *running new* research. What's
// missing, and what this actually is, is a fast way to FIND something
// you already have -- a past report or a watchlist ticker -- without
// navigating to /reports or /watchlist and scrolling. Both lists are
// small (a handful to a few dozen items for any real user), so a
// client-side filter over data fetched once on open is simpler and
// faster than a new backend search endpoint would be.
export default function SearchOverlay({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistItem[] | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetch("/api/reports/recent?limit=50")
      .then((r) => (r.ok ? r.json() : { reports: [] }))
      .then((data) => setReports(data.reports))
      .catch(() => setReports([]));
    fetch("/api/watchlist")
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => setWatchlist(data.items))
      .catch(() => setWatchlist([]));
  }, []);

  // The visible button alone wasn't enough -- pressing the actual
  // Escape key (what the button's own "ESC" label implied would work)
  // did nothing, and there was no way to dismiss by tapping outside
  // the box either, so a user who missed the small button felt stuck
  // behind the overlay. Both are real close paths now, not just the button.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const q = query.trim().toUpperCase();

  const matchedReports = useMemo(() => {
    if (!q || !reports) return [];
    return reports.filter((r) => r.ticker.includes(q) || (r.company_name || "").toUpperCase().includes(q));
  }, [q, reports]);

  const matchedWatchlist = useMemo(() => {
    if (!q || !watchlist) return [];
    return watchlist.filter((w) => w.ticker.includes(q));
  }, [q, watchlist]);

  const hasResults = matchedReports.length > 0 || matchedWatchlist.length > 0;

  function goToReport(jobId: string) {
    router.push(`/?job=${jobId}`);
    onClose();
  }

  function researchThis() {
    router.push(`/?q=${encodeURIComponent(query.trim())}`);
    onClose();
  }

  return (
    <div
      className="fixed inset-0 z-20 bg-bg/95 backdrop-blur-sm"
      onClick={(e) => {
        // Only the backdrop itself closes on click -- the content div
        // below stops propagation so clicking a result or the input
        // doesn't also dismiss the overlay.
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="mx-auto max-w-2xl px-5 py-6" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          onClick={onClose}
          className="mb-3 flex items-center gap-1.5 font-mono text-xs font-bold text-muted hover:text-accent"
        >
          &larr; BACK
        </button>

        <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3.5 py-3">
          <span className="font-mono text-sm font-bold text-accent">&gt;</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search reports and watchlist..."
            autoFocus
            className="flex-1 bg-transparent font-mono text-sm text-text placeholder:text-muted focus:outline-none"
          />
        </div>

        {q && (
          <div className="mt-4 space-y-4">
            {matchedWatchlist.length > 0 && (
              <div>
                <p className="font-mono text-[10px] tracking-wide text-dim">WATCHLIST</p>
                <div className="mt-2 flex flex-col gap-2">
                  {matchedWatchlist.map((w) => (
                    <div
                      key={w.ticker}
                      className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5"
                    >
                      <span className="font-mono text-sm font-bold text-text">{w.ticker}</span>
                      {w.rating && <RatingBadge rating={w.rating} size="sm" />}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {matchedReports.length > 0 && (
              <div>
                <p className="font-mono text-[10px] tracking-wide text-dim">REPORTS</p>
                <div className="mt-2 flex flex-col gap-2">
                  {matchedReports.map((r) => (
                    <button
                      key={r.job_id}
                      type="button"
                      onClick={() => goToReport(r.job_id)}
                      className="flex w-full items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5 text-left hover:border-accent"
                    >
                      <div className="min-w-0">
                        <div className="font-mono text-sm font-bold text-text">{r.ticker}</div>
                        {r.company_name && <div className="truncate font-mono text-[10px] text-dim">{r.company_name}</div>}
                      </div>
                      <RatingBadge rating={r.rating} size="sm" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {!hasResults && (reports !== null || watchlist !== null) && (
              <div className="rounded-lg border border-border-subtle bg-card/60 px-4 py-4 text-center">
                <p className="font-mono text-xs text-muted">No match in your reports or watchlist.</p>
                <button
                  type="button"
                  onClick={researchThis}
                  className="mt-3 rounded-lg bg-accent px-4 py-2 font-mono text-xs font-bold text-bg hover:opacity-90"
                >
                  Research &quot;{query.trim()}&quot;
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
