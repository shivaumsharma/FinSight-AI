"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SectionSkeleton from "./SectionSkeleton";
import LoadError from "./LoadError";
import type { IndexQuote } from "@/lib/types";
import { formatNumber, formatPercent } from "@/lib/numberFormat";

// Horizontally scrolling strip of a fixed, curated index list (see
// main.py's INDEX_LIST) -- not user-editable, unlike Watchlist. A
// failed individual index (null price/change_pct, see the backend's
// per-item isolation) renders as a dash rather than being dropped, so
// the strip's width/order stays stable across reloads. "VIEW ALL"
// links to the dedicated /indices page, which splits this same data
// into Indian/Global tabs -- not on the bottom nav (see BottomNav.tsx's
// own comment on deliberately staying at 4 tabs).
export default function IndicesCarousel() {
  const [indices, setIndices] = useState<IndexQuote[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    fetch("/api/market/indices")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => {
        setIndices(data.indices);
        setFailed(false);
      })
      .catch(() => setFailed(true));
  }, [attempt]);

  if (failed) {
    return (
      <div className="mt-6">
        <p className="font-mono text-[10px] tracking-wide text-dim">INDICES</p>
        <LoadError what="indices" onRetry={() => setAttempt((a) => a + 1)} className="mt-2" />
      </div>
    );
  }
  if (indices === null) return <SectionSkeleton label="INDICES" rows={4} variant="row" />;
  if (indices.length === 0) return null;

  return (
    <div className="mt-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">INDICES</p>
        <Link href="/indices" className="-my-2 inline-block py-2 font-mono text-[10px] font-bold text-muted hover:text-accent">
          VIEW ALL &rarr;
        </Link>
      </div>
      <div className="mt-2 -mx-5 overflow-x-auto px-5">
        <div className="flex gap-2.5">
          {indices.map((idx) => (
            <div
              key={idx.ticker}
              className="flex min-w-[108px] flex-shrink-0 flex-col gap-1 rounded-lg border border-border bg-card px-3 py-2.5"
            >
              <span className="font-mono text-[10px] text-dim">{idx.name}</span>
              {idx.price !== null ? (
                <>
                  <span className="font-mono text-sm font-bold text-text">
                    {formatNumber(idx.price, 2)}
                  </span>
                  {idx.change_pct !== null && (
                    <span className={`font-mono text-[10px] ${idx.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                      {formatPercent(idx.change_pct, { signed: true })}
                    </span>
                  )}
                </>
              ) : (
                <span className="font-mono text-sm text-dim">--</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
