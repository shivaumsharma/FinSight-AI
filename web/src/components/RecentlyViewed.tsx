"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { recordVisit, type RecentlyViewedEntry } from "@/lib/useRecentlyViewed";

// Records the CURRENT ticker as a visit on mount, then shows every
// OTHER entry from the resulting list -- a horizontal carousel a user
// can jump back into a previous ticker from, entirely client-side
// (localStorage), no backend call at all.
export default function RecentlyViewed({ ticker, companyName }: { ticker: string; companyName: string | null }) {
  const [entries, setEntries] = useState<RecentlyViewedEntry[]>([]);

  useEffect(() => {
    setEntries(recordVisit(ticker, companyName));
  }, [ticker, companyName]);

  const others = entries.filter((e) => e.ticker !== ticker);
  if (others.length === 0) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">RECENTLY VIEWED</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {others.map((e) => (
          <Link
            key={e.ticker}
            href={`/stock/${e.ticker}`}
            className="flex w-[110px] shrink-0 flex-col gap-0.5 rounded-lg border border-border bg-card px-3 py-2.5 hover:border-accent"
          >
            <span className="truncate font-mono text-xs font-bold text-text">{e.ticker}</span>
            {e.companyName && <span className="truncate font-mono text-[9px] text-dim">{e.companyName}</span>}
          </Link>
        ))}
      </div>
    </div>
  );
}
