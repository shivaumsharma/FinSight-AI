"use client";

import { useEffect, useState } from "react";
import { formatShortDate } from "@/lib/format";
import type { CorporateActionEvent, CorporateActionsFeedData } from "@/lib/types";

const EVENT_LABEL: Record<CorporateActionEvent["type"], string> = {
  earnings: "Earnings",
  ex_dividend: "Ex-Dividend",
};

// Chronological feed of upcoming earnings/ex-dividend dates across the
// watchlist and/or portfolio -- the per-row corporate-actions line
// already on Watchlist.tsx stays as-is; this is a dedicated calendar
// view of the SAME underlying data, not a replacement for it.
export default function CorporateActionsFeed() {
  const [data, setData] = useState<CorporateActionsFeedData | null>(null);
  const [scope, setScope] = useState<"all" | "portfolio">("all");

  useEffect(() => {
    setData(null);
    fetch(`/api/corporate-actions/feed?scope=${scope}`)
      .then((r) => (r.ok ? r.json() : { events: [], scope }))
      .then(setData)
      .catch(() => setData({ events: [], scope }));
  }, [scope]);

  return (
    <div>
      <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
        {(["all", "portfolio"] as const).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setScope(s)}
            className={`flex-1 rounded px-2.5 py-1.5 font-mono text-[10px] font-bold ${
              scope === s ? "bg-accent text-bg" : "text-muted hover:text-text"
            }`}
          >
            {s === "all" ? "ALL" : "PORTFOLIO"}
          </button>
        ))}
      </div>

      {data === null ? (
        <p className="mt-3 font-mono text-[11px] text-dim">Loading...</p>
      ) : data.events.length > 0 ? (
        <div className="mt-3 flex flex-col gap-2">
          {data.events.map((event, i) => (
            <div
              key={`${event.ticker}-${event.type}-${i}`}
              className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-bold text-text">{event.ticker}</span>
                  <span
                    className={`font-mono text-[10px] font-bold uppercase ${
                      event.type === "earnings" ? "text-accent" : "text-warn"
                    }`}
                  >
                    {EVENT_LABEL[event.type]}
                  </span>
                </div>
                <div className="truncate font-mono text-[10px] text-dim">{event.name}</div>
              </div>
              <span className="font-mono text-xs text-muted">{formatShortDate(event.date)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 font-mono text-[11px] text-dim">
          {scope === "portfolio"
            ? "No upcoming earnings or ex-dividend dates for your portfolio holdings."
            : "No upcoming earnings or ex-dividend dates for your tracked tickers."}
        </p>
      )}
    </div>
  );
}
