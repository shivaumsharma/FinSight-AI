"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import type { IndexQuote } from "@/lib/types";

// Full-size version of the home page's IndicesCarousel strip, split
// into Indian/Global tabs by region (see main.py's INDEX_LIST) --
// same /api/market/indices data, just a bigger card layout and a
// region filter instead of one unfiltered horizontal strip.
export default function IndicesPage() {
  const [indices, setIndices] = useState<IndexQuote[] | null>(null);
  const [tab, setTab] = useState<"global" | "india">("global");

  useEffect(() => {
    fetch("/api/market/indices")
      .then((r) => (r.ok ? r.json() : { indices: [] }))
      .then((data) => setIndices(data.indices))
      .catch(() => setIndices([]));
  }, []);

  const shown = (indices || []).filter((idx) => idx.region === tab);

  return (
    <AuthGate>
      {() => (
        <div className="min-h-screen bg-bg pb-20">
          <div className="mx-auto max-w-2xl px-5 py-8">
            <Link href="/" className="font-mono text-xs font-bold text-muted hover:text-accent">
              &larr; HOME
            </Link>
            <h1 className="mt-3 font-mono text-lg font-bold text-text">Indices</h1>

            <div className="mt-4 flex gap-1 rounded-lg border border-border bg-card p-0.5">
              {(["global", "india"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  className={`flex-1 rounded px-2.5 py-1.5 font-mono text-[10px] font-bold ${
                    tab === t ? "bg-accent text-bg" : "text-muted hover:text-text"
                  }`}
                >
                  {t === "global" ? "GLOBAL" : "INDIAN"}
                </button>
              ))}
            </div>

            {indices !== null && (
              <div className="mt-3 flex flex-col gap-2">
                {shown.map((idx) => (
                  <div
                    key={idx.ticker}
                    className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3"
                  >
                    <div>
                      <div className="font-mono text-sm font-bold text-text">{idx.name}</div>
                      <div className="font-mono text-[10px] text-dim">{idx.ticker}</div>
                    </div>
                    {idx.price !== null ? (
                      <div className="text-right">
                        <div className="font-mono text-sm text-text">
                          {idx.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </div>
                        {idx.change_pct !== null && (
                          <div className={`font-mono text-xs font-bold ${idx.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
                            {idx.change_pct >= 0 ? "+" : ""}
                            {idx.change_pct.toFixed(2)}%
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="font-mono text-xs text-dim">--</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}
