"use client";

import { useEffect, useState } from "react";
import SectionSkeleton from "./SectionSkeleton";
import type { FxRate } from "@/lib/types";

// Live USD/INR exchange rate -- the one FX pair this app actually
// needs (see get_usd_conversion_rate's own comment: US via SEC, India
// via NSE are the only two markets this app covers). Small single-row
// card, not a chart -- this is a reference number for reading a mixed-
// currency Portfolio summary, not a trading tool.
export default function CurrencyTracker() {
  const [fx, setFx] = useState<FxRate | null>(null);

  useEffect(() => {
    fetch("/api/market/fx")
      .then((r) => (r.ok ? r.json() : null))
      .then(setFx)
      .catch(() => setFx(null));
  }, []);

  if (fx === null) return <SectionSkeleton label="CURRENCY" rows={1} />;
  if (fx.rate === null) return null;

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">CURRENCY</p>
      <div className="mt-2 flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5">
        <span className="font-mono text-sm font-bold text-text">1 USD = ₹{fx.rate.toFixed(2)}</span>
        {fx.change_pct !== null && (
          <span className={`font-mono text-[10px] font-bold ${fx.change_pct >= 0 ? "text-accent" : "text-danger"}`}>
            {fx.change_pct >= 0 ? "+" : ""}
            {fx.change_pct.toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  );
}
