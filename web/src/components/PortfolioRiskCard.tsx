"use client";

import { useEffect, useState } from "react";
import { PORTFOLIO_UPDATED_EVENT } from "@/lib/portfolioEvents";
import type { PortfolioRiskOverlay } from "@/lib/types";

// Volatility-targeting risk view of the user's holdings (app/analysis/vol_overlay.py). It describes how much RISK the
// portfolio is carrying right now, not what the market will do: on the S&P 500 (2007-2025) the same rule cut the worst
// drawdown by 23-39% but added no meaningful return in the 2016-2025 bull market -- the card always says so.
// Renders nothing when the fetch fails or there is no portfolio to describe, same convention as PortfolioFitCard.tsx.
export default function PortfolioRiskCard() {
  const [overlay, setOverlay] = useState<PortfolioRiskOverlay | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const load = () => {
      setError(false);
      fetch("/api/portfolio/risk-overlay")
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((data: PortfolioRiskOverlay) => setOverlay(data))
        .catch(() => setError(true));
    };
    load();
    window.addEventListener(PORTFOLIO_UPDATED_EVENT, load);
    return () => window.removeEventListener(PORTFOLIO_UPDATED_EVENT, load);
  }, []);

  if (error) return null;
  if (!overlay) {
    return <div className="mt-2 h-[72px] animate-pulse rounded-lg border border-border-subtle bg-card/60" />;
  }
  if (!overlay.available) {
    return (
      <div className="mt-2 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-2">
        <p className="font-mono text-[10px] tracking-wide text-dim">RISK VIEW</p>
        <p className="mt-1 text-xs text-muted">{overlay.reason ?? "Not available right now."}</p>
      </div>
    );
  }

  const exposure = overlay.suggested_exposure_pct ?? 100;
  const reduced = exposure < 99.5;
  return (
    <div className="mt-2 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-2.5">
      <div className="flex items-center justify-between font-mono text-[10px]">
        <span className="tracking-wide text-dim">RISK VIEW (LAST {overlay.window_days} DAYS)</span>
        <span className="font-bold text-text">
          {overlay.realised_volatility_pct?.toFixed(1)}% volatility · target {overlay.target_volatility_pct?.toFixed(0)}%
        </span>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded bg-border-subtle" aria-hidden="true">
        <div className={`h-full ${reduced ? "bg-danger" : "bg-accent"}`} style={{ width: `${Math.min(100, Math.max(0, exposure))}%` }} />
      </div>
      <p className="mt-1.5 text-xs text-text/90">
        {reduced
          ? `A volatility-targeting rule would hold about ${exposure.toFixed(0)}% invested and ${overlay.suggested_cash_pct?.toFixed(0)}% in cash right now.`
          : "Recent volatility is at or below the target, so the rule would stay fully invested."}
        {" "}Worst dip in this window: {overlay.max_drawdown_in_window_pct?.toFixed(1)}%.
      </p>
      <p className="mt-1.5 text-[11px] text-muted">{overlay.evidence?.takeaway}</p>
      <p className="mt-1 text-[10px] text-dim">{overlay.disclaimer}</p>
      {overlay.excluded_tickers && overlay.excluded_tickers.length > 0 && (
        <p className="mt-1 text-[10px] text-dim">Left out (non-USD): {overlay.excluded_tickers.join(", ")}</p>
      )}
    </div>
  );
}
