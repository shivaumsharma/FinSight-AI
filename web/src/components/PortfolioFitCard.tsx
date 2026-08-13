"use client";

import { useEffect, useState } from "react";
import type { PortfolioFit } from "@/lib/types";

// Deterministic, LLM-free (see app/reasoning/portfolio_fit.py) --
// whether this ticker's sector overlaps with the user's existing
// holdings, or would diversify them into something new. Renders
// nothing on error/empty rather than a broken card -- same convention
// as SimilarStocks.tsx.
export default function PortfolioFitCard({ ticker }: { ticker: string }) {
  const [fit, setFit] = useState<PortfolioFit | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setFit(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/portfolio-fit`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: PortfolioFit) => setFit(data))
      .catch(() => setError(true));
  }, [ticker]);

  if (error) return null;
  if (!fit) {
    return (
      <div className="mt-4">
        <div className="h-[52px] animate-pulse rounded-lg border border-border bg-card/60" />
      </div>
    );
  }

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">PORTFOLIO FIT</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <p className="text-xs text-text/90">{fit.summary}</p>
      </div>
    </div>
  );
}
