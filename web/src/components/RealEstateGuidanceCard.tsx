"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface RealEstateGuidance {
  target_allocation_low_pct: number;
  target_allocation_high_pct: number;
  example_reit_tickers: string[];
  disclaimer: string;
}

// Self-contained, like PortfolioFitCard/SimilarStocks -- fetches its
// own data and renders nothing if the user never opted into real
// estate during onboarding (the backend returns {} in that case, see
// GET /v1/auth/real-estate-guidance). Allocation-level guidance only,
// never a specific property recommendation -- no free data source
// exists for property-level valuation (see
// app/reasoning/real_estate_guidance.py's own docstring).
export default function RealEstateGuidanceCard() {
  const [guidance, setGuidance] = useState<RealEstateGuidance | null>(null);

  useEffect(() => {
    fetch("/api/auth/real-estate-guidance")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.target_allocation_low_pct !== undefined) setGuidance(data);
      })
      .catch(() => {});
  }, []);

  if (!guidance) return null;

  return (
    <div className="rounded-lg border border-border bg-card px-3.5 py-3">
      <p className="font-mono text-[10px] tracking-wide text-dim">REAL ESTATE</p>
      <p className="mt-1.5 text-xs text-text/90">
        A {guidance.target_allocation_low_pct}&ndash;{guidance.target_allocation_high_pct}% allocation is a
        reasonable diversification target for your risk profile. Examples to look up (not a recommendation to buy
        these specifically):{" "}
        {guidance.example_reit_tickers.map((t, i) => (
          <span key={t}>
            {i > 0 && ", "}
            <Link href={`/stock/${t}`} className="text-accent hover:underline">
              {t}
            </Link>
          </span>
        ))}
        .
      </p>
      <p className="mt-1.5 font-mono text-[10px] text-dim">{guidance.disclaimer}</p>
    </div>
  );
}
