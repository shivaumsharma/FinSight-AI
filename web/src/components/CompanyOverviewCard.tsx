"use client";

import { useState } from "react";
import type { StockOverview } from "@/lib/types";

const TRUNCATE_LENGTH = 220;

// Real, already-written yfinance business description (same
// longBusinessSummary field app/data/market_data.py's
// get_company_info() already surfaces for the research report's
// Company Overview section) -- not an LLM-generated summary, so this
// card never runs a model call just to render some prose.
export default function CompanyOverviewCard({ overview }: { overview: StockOverview }) {
  const [expanded, setExpanded] = useState(false);
  const summary = overview.business_summary;

  if (!summary) return null;

  const isLong = summary.length > TRUNCATE_LENGTH;
  const shown = expanded || !isLong ? summary : `${summary.slice(0, TRUNCATE_LENGTH).trimEnd()}...`;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">COMPANY OVERVIEW</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        {(overview.industry || overview.sector) && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {overview.sector && (
              <span className="rounded border border-border-subtle px-2 py-0.5 font-mono text-[10px] text-muted">{overview.sector}</span>
            )}
            {overview.industry && (
              <span className="rounded border border-border-subtle px-2 py-0.5 font-mono text-[10px] text-muted">{overview.industry}</span>
            )}
          </div>
        )}
        <p className="text-sm leading-relaxed text-text/90">{shown}</p>
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="mt-2 font-mono text-[10px] font-bold text-accent hover:underline"
          >
            {expanded ? "SHOW LESS" : "READ MORE"}
          </button>
        )}
      </div>
    </div>
  );
}
