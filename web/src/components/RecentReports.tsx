"use client";

import { useEffect, useState } from "react";
import RatingBadge from "./RatingBadge";
import type { ReportSummary } from "@/lib/types";

function relativeTime(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const minutes = Math.round(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

// Renders nothing (not even a header) when the list is empty -- for a
// brand-new account with no completed research yet, PipelinePreview
// already fills the idle-state space; an empty "Recent Reports" shell
// here would just be more clutter, not less.
export default function RecentReports({ onSelectReport }: { onSelectReport: (jobId: string) => void }) {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);

  useEffect(() => {
    fetch("/api/reports/recent")
      .then((r) => (r.ok ? r.json() : { reports: [] }))
      .then((data) => setReports(data.reports))
      .catch(() => setReports([]));
  }, []);

  if (!reports || reports.length === 0) return null;

  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">RECENT REPORTS</p>
      <div className="mt-2 flex flex-col gap-2">
        {reports.map((r) => (
          <button
            key={r.job_id}
            type="button"
            onClick={() => onSelectReport(r.job_id)}
            className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5 text-left hover:border-accent"
          >
            <div className="min-w-0">
              <div className="font-mono text-sm font-bold text-text">{r.ticker}</div>
              {r.company_name && (
                <div className="truncate font-mono text-[10px] text-dim">{r.company_name}</div>
              )}
            </div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] text-dim">{relativeTime(r.started_at)}</span>
              <RatingBadge rating={r.rating} size="sm" />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
