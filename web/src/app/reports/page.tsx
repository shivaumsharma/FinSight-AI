"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import RatingBadge from "@/components/RatingBadge";
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

// A fuller page than the home widget it's promoted from -- a real
// empty state (rather than just hiding, which is right for a small
// home-page section but not for a dedicated page someone navigated to
// on purpose) and a higher limit (50 vs the widget's 10), since
// browsing history is exactly what this page is for.
export default function ReportsPage() {
  return (
    <AuthGate>
      {() => (
        <div className="min-h-screen bg-bg pb-20">
          <div className="mx-auto max-w-2xl px-5 py-8">
            <h1 className="font-mono text-lg font-bold text-text">Reports</h1>
            <ReportsList />
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}

function ReportsList() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetch("/api/reports/recent?limit=50")
      .then((r) => (r.ok ? r.json() : { reports: [] }))
      .then((data) => setReports(data.reports))
      .catch(() => setReports([]));
  }, []);

  if (reports === null) {
    return <p className="mt-6 font-mono text-xs text-dim">loading...</p>;
  }

  if (reports.length === 0) {
    return (
      <div className="mt-6 rounded-lg border border-border-subtle bg-card/60 px-4 py-6 text-center">
        <p className="font-mono text-xs text-muted">No reports yet.</p>
        <p className="mt-1 font-mono text-[11px] text-dim">Run your first research query from Home.</p>
      </div>
    );
  }

  return (
    <div className="mt-4 flex flex-col gap-2">
      {reports.map((r) => (
        <button
          key={r.job_id}
          type="button"
          onClick={() => router.push(`/?job=${r.job_id}`)}
          className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-2.5 text-left hover:border-accent"
        >
          <div className="min-w-0">
            <div className="font-mono text-sm font-bold text-text">{r.ticker}</div>
            {r.company_name && <div className="truncate font-mono text-[10px] text-dim">{r.company_name}</div>}
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] text-dim">{relativeTime(r.started_at)}</span>
            <RatingBadge rating={r.rating} size="sm" />
          </div>
        </button>
      ))}
    </div>
  );
}
