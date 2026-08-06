"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { usePushNotifications } from "@/lib/usePushNotifications";
import { relativeTime } from "@/lib/format";
import RatingBadge from "./RatingBadge";
import type { ReportSummary } from "@/lib/types";

const LAST_SEEN_KEY = "finsight:notifications:lastSeenAt";

// Single place for both halves of "notifications" that the earlier
// header split left ambiguous: a real history of what happened (past
// completed reports, sourced from the same list /reports already
// shows -- no new backend endpoint) and the push-permission control
// that decides whether those completions also ping you outside the
// tab. Opening the panel is what marks it read, same convention as a
// bell icon anywhere else.
export default function NotificationsPanel() {
  const [open, setOpen] = useState(false);
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [lastSeenAt, setLastSeenAt] = useState(0);
  const router = useRouter();
  const { status, subscribe, unsubscribe } = usePushNotifications();

  useEffect(() => {
    setLastSeenAt(Number(localStorage.getItem(LAST_SEEN_KEY) || 0));
    fetch("/api/reports/recent?limit=10")
      .then((r) => (r.ok ? r.json() : { reports: [] }))
      .then((data) => setReports(data.reports))
      .catch(() => setReports([]));
  }, []);

  const unreadCount = useMemo(() => {
    if (!reports) return 0;
    return reports.filter((r) => r.started_at > lastSeenAt).length;
  }, [reports, lastSeenAt]);

  function toggleOpen() {
    const next = !open;
    setOpen(next);
    if (next) {
      const now = Date.now() / 1000;
      localStorage.setItem(LAST_SEEN_KEY, String(now));
      setLastSeenAt(now);
    }
  }

  function goToReport(jobId: string) {
    router.push(`/?job=${jobId}`);
    setOpen(false);
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={toggleOpen}
        aria-label="Notifications"
        className="relative font-mono text-xs text-muted hover:text-accent"
      >
        &#128276;
        {unreadCount > 0 && (
          <span className="absolute -right-1.5 -top-1.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-accent font-mono text-[8px] font-bold text-bg">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-6 z-20 w-72 rounded-lg border border-border bg-card shadow-lg">
          <div className="flex items-center justify-between border-b border-border-subtle px-3.5 py-2.5">
            <p className="font-mono text-[10px] tracking-wide text-dim">NOTIFICATIONS</p>
            {status === "subscribed" ? (
              <button type="button" onClick={unsubscribe} className="font-mono text-[10px] text-accent hover:text-muted">
                PUSH ON
              </button>
            ) : status === "unsubscribed" ? (
              <button type="button" onClick={subscribe} className="font-mono text-[10px] text-muted hover:text-accent">
                ENABLE PUSH
              </button>
            ) : status === "denied" ? (
              <span className="font-mono text-[10px] text-dim">PUSH BLOCKED</span>
            ) : null}
          </div>

          <div className="max-h-80 overflow-y-auto">
            {reports === null && <p className="px-3.5 py-4 font-mono text-[11px] text-dim">loading...</p>}

            {reports !== null && reports.length === 0 && (
              <p className="px-3.5 py-4 text-center font-mono text-[11px] text-dim">
                Nothing yet -- completed reports show up here.
              </p>
            )}

            {reports?.map((r) => (
              <button
                key={r.job_id}
                type="button"
                onClick={() => goToReport(r.job_id)}
                className="flex w-full items-center justify-between border-b border-border-subtle px-3.5 py-2.5 text-left last:border-b-0 hover:bg-bg/60"
              >
                <div className="min-w-0">
                  <div className="font-mono text-xs text-text">
                    Your <span className="font-bold">{r.ticker}</span> report is ready
                  </div>
                  <div className="font-mono text-[10px] text-dim">{relativeTime(r.started_at)}</div>
                </div>
                <RatingBadge rating={r.rating} size="sm" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
