"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";

export default function Profile() {
  return (
    <AuthGate>
      {({ email, createdAt, jobsUsedToday, dailyLimit, logout }) => (
        <ProfilePage email={email} createdAt={createdAt} jobsUsedToday={jobsUsedToday} dailyLimit={dailyLimit} onLogout={logout} />
      )}
    </AuthGate>
  );
}

function initialsFromEmail(email: string | null): string {
  if (!email) return "AI";
  return email.split("@")[0].slice(0, 2).toUpperCase();
}

function memberSince(createdAt: number | null): string {
  if (!createdAt) return "unknown";
  return new Date(createdAt * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function ProfilePage({
  email,
  createdAt,
  jobsUsedToday,
  dailyLimit,
  onLogout,
}: {
  email: string | null;
  createdAt: number | null;
  jobsUsedToday: number | null;
  dailyLimit: number | null;
  onLogout: () => void;
}) {
  // Watchlist count isn't part of /v1/auth/me -- fetched separately,
  // best-effort (a failure here just means the tile stays blank, not a
  // broken page).
  const [watchlistCount, setWatchlistCount] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/watchlist")
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => setWatchlistCount(data.items.length))
      .catch(() => setWatchlistCount(null));
  }, []);

  const used = jobsUsedToday ?? 0;
  const limit = dailyLimit ?? 0;
  const usagePct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;

  return (
    <div className="min-h-screen bg-bg">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <Link href="/" className="font-mono text-xs font-bold text-muted hover:text-accent">
            &larr; FINSIGHT
          </Link>
        </div>

        <div className="mt-6 flex items-center gap-3.5">
          <span className="flex h-14 w-14 items-center justify-center rounded-full border border-border font-mono text-base font-bold text-text">
            {initialsFromEmail(email)}
          </span>
          <div className="min-w-0">
            <div className="truncate font-mono text-sm font-bold text-text">{email || "unknown"}</div>
            <div className="font-mono text-[11px] text-dim">Member since {memberSince(createdAt)}</div>
          </div>
        </div>

        <p className="mt-8 font-mono text-[10px] tracking-wide text-dim">USAGE TODAY</p>
        <div className="mt-2 flex gap-3">
          <div className="flex-1 rounded-lg border border-border bg-card px-3.5 py-3">
            <div className="font-mono text-[10px] text-muted">REPORTS RUN</div>
            <div className="mt-0.5 font-mono text-sm font-bold text-text">
              {used} / {limit}
            </div>
          </div>
          <div className="flex-1 rounded-lg border border-border bg-card px-3.5 py-3">
            <div className="font-mono text-[10px] text-muted">WATCHLIST</div>
            <div className="mt-0.5 font-mono text-sm font-bold text-text">
              {watchlistCount === null ? "--" : `${watchlistCount} ticker${watchlistCount === 1 ? "" : "s"}`}
            </div>
          </div>
        </div>

        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-card">
          <div className="h-full rounded-full bg-accent" style={{ width: `${usagePct}%` }} />
        </div>
        <p className="mt-1.5 font-mono text-[10px] text-dim">
          Resets on a rolling 24h window, not at a fixed time of day.
        </p>

        <button
          type="button"
          onClick={onLogout}
          className="mt-10 w-full rounded-lg border border-red-900/60 py-2.5 font-mono text-xs font-bold text-danger hover:bg-red-950/40"
        >
          SIGN OUT
        </button>
      </div>
    </div>
  );
}
