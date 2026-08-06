"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import { usePushNotifications } from "@/lib/usePushNotifications";
import type { WatchlistItem } from "@/lib/types";

export default function Profile() {
  return (
    <AuthGate>
      {({
        email, createdAt, jobsUsedToday, dailyLimit, totalReports, sessionExpiresAt,
        riskTolerance, setRiskTolerance, logout, deleteAccount, deleteError,
      }) => (
        <ProfilePage
          email={email}
          createdAt={createdAt}
          jobsUsedToday={jobsUsedToday}
          dailyLimit={dailyLimit}
          totalReports={totalReports}
          sessionExpiresAt={sessionExpiresAt}
          riskTolerance={riskTolerance}
          onSetRiskTolerance={setRiskTolerance}
          onLogout={logout}
          onDeleteAccount={deleteAccount}
          deleteError={deleteError}
        />
      )}
    </AuthGate>
  );
}

function initialsFromEmail(email: string | null): string {
  if (!email) return "AI";
  return email.split("@")[0].slice(0, 2).toUpperCase();
}

function formatDate(unixSeconds: number | null): string {
  if (!unixSeconds) return "unknown";
  return new Date(unixSeconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function formatDateTime(unixSeconds: number | null): string {
  if (!unixSeconds) return "unknown";
  return new Date(unixSeconds * 1000).toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

// Real toggle, not decorative -- reuses the same usePushNotifications
// hook page.tsx's header toggle already calls, just with a
// switch-style visual instead of a compact text button (matches this
// page's row-based Preferences layout better than the header's
// space-constrained one).
function NotificationRow() {
  const { status, subscribe, unsubscribe } = usePushNotifications();

  if (status === "unsupported" || status === "checking") return null;

  const subscribed = status === "subscribed";
  const denied = status === "denied";

  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div>
        <div className="font-mono text-xs text-text">Push notifications</div>
        {denied && (
          <div className="mt-0.5 font-mono text-[10px] text-dim">Blocked in your browser settings</div>
        )}
      </div>
      <button
        type="button"
        disabled={denied}
        onClick={subscribed ? unsubscribe : subscribe}
        title={denied ? "Blocked in your browser settings for this site" : undefined}
        className={`relative h-6 w-11 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
          subscribed ? "bg-accent" : "bg-border"
        }`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-bg transition-transform ${
            subscribed ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </button>
    </div>
  );
}

const RISK_LEVELS = ["Conservative", "Moderate", "Aggressive"] as const;
const RISK_COLORS: Record<string, string> = {
  Conservative: "text-accent border-accent",
  Moderate: "text-warn border-warn",
  Aggressive: "text-danger border-danger",
};

// Real, persisted preference (app/api/db.py's users.risk_tolerance) --
// clicking cycles Conservative -> Moderate -> Aggressive -> back, same
// interaction as a settings toggle in most mobile apps when there's no
// room for a dropdown. NOT yet applied to the research pipeline's own
// WACC/discount assumptions -- saved and reflected back honestly, not
// wired into report generation in this pass.
function RiskToleranceRow({
  riskTolerance,
  onChange,
}: {
  riskTolerance: string | null;
  onChange: (level: string) => void;
}) {
  const current = riskTolerance || "Moderate";

  function cycle() {
    const next = RISK_LEVELS[(RISK_LEVELS.indexOf(current as (typeof RISK_LEVELS)[number]) + 1) % RISK_LEVELS.length];
    onChange(next);
  }

  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text">Risk tolerance</div>
      <button
        type="button"
        onClick={cycle}
        title="Click to cycle"
        className={`rounded border px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wide ${RISK_COLORS[current]}`}
      >
        {current}
      </button>
    </div>
  );
}

// Static, informational rows -- not editable toggles, because there's
// no real alternate backend behavior for them to switch between: this
// app only ever runs one valuation methodology (WACC/FCFF/DCF), and
// quotes are already live (get_quote's 45s cache is a performance
// detail, not a "refresh interval" a user actually controls). Showing
// them as plain text describes what the app already does, rather than
// implying a setting that doesn't exist.
function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text">{label}</div>
      <div className="font-mono text-[11px] text-muted">{value}</div>
    </div>
  );
}

function ConnectedSourceRow({ label, linked }: { label: string; linked: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text">{label}</div>
      <div className="flex items-center gap-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${linked ? "bg-accent" : "bg-dim"}`} />
        <span className={`font-mono text-[10px] font-bold uppercase tracking-wide ${linked ? "text-accent" : "text-dim"}`}>
          {linked ? "Linked" : "Not connected"}
        </span>
      </div>
    </div>
  );
}

function DeleteAccountSection({
  onDeleteAccount,
  deleteError,
}: {
  onDeleteAccount: (password: string) => Promise<boolean>;
  deleteError: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const [password, setPassword] = useState("");
  const [deleting, setDeleting] = useState(false);

  async function handleConfirm(e: React.FormEvent) {
    e.preventDefault();
    if (!password || deleting) return;
    setDeleting(true);
    const ok = await onDeleteAccount(password);
    setDeleting(false);
    // On success AuthGate's own status flips to "unauthenticated" and
    // this whole page unmounts in favor of the login screen -- no
    // manual redirect needed. On failure, stay expanded so deleteError
    // (surfaced below) is visible next to the form that produced it.
    if (!ok) setPassword("");
  }

  return (
    <div className="mt-10 rounded-lg border border-red-900/60 p-4">
      <p className="font-mono text-[10px] tracking-wide text-danger">DANGER ZONE</p>
      <p className="mt-1.5 text-xs leading-relaxed text-dim">
        Permanently deletes your account and every report, watchlist item, and notification
        subscription tied to it. This cannot be undone.
      </p>

      {!expanded ? (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="mt-3 rounded-lg border border-red-900/60 px-4 py-2 font-mono text-xs font-bold text-danger hover:bg-red-950/40"
        >
          DELETE ACCOUNT
        </button>
      ) : (
        <form onSubmit={handleConfirm} className="mt-3 space-y-2">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="confirm your password"
            autoComplete="current-password"
            autoFocus
            className="w-full rounded-lg border border-red-900/60 bg-card px-3.5 py-2.5 font-mono text-sm text-text placeholder:text-muted focus:outline-none"
          />
          {deleteError && <p className="font-mono text-[10px] text-danger">{deleteError}</p>}
          <div className="flex gap-2">
            <button
              type="submit"
              disabled={!password || deleting}
              className="flex-1 rounded-lg bg-danger py-2 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
            >
              {deleting ? "..." : "CONFIRM DELETE"}
            </button>
            <button
              type="button"
              onClick={() => {
                setExpanded(false);
                setPassword("");
              }}
              disabled={deleting}
              className="rounded-lg border border-border px-4 py-2 font-mono text-xs font-bold text-muted hover:text-text"
            >
              CANCEL
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

function ProfilePage({
  email,
  createdAt,
  jobsUsedToday,
  dailyLimit,
  totalReports,
  sessionExpiresAt,
  riskTolerance,
  onSetRiskTolerance,
  onLogout,
  onDeleteAccount,
  deleteError,
}: {
  email: string | null;
  createdAt: number | null;
  jobsUsedToday: number | null;
  dailyLimit: number | null;
  totalReports: number | null;
  sessionExpiresAt: number | null;
  riskTolerance: string | null;
  onSetRiskTolerance: (level: string) => Promise<void>;
  onLogout: () => void;
  onDeleteAccount: (password: string) => Promise<boolean>;
  deleteError: string | null;
}) {
  // Watchlist items aren't part of /v1/auth/me -- fetched separately,
  // best-effort (a failure here just means the tile stays blank, not a
  // broken page).
  const [watchlist, setWatchlist] = useState<WatchlistItem[] | null>(null);

  useEffect(() => {
    fetch("/api/watchlist")
      .then((r) => (r.ok ? r.json() : { items: [] }))
      .then((data) => setWatchlist(data.items))
      .catch(() => setWatchlist(null));
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
            <div className="font-mono text-[11px] text-dim">Member since {formatDate(createdAt)}</div>
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
              {watchlist === null ? "--" : `${watchlist.length} ticker${watchlist.length === 1 ? "" : "s"}`}
            </div>
          </div>
        </div>

        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-card">
          <div className="h-full rounded-full bg-accent" style={{ width: `${usagePct}%` }} />
        </div>
        <p className="mt-1.5 font-mono text-[10px] text-dim">
          Resets on a rolling 24h window, not at a fixed time of day
          {totalReports !== null && ` · ${totalReports} report${totalReports === 1 ? "" : "s"} all-time`}.
        </p>

        {watchlist && watchlist.length > 0 && (
          <p className="mt-2 font-mono text-[11px] text-muted">
            {watchlist.map((w) => w.ticker).join(", ")}
          </p>
        )}

        <p className="mt-8 font-mono text-[10px] tracking-wide text-dim">PREFERENCES</p>
        <div className="mt-2 space-y-2">
          <RiskToleranceRow riskTolerance={riskTolerance} onChange={onSetRiskTolerance} />
          <NotificationRow />
          <InfoRow label="Default valuation model" value="DCF · FCFF" />
          <InfoRow label="Data refresh" value="REAL-TIME" />
        </div>

        <p className="mt-8 font-mono text-[10px] tracking-wide text-dim">CONNECTED SOURCES</p>
        <div className="mt-2 space-y-2">
          <ConnectedSourceRow label="SEC EDGAR" linked />
          <ConnectedSourceRow label="yfinance market data" linked />
          <ConnectedSourceRow label="Brokerage sync" linked={false} />
        </div>

        <p className="mt-8 font-mono text-[10px] tracking-wide text-dim">SESSION</p>
        <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-3">
          <div className="font-mono text-xs text-text">Signed in until {formatDateTime(sessionExpiresAt)}</div>
          <div className="mt-0.5 font-mono text-[10px] text-dim">Logging in again resets this to 30 days out</div>
        </div>

        <button
          type="button"
          onClick={onLogout}
          className="mt-8 w-full rounded-lg border border-red-900/60 py-2.5 font-mono text-xs font-bold text-danger hover:bg-red-950/40"
        >
          SIGN OUT
        </button>

        <DeleteAccountSection onDeleteAccount={onDeleteAccount} deleteError={deleteError} />
      </div>
    </div>
  );
}
