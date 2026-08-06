"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import { usePushNotifications } from "@/lib/usePushNotifications";
import type { WatchlistItem } from "@/lib/types";

export default function Profile() {
  return (
    <AuthGate>
      {({
        email, createdAt, jobsUsedToday, dailyLimit, totalReports, sessionExpiresAt,
        riskTolerance, setRiskTolerance, resetAt, waitlistFeatures, joinWaitlist,
        displayName, setDisplayName,
        logout, deleteAccount, deleteError,
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
          resetAt={resetAt}
          waitlistFeatures={waitlistFeatures}
          onJoinWaitlist={joinWaitlist}
          displayName={displayName}
          onSetDisplayName={setDisplayName}
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

// Same heuristic page.tsx's greeting falls back to when no display_name
// is set -- shown here as the input's placeholder so editing feels like
// starting from what's already displayed, not a blank field.
function fallbackNameFromEmail(email: string | null): string {
  if (!email) return "there";
  const local = email.split("@")[0];
  const first = local.split(/[._-]/)[0];
  return first.charAt(0).toUpperCase() + first.slice(1);
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

// A static "as of page load" countdown, not a live-ticking one -- this
// value doesn't need per-second accuracy (it's "resets in about 18
// hours", not a race timer), so no setInterval/re-render machinery for
// something a user glances at once.
function formatCountdown(resetAt: number | null): string | null {
  if (!resetAt) return null;
  const diffMs = resetAt * 1000 - Date.now();
  if (diffMs <= 0) return null;
  const hours = Math.floor(diffMs / 3_600_000);
  const minutes = Math.floor((diffMs % 3_600_000) / 60_000);
  return `${hours}h ${minutes}m`;
}

// Minimal inline SVGs, not an icon library -- consistent with this
// project's stated minimal-dependency stance elsewhere (stdlib
// sqlite3 instead of an ORM, a hand-rolled service worker instead of
// next-pwa). Each is ~12px, single-color, matches the mono/terminal
// aesthetic instead of importing a mismatched icon set.
function UsageIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 14V9M8 14V4M14 14V7" strokeLinecap="round" />
    </svg>
  );
}
function PreferencesIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 4h6M12 4h2M2 8h1M5 8h9M2 12h7M11 12h3" strokeLinecap="round" />
      <circle cx="9" cy="4" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="3" cy="8" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="9" cy="12" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}
function SourcesIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path
        d="M6.5 9.5L9.5 6.5M5 11L3.5 12.5a2.1 2.1 0 01-3-3L2 8M11 5l1.5-1.5a2.1 2.1 0 013 3L14 8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function SessionIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <path d="M8 5v3l2 2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SectionHeader({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="mt-8 flex items-center gap-1.5 text-dim">
      {icon}
      <p className="font-mono text-[10px] tracking-wide">{label}</p>
    </div>
  );
}

// Real toggle, not decorative -- reuses the same usePushNotifications
// hook page.tsx's header toggle already calls, just with a
// switch-style visual instead of a compact text button (matches this
// page's row-based Preferences layout better than the header's
// space-constrained one). Copy is deliberately specific about the one
// real trigger this fires today (a completed research job) rather
// than implying categories (earnings alerts, price alerts) that don't
// exist as backend event types yet -- a toggle that claims to control
// more than it does is worse than one that's honest about doing less.
function NotificationRow() {
  const { status, subscribe, unsubscribe } = usePushNotifications();

  if (status === "unsupported" || status === "checking") return null;

  const subscribed = status === "subscribed";
  const denied = status === "denied";

  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div>
        <div className="font-mono text-xs text-text">Research complete notifications</div>
        <div className="mt-0.5 font-mono text-[10px] text-dim">
          {denied ? "Blocked in your browser settings" : "Notifies you when a report finishes, even if you've closed the tab"}
        </div>
      </div>
      <button
        type="button"
        disabled={denied}
        onClick={subscribed ? unsubscribe : subscribe}
        title={denied ? "Blocked in your browser settings for this site" : undefined}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
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
// room for a dropdown. The title tooltip is explicit that this is
// saved-but-not-yet-applied -- better than letting a user assume their
// reports already changed because of it.
// Fixes the greeting header showing a typo-looking guess (e.g.
// "sshivaum@..." -> "Sshivaum") for anyone whose email doesn't cleanly
// split into a real name -- a real, persisted, user-editable field
// instead of a smarter guess, since no heuristic can reliably recover
// an actual name from an arbitrary email local-part.
function DisplayNameRow({
  displayName,
  emailFallback,
  onChange,
}: {
  displayName: string | null;
  emailFallback: string;
  onChange: (name: string | null) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(displayName || "");

  function startEditing() {
    setValue(displayName || "");
    setEditing(true);
  }

  function save() {
    onChange(value.trim() || null);
    setEditing(false);
  }

  if (editing) {
    return (
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <div className="font-mono text-xs text-text">Display name</div>
        <div className="mt-2 flex gap-2">
          <input
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setEditing(false);
            }}
            maxLength={40}
            autoFocus
            placeholder={emailFallback}
            className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-1.5 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent"
          />
          <button type="button" onClick={save} className="rounded-lg bg-accent px-3 py-1.5 font-mono text-[10px] font-bold text-bg">
            SAVE
          </button>
          <button type="button" onClick={() => setEditing(false)} className="font-mono text-[10px] text-dim hover:text-muted">
            CANCEL
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text">Display name</div>
      <button type="button" onClick={startEditing} className="font-mono text-[11px] text-muted hover:text-accent">
        {displayName || emailFallback} &middot; edit
      </button>
    </div>
  );
}

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
      <div
        className="font-mono text-xs text-text"
        title="Saved to your profile. Not yet applied to report generation -- DCF assumptions and ratings are the same regardless of this setting today."
      >
        Risk tolerance
      </div>
      <button
        type="button"
        onClick={cycle}
        title="Click to cycle · saved but not yet applied to reports"
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
// implying a setting that doesn't exist. `hint` powers a native title
// tooltip -- explains the jargon (DCF/FCFF) without a new UI pattern.
function InfoRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text" title={hint}>
        {label}
      </div>
      <div className="font-mono text-[11px] text-muted">{value}</div>
    </div>
  );
}

// Brokerage Sync gets a real action instead of a dead "Not Connected"
// state -- clicking joins a genuine waitlist (app/api/db.py's
// `waitlist` table), not a fake state change with nowhere for the
// click to go.
function WaitlistRow({
  label,
  feature,
  joined,
  onJoin,
}: {
  label: string;
  feature: string;
  joined: boolean;
  onJoin: (feature: string) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="font-mono text-xs text-text">{label}</div>
      {joined ? (
        <span className="font-mono text-[10px] font-bold uppercase tracking-wide text-accent">On the list</span>
      ) : (
        <button
          type="button"
          onClick={() => onJoin(feature)}
          className="rounded border border-border px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wide text-muted hover:border-accent hover:text-accent"
        >
          Join Waitlist
        </button>
      )}
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
  resetAt,
  waitlistFeatures,
  onJoinWaitlist,
  displayName,
  onSetDisplayName,
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
  resetAt: number | null;
  waitlistFeatures: string[];
  onJoinWaitlist: (feature: string) => void;
  displayName: string | null;
  onSetDisplayName: (name: string | null) => Promise<void>;
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
  const remaining = Math.max(0, limit - used);
  const usagePct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  const countdown = formatCountdown(resetAt);

  return (
    <div className="min-h-screen bg-bg pb-20">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <Link href="/" className="font-mono text-xs font-bold text-muted hover:text-accent">
            &larr; FINSIGHT
          </Link>
          {/* Deliberately a small text link here, not the big red
              full-width button this used to be -- Sign Out is not the
              action this page should be loudest about; the Danger
              Zone below stays the visually strongest destructive
              action, which is the one that actually deserves it. */}
          <button type="button" onClick={onLogout} className="font-mono text-[10px] text-dim hover:text-muted">
            Sign out
          </button>
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

        <SectionHeader icon={<UsageIcon />} label="USAGE TODAY" />
        <div className="mt-2 flex gap-3">
          <div className="flex-1 rounded-lg border border-border bg-card px-3.5 py-3">
            <div className="font-mono text-[10px] text-muted">REPORTS REMAINING</div>
            <div className="mt-0.5 font-mono text-lg font-bold text-text">{remaining}</div>
            <div className="mt-0.5 font-mono text-[10px] text-dim">
              {countdown ? `Resets in ${countdown}` : "of " + limit + " today"}
            </div>
          </div>
          <div className="flex-1 rounded-lg border border-border bg-card px-3.5 py-3">
            <div className="font-mono text-[10px] text-muted">WATCHLIST</div>
            <div className="mt-0.5 font-mono text-lg font-bold text-text">{watchlist === null ? "--" : watchlist.length}</div>
            <div className="mt-0.5 font-mono text-[10px] text-dim">
              {watchlist === null ? "loading" : watchlist.length === 1 ? "ticker" : "tickers"}
            </div>
          </div>
        </div>

        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-card">
          <div className="h-full rounded-full bg-accent" style={{ width: `${usagePct}%` }} />
        </div>
        <p className="mt-1.5 font-mono text-[10px] text-dim">
          Rolling 24h window, not a fixed reset time
          {totalReports !== null && ` · ${totalReports} report${totalReports === 1 ? "" : "s"} all-time`}.
        </p>

        {watchlist && watchlist.length > 0 && (
          <p className="mt-2 font-mono text-[11px] text-muted">{watchlist.map((w) => w.ticker).join(", ")}</p>
        )}

        <SectionHeader icon={<PreferencesIcon />} label="PREFERENCES" />
        <div className="mt-2 space-y-2">
          <DisplayNameRow displayName={displayName} emailFallback={fallbackNameFromEmail(email)} onChange={onSetDisplayName} />
          <RiskToleranceRow riskTolerance={riskTolerance} onChange={onSetRiskTolerance} />
          <NotificationRow />
          <InfoRow
            label="Valuation methodology"
            value="DCF · FCFF"
            hint="Discounted Cash Flow valuation using Free Cash Flow to Firm -- the only valuation model this app runs today, so it isn't a switchable setting."
          />
          <InfoRow
            label="Quote freshness"
            value="~45s"
            hint="Watchlist prices are cached for up to 45 seconds to avoid hammering the market data provider on every page load. Report prices are always fetched fresh, never cached."
          />
        </div>

        <SectionHeader icon={<SourcesIcon />} label="CONNECTED SOURCES" />
        <div className="mt-2 space-y-2">
          <ConnectedSourceRow label="SEC EDGAR" linked />
          <ConnectedSourceRow label="yfinance market data" linked />
          <WaitlistRow
            label="Brokerage sync"
            feature="brokerage_sync"
            joined={waitlistFeatures.includes("brokerage_sync")}
            onJoin={onJoinWaitlist}
          />
        </div>

        <SectionHeader icon={<SessionIcon />} label="SESSION" />
        <div className="mt-2 rounded-lg border border-border bg-card px-3.5 py-3">
          <div className="font-mono text-xs text-text">Signed in until {formatDateTime(sessionExpiresAt)}</div>
          <div className="mt-0.5 font-mono text-[10px] text-dim">Logging in again resets this to 30 days out</div>
        </div>

        <DeleteAccountSection onDeleteAccount={onDeleteAccount} deleteError={deleteError} />
      </div>
      <BottomNav />
    </div>
  );
}
