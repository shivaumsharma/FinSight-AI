"use client";

import { useState } from "react";
import { useAuth } from "@/lib/useAuth";

// Gates children behind a real logged-in session -- unauthenticated
// visitors see a login/signup form instead of the research UI. Scope
// is deliberately just signup/login/logout; no password reset, no
// account settings (see the project plan doc's own scope note).
interface AuthedProps {
  userId: string;
  email: string | null;
  createdAt: number | null;
  jobsUsedToday: number | null;
  dailyLimit: number | null;
  totalReports: number | null;
  sessionExpiresAt: number | null;
  riskTolerance: string | null;
  setRiskTolerance: (level: string) => Promise<void>;
  logout: () => void;
  deleteAccount: (password: string) => Promise<boolean>;
  deleteError: string | null;
}

export default function AuthGate({ children }: { children: (auth: AuthedProps) => React.ReactNode }) {
  const {
    status, userId, email: authedEmail, createdAt, jobsUsedToday, dailyLimit,
    totalReports, sessionExpiresAt, riskTolerance, error, signup, login, logout,
    deleteAccount, setRiskTolerance,
  } = useAuth();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg">
        <span className="font-mono text-xs text-muted">loading...</span>
      </div>
    );
  }

  if (status === "authenticated" && userId) {
    return (
      <>
        {children({
          userId, email: authedEmail, createdAt, jobsUsedToday, dailyLimit,
          totalReports, sessionExpiresAt, riskTolerance, setRiskTolerance,
          logout, deleteAccount, deleteError: error,
        })}
      </>
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim() || !password || submitting) return;
    setSubmitting(true);
    const ok = mode === "signup" ? await signup(email.trim(), password) : await login(email.trim(), password);
    setSubmitting(false);
    if (ok) setPassword("");
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-5">
      <div className="w-full max-w-sm">
        <div className="flex items-center justify-center gap-2 pb-6">
          <span className="font-mono text-base font-bold tracking-wide text-text">FINSIGHT</span>
          <span className="flex h-7 w-7 items-center justify-center rounded-full border border-border font-mono text-[11px] text-muted">
            AI
          </span>
        </div>

        <div className="flex rounded-lg border border-border bg-card p-1">
          {(["login", "signup"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`flex-1 rounded-md py-2 font-mono text-xs font-bold transition-colors ${
                mode === m ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {m === "login" ? "LOG IN" : "SIGN UP"}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="email"
            autoComplete="email"
            className="w-full rounded-lg border border-border bg-card px-3.5 py-3 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="password"
            autoComplete={mode === "signup" ? "new-password" : "current-password"}
            minLength={mode === "signup" ? 8 : undefined}
            className="w-full rounded-lg border border-border bg-card px-3.5 py-3 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent"
          />

          {error && (
            <div className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-danger">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !email.trim() || !password}
            className="w-full rounded-lg bg-accent py-2.5 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "..." : mode === "signup" ? "CREATE ACCOUNT" : "LOG IN"}
          </button>
        </form>

        <p className="mt-6 text-center text-[10px] leading-relaxed text-dim">
          FinSight AI is an automated research tool, not a registered investment adviser,
          broker-dealer, or provider of personalized financial advice. Reports are
          algorithmically generated informational signals only, not a recommendation to buy
          or sell any security. You are solely responsible for your own investment decisions
          -- consult a licensed financial advisor before acting on anything here.
        </p>
      </div>
    </div>
  );
}
