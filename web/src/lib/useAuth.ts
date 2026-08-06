"use client";

import { useCallback, useEffect, useState } from "react";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthState {
  status: AuthStatus;
  userId: string | null;
  email: string | null;
  createdAt: number | null;
  jobsUsedToday: number | null;
  dailyLimit: number | null;
  totalReports: number | null;
  sessionExpiresAt: number | null;
  riskTolerance: string | null;
  resetAt: number | null;
  waitlistFeatures: string[];
  error: string | null;
}

const UNAUTHENTICATED_STATE: AuthState = {
  status: "unauthenticated",
  userId: null,
  email: null,
  createdAt: null,
  jobsUsedToday: null,
  dailyLimit: null,
  totalReports: null,
  sessionExpiresAt: null,
  riskTolerance: null,
  resetAt: null,
  waitlistFeatures: [],
  error: null,
};

// Mirrors useResearch.ts's shape (status/error state + an imperative
// action that updates it) -- same pattern already established for the
// one other stateful client/server round-trip in this app.
export function useAuth() {
  const [state, setState] = useState<AuthState>({ ...UNAUTHENTICATED_STATE, status: "loading" });

  const checkSession = useCallback(async () => {
    try {
      const resp = await fetch("/api/auth/me", { cache: "no-store" });
      if (resp.ok) {
        const data = await resp.json();
        setState({
          status: "authenticated",
          userId: data.user_id,
          email: data.email ?? null,
          createdAt: data.created_at ?? null,
          jobsUsedToday: data.jobs_used_today ?? null,
          dailyLimit: data.daily_limit ?? null,
          totalReports: data.total_reports ?? null,
          sessionExpiresAt: data.session_expires_at ?? null,
          riskTolerance: data.risk_tolerance ?? null,
          resetAt: data.reset_at ?? null,
          waitlistFeatures: data.waitlist_features ?? [],
          error: null,
        });
      } else {
        setState(UNAUTHENTICATED_STATE);
      }
    } catch {
      // A network hiccup checking auth shouldn't strand the user on an
      // infinite loading spinner -- fail to logged-out, same as a real
      // 401 would, since the research form itself will surface a clear
      // "couldn't reach the service" error if they try to use it.
      setState(UNAUTHENTICATED_STATE);
    }
  }, []);

  useEffect(() => {
    checkSession();
  }, [checkSession]);

  const signup = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, error: null }));
    const resp = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (resp.ok) {
      await checkSession();
      return true;
    }
    const body = await resp.json().catch(() => ({ message: "Something went wrong." }));
    setState((s) => ({ ...s, error: body.message || "Couldn't create an account." }));
    return false;
  }, [checkSession]);

  const login = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, error: null }));
    const resp = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (resp.ok) {
      await checkSession();
      return true;
    }
    const body = await resp.json().catch(() => ({ message: "Something went wrong." }));
    setState((s) => ({ ...s, error: body.message || "Couldn't log in." }));
    return false;
  }, [checkSession]);

  const logout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    setState(UNAUTHENTICATED_STATE);
  }, []);

  // Irreversible -- the backend re-verifies the password before
  // deleting anything (see main.py's DELETE /v1/auth/me), this is just
  // the client-side plumbing for that request plus clearing local
  // state to logged-out on success, same as logout() above.
  const deleteAccount = useCallback(async (password: string) => {
    setState((s) => ({ ...s, error: null }));
    const resp = await fetch("/api/auth/me", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (resp.ok) {
      setState(UNAUTHENTICATED_STATE);
      return true;
    }
    const body = await resp.json().catch(() => ({ message: "Something went wrong." }));
    setState((s) => ({ ...s, error: body.message || "Couldn't delete account." }));
    return false;
  }, []);

  // Optimistic update -- the profile UI is a cycle-through-3-values
  // button, and waiting on a round-trip before the badge visibly
  // changes would make every click feel laggy for a preference this
  // low-stakes (worst case on a failed request: it silently reverts
  // next time /api/auth/me is refetched, not a real correctness issue).
  const setRiskTolerance = useCallback(async (level: string) => {
    setState((s) => ({ ...s, riskTolerance: level }));
    await fetch("/api/auth/risk-tolerance", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ risk_tolerance: level }),
    }).catch(() => {});
  }, []);

  // Optimistic, same reasoning as setRiskTolerance above -- a "Join
  // Waitlist" click should feel instant, and the worst case on a
  // failed request is the button reverting on the next /api/auth/me
  // refetch, not a real correctness problem.
  const joinWaitlist = useCallback(async (feature: string) => {
    setState((s) => ({ ...s, waitlistFeatures: [...s.waitlistFeatures, feature] }));
    await fetch("/api/waitlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature }),
    }).catch(() => {});
  }, []);

  return { ...state, signup, login, logout, deleteAccount, setRiskTolerance, joinWaitlist };
}
