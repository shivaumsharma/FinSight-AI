"use client";

import { useCallback, useEffect, useState } from "react";

// "unreachable" is distinct from "unauthenticated" -- see checkSession's
// own comment below on why /api/auth/me's proxy route (src/app/api/
// auth/me/route.ts) reports a backend-unreachable failure differently
// from a real 401, and why this hook must not collapse the two.
type AuthStatus = "loading" | "authenticated" | "unauthenticated" | "unreachable";

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
  displayName: string | null;
  onboardingCompleted: boolean;
  investmentGoal: string | null;
  investmentHorizon: string | null;
  interestedInCrypto: boolean;
  interestedInRealEstate: boolean;
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
  displayName: null,
  onboardingCompleted: false,
  investmentGoal: null,
  investmentHorizon: null,
  interestedInCrypto: false,
  interestedInRealEstate: false,
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
          displayName: data.display_name ?? null,
          onboardingCompleted: data.onboarding_completed ?? false,
          investmentGoal: data.investment_goal ?? null,
          investmentHorizon: data.investment_horizon ?? null,
          interestedInCrypto: data.interested_in_crypto ?? false,
          interestedInRealEstate: data.interested_in_real_estate ?? false,
          error: null,
        });
        return;
      }

      // A crashed/unreachable backend surfaces as 503 with this specific
      // code (see src/lib/proxyFetch.ts) -- it must NOT be treated the
      // same as a real 401 "not logged in" response, since collapsing
      // the two would silently log out a validly-authenticated user on
      // a mere backend hiccup.
      const body = await resp.json().catch(() => null);
      if (resp.status === 503 && body?.code === "BACKEND_UNREACHABLE") {
        setState((s) => ({ ...s, status: "unreachable" }));
        return;
      }
      setState(UNAUTHENTICATED_STATE);
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
  // low-stakes. On a failed request the optimistic value is reverted
  // back to what it was before the click, rather than silently left
  // showing a value that was never actually persisted -- the hook
  // handles this itself so callers (e.g. profile/page.tsx) don't need
  // to do anything with the returned promise.
  const setRiskTolerance = useCallback(async (level: string) => {
    let previous: string | null = null;
    setState((s) => {
      previous = s.riskTolerance;
      return { ...s, riskTolerance: level };
    });
    try {
      const resp = await fetch("/api/auth/risk-tolerance", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ risk_tolerance: level }),
      });
      if (!resp.ok) setState((s) => ({ ...s, riskTolerance: previous }));
    } catch {
      setState((s) => ({ ...s, riskTolerance: previous }));
    }
  }, []);

  // Optimistic, same revert-on-failure reasoning as setRiskTolerance
  // above -- a "Join Waitlist" click should feel instant, but a failed
  // request must not leave the button showing "On the list" for a
  // waitlist entry that was never actually saved.
  const joinWaitlist = useCallback(async (feature: string) => {
    setState((s) => ({ ...s, waitlistFeatures: [...s.waitlistFeatures, feature] }));
    try {
      const resp = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feature }),
      });
      if (!resp.ok) {
        setState((s) => ({ ...s, waitlistFeatures: s.waitlistFeatures.filter((f) => f !== feature) }));
      }
    } catch {
      setState((s) => ({ ...s, waitlistFeatures: s.waitlistFeatures.filter((f) => f !== feature) }));
    }
  }, []);

  // Optimistic, same revert-on-failure reasoning as setRiskTolerance --
  // the 40-char cap is already enforced client-side on the input, so a
  // server-side rejection here would only happen from a genuine
  // request failure, not a real user path, but it still shouldn't
  // leave the UI showing an unsaved name as if it were saved.
  const setDisplayName = useCallback(async (name: string | null) => {
    let previous: string | null = null;
    setState((s) => {
      previous = s.displayName;
      return { ...s, displayName: name };
    });
    try {
      const resp = await fetch("/api/auth/display-name", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: name }),
      });
      if (!resp.ok) setState((s) => ({ ...s, displayName: previous }));
    } catch {
      setState((s) => ({ ...s, displayName: previous }));
    }
  }, []);

  // NOT optimistic, unlike setRiskTolerance/joinWaitlist/setDisplayName
  // above -- onboardingCompleted gates AuthGate.tsx's whole render
  // branch, so flipping it locally before the server confirms the save
  // would let a user past the one-time questionnaire on a request that
  // actually failed, with no way back to it since AuthGate wouldn't
  // show it again.
  const completeOnboarding = useCallback(async (answers: {
    riskTolerance: string; investmentGoal: string; investmentHorizon: string;
    interestedInCrypto: boolean; interestedInRealEstate: boolean;
  }) => {
    setState((s) => ({ ...s, error: null }));
    const resp = await fetch("/api/auth/onboarding", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        risk_tolerance: answers.riskTolerance,
        investment_goal: answers.investmentGoal,
        investment_horizon: answers.investmentHorizon,
        interested_in_crypto: answers.interestedInCrypto,
        interested_in_real_estate: answers.interestedInRealEstate,
      }),
    });
    if (resp.ok) {
      await checkSession();
      return true;
    }
    const body = await resp.json().catch(() => ({ message: "Something went wrong." }));
    setState((s) => ({ ...s, error: body.message || "Couldn't save your preferences." }));
    return false;
  }, [checkSession]);

  return {
    ...state, signup, login, logout, deleteAccount, setRiskTolerance, joinWaitlist, setDisplayName,
    completeOnboarding, retry: checkSession,
  };
}
