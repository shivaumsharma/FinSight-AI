"use client";

import { useCallback, useEffect, useState } from "react";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthState {
  status: AuthStatus;
  userId: string | null;
  error: string | null;
}

// Mirrors useResearch.ts's shape (status/error state + an imperative
// action that updates it) -- same pattern already established for the
// one other stateful client/server round-trip in this app.
export function useAuth() {
  const [state, setState] = useState<AuthState>({ status: "loading", userId: null, error: null });

  const checkSession = useCallback(async () => {
    try {
      const resp = await fetch("/api/auth/me", { cache: "no-store" });
      if (resp.ok) {
        const data = await resp.json();
        setState({ status: "authenticated", userId: data.user_id, error: null });
      } else {
        setState({ status: "unauthenticated", userId: null, error: null });
      }
    } catch {
      // A network hiccup checking auth shouldn't strand the user on an
      // infinite loading spinner -- fail to logged-out, same as a real
      // 401 would, since the research form itself will surface a clear
      // "couldn't reach the service" error if they try to use it.
      setState({ status: "unauthenticated", userId: null, error: null });
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
    setState({ status: "unauthenticated", userId: null, error: null });
  }, []);

  return { ...state, signup, login, logout };
}
