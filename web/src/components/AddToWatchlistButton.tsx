"use client";

import { useState } from "react";

function BookmarkIcon({ filled }: { filled: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.8}>
      <path d="M6 3.5h12a.5.5 0 0 1 .5.5v17l-6.5-4-6.5 4V4a.5.5 0 0 1 .5-.5z" strokeLinejoin="round" />
    </svg>
  );
}

// Extracted from MarketMovers.tsx (its original owner) once Screener.tsx
// needed the exact same one-tap add -- both surface tickers a user has
// never explicitly researched, so "add to watchlist right from the row"
// is the same real action in both places, not two coincidentally similar
// ones.
export default function AddToWatchlistButton({ ticker }: { ticker: string }) {
  const [state, setState] = useState<"idle" | "adding" | "added" | "error">("idle");

  async function handleAdd() {
    setState("adding");
    try {
      const resp = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      if (!resp.ok) throw new Error("watchlist add failed");
      setState("added");
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 2000);
    }
  }

  return (
    <button
      type="button"
      onClick={handleAdd}
      disabled={state === "adding" || state === "added"}
      title={state === "added" ? "On watchlist" : "Add to watchlist"}
      className={`shrink-0 ${state === "added" ? "text-accent" : "text-dim hover:text-accent"} ${state === "error" ? "text-danger" : ""}`}
    >
      <BookmarkIcon filled={state === "added"} />
    </button>
  );
}
