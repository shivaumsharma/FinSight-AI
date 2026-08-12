"use client";

import { useEffect, useState } from "react";

// Same localStorage-JSON convention as useResearch.ts's own
// finsight_last_report key -- a real, client-side-only "recently
// viewed" list, capped at MAX_ENTRIES, most-recent first, deduplicated
// on re-visit rather than appending a second entry for the same ticker.
const STORAGE_KEY = "finsight_recently_viewed";
const MAX_ENTRIES = 20;

export interface RecentlyViewedEntry {
  ticker: string;
  companyName: string | null;
  viewedAt: number;
}

function readAll(): RecentlyViewedEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// Call on mount from the stock detail page -- records the visit and
// returns the up-to-date list (including this visit) for immediate
// use, since a component reading its own localStorage write in the
// same tick would otherwise need a second render to see it.
export function recordVisit(ticker: string, companyName: string | null): RecentlyViewedEntry[] {
  const existing = readAll().filter((e) => e.ticker !== ticker);
  const updated = [{ ticker, companyName, viewedAt: Date.now() }, ...existing].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch {
    // Safari private mode / quota exceeded -- the list just doesn't
    // persist this visit, never a hard failure for the page itself.
  }
  return updated;
}

// Read-only hook for components that just want to DISPLAY the list
// (RecentlyViewed.tsx) without recording a new visit themselves.
export function useRecentlyViewed(): RecentlyViewedEntry[] {
  const [entries, setEntries] = useState<RecentlyViewedEntry[]>([]);
  useEffect(() => {
    setEntries(readAll());
  }, []);
  return entries;
}
