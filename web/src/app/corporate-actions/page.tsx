"use client";

import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import CorporateActionsFeed from "@/components/CorporateActionsFeed";

// Not on the bottom nav (see BottomNav.tsx's own comment on
// deliberately staying at 4 tabs) -- reached via a link from the
// Watchlist page instead, same pattern as /indices being reached from
// the home IndicesCarousel.
export default function CorporateActionsPage() {
  return (
    <AuthGate>
      {() => (
        <div className="min-h-screen bg-bg pb-20">
          <div className="mx-auto max-w-2xl px-5 py-8">
            <Link href="/watchlist" className="font-mono text-xs font-bold text-muted hover:text-accent">
              &larr; WATCHLIST
            </Link>
            <h1 className="mt-3 font-mono text-lg font-bold text-text">Corporate Actions</h1>
            <p className="mt-1 text-xs text-dim">Upcoming earnings and ex-dividend dates for your tracked tickers.</p>
            <div className="mt-4">
              <CorporateActionsFeed />
            </div>
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}
