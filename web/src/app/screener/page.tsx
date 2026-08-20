"use client";

import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import Screener from "@/components/Screener";

// Not on the bottom nav (see BottomNav.tsx's own comment on deliberately
// staying at 4 tabs) -- reached via a link from Home's Market Movers
// section instead, same "linked page, not a tab" pattern as
// /corporate-actions being reached from Watchlist.
export default function ScreenerPage() {
  return (
    <AuthGate>
      {() => (
        <div className="min-h-screen bg-bg pb-20">
          <div className="mx-auto max-w-2xl px-5 py-8">
            <Link href="/" className="font-mono text-xs font-bold text-muted hover:text-accent">
              &larr; HOME
            </Link>
            <h1 className="mt-3 font-mono text-lg font-bold text-text">Screener</h1>
            <p className="mt-1 text-xs text-dim">
              Filter the Tracked Universe by market cap, valuation, dividend yield, and volume.
            </p>
            <div className="mt-4">
              <Screener />
            </div>
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}
