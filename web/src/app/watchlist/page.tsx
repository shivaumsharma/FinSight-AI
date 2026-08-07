"use client";

import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import Watchlist from "@/components/Watchlist";

// Promoted from a home-page section to its own route now that there's
// a persistent bottom nav -- same <Watchlist/> component, just given a
// full page instead of squeezed between the search form and the
// pipeline preview.
export default function WatchlistPage() {
  return (
    <AuthGate>
      {() => (
        <div className="min-h-screen bg-bg pb-20">
          <div className="mx-auto max-w-2xl px-5 py-8">
            <div className="flex items-center justify-between">
              <h1 className="font-mono text-lg font-bold text-text">Watchlist</h1>
              <Link href="/corporate-actions" className="font-mono text-[10px] font-bold text-muted hover:text-accent">
                CORPORATE ACTIONS &rarr;
              </Link>
            </div>
            <Watchlist />
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}
