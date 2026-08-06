"use client";

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
            <h1 className="font-mono text-lg font-bold text-text">Watchlist</h1>
            <Watchlist />
          </div>
          <BottomNav />
        </div>
      )}
    </AuthGate>
  );
}
