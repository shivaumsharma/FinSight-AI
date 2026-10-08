"use client";

import type { StreamStatus } from "@/lib/priceStream";

// Shown above live-price screens. "reconnecting" means prices were live and the link dropped; "unavailable" means the
// live feed never connected, so the numbers are whatever was loaded with the page.
export default function ConnectionBanner({ status }: { status: StreamStatus }) {
  if (status !== "reconnecting" && status !== "unavailable") return null;
  const reconnecting = status === "reconnecting";
  return (
    <div
      role="status"
      data-testid="connection-banner"
      className="mt-2 flex items-center gap-2 rounded-lg border border-warn/50 bg-card px-3 py-2 font-mono text-[11px] text-muted"
    >
      <span className={`h-2 w-2 flex-shrink-0 rounded-full bg-warn ${reconnecting ? "motion-safe:animate-pulse" : ""}`} aria-hidden />
      {reconnecting
        ? "Reconnecting to live prices. Showing the last prices received."
        : "Live prices aren't available right now. Showing prices from when the page loaded."}
    </div>
  );
}
