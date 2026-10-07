"use client";

import { useEffect } from "react";
import Link from "next/link";

// Route-level error boundary. Without it, any render-time exception (a
// malformed API payload, an unguarded property read) blanks the entire page.
// This keeps the app shell, says what happened, and offers a way back.
export default function RouteError({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-5">
      <div role="alert" className="w-full max-w-sm text-center">
        <p className="font-mono text-sm text-text">Something went wrong on this page.</p>
        <p className="mt-2 font-mono text-xs text-muted">
          Your data is safe. Try again, or head back to the home screen.
        </p>
        <div className="mt-5 flex justify-center gap-2">
          <button
            type="button"
            onClick={() => unstable_retry()}
            className="rounded-lg border border-border bg-card px-4 py-2 font-mono text-xs font-bold text-text hover:border-accent"
          >
            TRY AGAIN
          </button>
          <Link
            href="/"
            className="rounded-lg border border-border px-4 py-2 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent"
          >
            HOME
          </Link>
        </div>
      </div>
    </div>
  );
}
