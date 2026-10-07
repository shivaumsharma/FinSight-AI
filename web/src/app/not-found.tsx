import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-5">
      <div className="w-full max-w-sm text-center">
        <p className="font-mono text-[10px] tracking-wide text-dim">404</p>
        <p className="mt-2 font-mono text-sm text-text">We couldn&apos;t find that page.</p>
        <p className="mt-2 font-mono text-xs text-muted">The link may be old or mistyped.</p>
        <Link
          href="/"
          className="mt-5 inline-block rounded-lg border border-border bg-card px-4 py-2 font-mono text-xs font-bold text-text hover:border-accent"
        >
          GO HOME
        </Link>
      </div>
    </div>
  );
}
