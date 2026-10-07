// Shared "the fetch failed" block. List/home surfaces used to collapse a
// failed request into their empty-state copy ("No reports yet", "Empty for
// now"), which tells a user their data is gone when the backend merely
// hiccuped. This keeps the two cases visibly different and gives a way out.
export default function LoadError({
  what,
  onRetry,
  className = "mt-3",
}: {
  what: string;
  onRetry: () => void;
  className?: string;
}) {
  return (
    <div role="alert" className={`${className} flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3.5 py-3`}>
      <p className="font-mono text-[11px] text-dim">Couldn&apos;t load {what}.</p>
      <button
        type="button"
        onClick={onRetry}
        className="flex-shrink-0 rounded border border-border px-2.5 py-1 font-mono text-[10px] font-bold text-muted hover:border-accent hover:text-accent"
      >
        RETRY
      </button>
    </div>
  );
}
