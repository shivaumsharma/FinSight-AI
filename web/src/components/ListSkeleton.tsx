// Placeholder rows for a list that is still loading. Announced to screen readers; the pulse is skipped when the user
// prefers reduced motion.
export default function ListSkeleton({ rows = 3, className = "mt-3" }: { rows?: number; className?: string }) {
  return (
    <div role="status" aria-busy="true" className={`${className} flex flex-col gap-2`}>
      <span className="sr-only">Loading</span>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-[52px] rounded-lg border border-border-subtle bg-card/60 motion-safe:animate-pulse" />
      ))}
    </div>
  );
}
