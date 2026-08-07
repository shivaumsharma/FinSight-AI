// Shared "still loading" placeholder for home-dashboard cards -- keeps
// each section's own label visible immediately instead of the whole
// block popping into existence once its fetch resolves, which reads as
// layout jank on a page with 6+ independent fetches racing each other.
export default function SectionSkeleton({
  label,
  rows = 2,
  variant = "stack",
}: {
  label: string;
  rows?: number;
  variant?: "stack" | "row";
}) {
  return (
    <div className="mt-6">
      <p className="font-mono text-[10px] tracking-wide text-dim">{label}</p>
      {variant === "row" ? (
        <div className="mt-2 flex gap-2.5 overflow-hidden">
          {Array.from({ length: rows }).map((_, i) => (
            <div
              key={i}
              className="h-[64px] w-[108px] flex-shrink-0 animate-pulse rounded-lg border border-border-subtle bg-card/60"
            />
          ))}
        </div>
      ) : (
        <div className="mt-2 flex flex-col gap-2">
          {Array.from({ length: rows }).map((_, i) => (
            <div key={i} className="h-[52px] animate-pulse rounded-lg border border-border-subtle bg-card/60" />
          ))}
        </div>
      )}
    </div>
  );
}
