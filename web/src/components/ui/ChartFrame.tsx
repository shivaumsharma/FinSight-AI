import type { ReactNode } from "react";

export type ChartState = "ready" | "loading" | "error" | "empty";

interface Props {
  title: ReactNode;
  // Right side of the header row, for example a range selector.
  controls?: ReactNode;
  // Row between the header and the chart, for example overlay toggles.
  toolbar?: ReactNode;
  // Status strip above the chart, for example the reconnecting banner.
  banner?: ReactNode;
  state?: ChartState;
  height?: number;
  errorMessage?: string;
  emptyMessage?: string;
  caption?: ReactNode;
  // The chart's own container. Always rendered at full size (even while loading) so the chart library can measure it.
  children: ReactNode;
}

const OVERLAY = "absolute inset-0 grid place-items-center bg-card px-4 text-center font-mono text-caption text-dim";

// Frame for any chart: title row, controls, and loading/error/empty overlays on top of the chart area.
export default function ChartFrame({
  title,
  controls,
  toolbar,
  banner,
  state = "ready",
  height = 320,
  errorMessage = "Couldn't load this chart.",
  emptyMessage = "No data available for this range.",
  caption,
  children,
}: Props) {
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-micro tracking-wide text-dim">{title}</p>
        {controls}
      </div>
      {banner}
      {toolbar}
      <div className="relative mt-2 rounded-lg border border-border bg-card px-2 py-2" style={{ height: height + 16 }}>
        {children}
        {state === "error" && (
          <p role="alert" className={OVERLAY}>
            {errorMessage}
          </p>
        )}
        {state === "loading" && <div role="status" aria-busy="true" className="absolute inset-2 rounded bg-card/60 motion-safe:animate-pulse" />}
        {state === "empty" && <p className={OVERLAY}>{emptyMessage}</p>}
      </div>
      {caption && <p className="mt-1 font-mono text-[9px] text-dim">{caption}</p>}
    </div>
  );
}
