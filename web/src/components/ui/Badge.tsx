import type { ReactNode } from "react";

export type BadgeTone = "gain" | "warn" | "loss" | "neutral";
export type BadgeSize = "sm" | "md";

const TONES: Record<BadgeTone, string> = {
  gain: "text-accent border-accent",
  warn: "text-warn border-warn",
  loss: "text-danger border-danger",
  neutral: "text-muted border-dim",
};

const SIZES: Record<BadgeSize, string> = {
  sm: "px-2 py-0.5 text-micro",
  md: "px-3 py-1 text-body",
};

export default function Badge({ tone = "neutral", size = "md", children }: { tone?: BadgeTone; size?: BadgeSize; children: ReactNode }) {
  return (
    <span className={`inline-block rounded border font-mono font-bold uppercase tracking-wide ${TONES[tone]} ${SIZES[size]}`}>
      {children}
    </span>
  );
}
