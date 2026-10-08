// Design tokens, mirrored from src/app/globals.css (tokens.test.ts fails if the two drift apart). Names here are the
// names to use for Figma variables, so design and code share one vocabulary.

export const COLOR_TOKENS = {
  bg: { dark: "#05070a", light: "#f7f2e6" },
  card: { dark: "#0d1117", light: "#fffcf5" },
  border: { dark: "#1f2733", light: "#e2d9c3" },
  "border-subtle": { dark: "#131a22", light: "#ece4d1" },
  text: { dark: "#e6edf3", light: "#221d14" },
  muted: { dark: "#7d8590", light: "#6e6350" },
  dim: { dark: "#3a4353", light: "#8f8265" },
  accent: { dark: "#00d97e", light: "#7a5a1c" },
  danger: { dark: "#ff4d4f", light: "#b3352b" },
  warn: { dark: "#f2a900", light: "#c17817" },
} as const;

// CSS variable -> value, with the Tailwind utility each one produces.
export const TYPE_TOKENS = {
  micro: { cssVar: "--text-micro", value: "0.625rem", px: 10, utility: "text-micro" },
  caption: { cssVar: "--text-caption", value: "0.6875rem", px: 11, utility: "text-caption" },
  small: { cssVar: "--text-small", value: "0.75rem", px: 12, utility: "text-small" },
  body: { cssVar: "--text-body", value: "0.875rem", px: 14, utility: "text-body" },
  title: { cssVar: "--text-title", value: "1.125rem", px: 18, utility: "text-title" },
  display: { cssVar: "--text-display", value: "1.25rem", px: 20, utility: "text-display" },
} as const;

export const SPACING_TOKENS = {
  gutter: { cssVar: "--spacing-gutter", value: "1.25rem", px: 20, utility: "px-gutter" },
  "card-x": { cssVar: "--spacing-card-x", value: "0.875rem", px: 14, utility: "px-card-x" },
  "card-y": { cssVar: "--spacing-card-y", value: "0.625rem", px: 10, utility: "py-card-y" },
} as const;

export const RADIUS_TOKENS = {
  control: { cssVar: "--radius-control", value: "0.5rem", px: 8, utility: "rounded-control" },
} as const;
