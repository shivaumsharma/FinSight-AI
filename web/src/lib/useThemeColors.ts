"use client";

import { useEffect, useState } from "react";

// Colors read from this app's own CSS variables (set globally in
// globals.css) rather than hardcoded hex, matching the rest of the
// app's theme-driven styling -- recharts needs a resolved string, not a
// CSS var reference, so these are read from computed styles once on
// mount. Extracted from FinancialPerformanceChart.tsx (its original
// owner) once Calculators.tsx needed the exact same resolved palette
// for its own recharts usage.
export function useThemeColors() {
  const [colors, setColors] = useState({ accent: "#c9a227", danger: "#e05252", dim: "#7a7a7a" });
  useEffect(() => {
    const style = getComputedStyle(document.documentElement);
    setColors({
      accent: style.getPropertyValue("--accent").trim() || "#c9a227",
      danger: style.getPropertyValue("--danger").trim() || "#e05252",
      dim: style.getPropertyValue("--dim").trim() || "#7a7a7a",
    });
  }, []);
  return colors;
}
