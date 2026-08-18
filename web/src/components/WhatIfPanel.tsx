"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import RatingBadge from "./RatingBadge";
import type { WhatIfResponse, WhatIfResult } from "@/lib/types";

// Ports streamlit_app.py's "What-If: Adjust DCF Assumptions" sliders
// panel (app/valuation/what_if_dcf.py) to the API-backed frontend.
// Unlike ModelCompare.tsx (a single button click), this is
// slider-driven: every drag debounces a re-POST to `endpoint` (see
// web/src/app/api/research/[jobId]/what-if/route.ts) so dragging
// doesn't spam the backend while still feeling close to live.
// `endpoint` is the full proxy path, same endpoint-agnostic
// convention ModelCompare.tsx already uses.
const DEBOUNCE_MS = 300;

type SliderValues = {
  growth_rate_pct: number;
  wacc_pct: number;
  terminal_growth_pct: number;
};

export default function WhatIfPanel({ endpoint, symbol }: { endpoint: string; symbol: string }) {
  const [state, setState] = useState<"loading" | "unavailable" | "error" | "ready">("loading");
  const [bounds, setBounds] = useState<NonNullable<WhatIfResponse["bounds"]> | null>(null);
  const [values, setValues] = useState<SliderValues | null>(null);
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchWhatIf = useCallback(
    async (body: Partial<SliderValues>) => {
      try {
        const resp = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error("what-if request failed");
        const data: WhatIfResponse = await resp.json();
        if (!data.available || !data.bounds || !data.used || !data.result) {
          setState("unavailable");
          return;
        }
        setBounds(data.bounds);
        setValues(data.used);
        setResult(data.result);
        setState("ready");
      } catch {
        setState("error");
      }
    },
    [endpoint]
  );

  // On mount (and whenever the panel switches to a different report),
  // fetch once with an empty body -- the backend fills in
  // bounds/defaults/an initial result computed at those defaults.
  useEffect(() => {
    setState("loading");
    fetchWhatIf({});
  }, [fetchWhatIf]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  function handleSliderChange(key: keyof SliderValues, value: number) {
    if (!values) return;
    const next = { ...values, [key]: value };
    setValues(next);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchWhatIf(next);
    }, DEBOUNCE_MS);
  }

  if (state === "loading") {
    return <div className="mt-4 font-mono text-xs text-muted">Loading what-if explorer&hellip;</div>;
  }

  if (state === "unavailable") {
    return <p className="mt-4 text-xs text-muted">What-if exploration isn&apos;t available for this company.</p>;
  }

  if (state === "error") {
    return <p className="mt-4 text-xs text-danger">Couldn&apos;t load the what-if explorer -- try again.</p>;
  }

  if (!values || !bounds || !result) return null;

  return (
    <div className="mt-2">
      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
        WHAT-IF: ADJUST DCF ASSUMPTIONS
      </div>
      <p className="mb-4 text-[11px] leading-relaxed text-muted">
        This is a what-if exploration, not the official recommendation above -- relative valuation stays
        fixed at its actual computed value; only the DCF assumptions change.
      </p>

      <div className="space-y-4">
        <SliderRow
          label="Revenue Growth Rate (near-term)"
          value={values.growth_rate_pct}
          min={bounds.growth_rate_pct.min}
          max={bounds.growth_rate_pct.max}
          step={0.5}
          onChange={(v) => handleSliderChange("growth_rate_pct", v)}
        />
        <SliderRow
          label="WACC"
          value={values.wacc_pct}
          min={bounds.wacc_pct.min}
          max={bounds.wacc_pct.max}
          step={0.25}
          onChange={(v) => handleSliderChange("wacc_pct", v)}
        />
        <SliderRow
          label="Terminal Growth Rate"
          value={values.terminal_growth_pct}
          min={bounds.terminal_growth_pct.min}
          max={bounds.terminal_growth_pct.max}
          step={0.25}
          onChange={(v) => handleSliderChange("terminal_growth_pct", v)}
        />
      </div>

      {result.wacc_floored && (
        <p className="mt-3 text-[10px] text-warn">
          Note: WACC floored to {result.wacc_used.toFixed(2)}% at this slider position to avoid
          terminal-value instability (same floor the real DCF uses).
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <StatTile
          label="INTRINSIC VALUE"
          value={`${symbol}${result.intrinsic_value.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}`}
        />
        <StatTile label="UPSIDE" value={`${result.upside_percent >= 0 ? "+" : ""}${result.upside_percent.toFixed(1)}%`} />
        <StatTile
          label="COMPOSITE SCORE"
          value={result.composite_score !== null ? `${result.composite_score >= 0 ? "+" : ""}${result.composite_score.toFixed(1)}` : "N/A"}
        />
      </div>

      <div className="mt-3">
        <RatingBadge rating={result.rating} size="sm" />
      </div>
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-[120px] flex-1 rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="font-mono text-[10px] text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-sm font-bold text-text">{value}</div>
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label className="font-mono text-[11px] text-muted">{label}</label>
        <span className="font-mono text-xs font-bold text-accent">{value.toFixed(2)}%</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}
