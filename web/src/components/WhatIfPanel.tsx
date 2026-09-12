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
  const [scoring, setScoring] = useState<NonNullable<WhatIfResponse["scoring"]> | null>(null);
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
        setScoring(data.scoring ?? null);
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
      </div>

      <div className="mt-3">
        <RatingBadge rating={result.rating} size="sm" />
      </div>

      {scoring && <CompositeScoreBreakdown result={result} scoring={scoring} />}
    </div>
  );
}

// Shows HOW the composite score above was actually built -- both
// weighted component contributions, not just the blended total, and
// where that total falls relative to the real Buy/Hold/Sell zone
// boundaries. scoring's numbers come straight from the backend (see
// WhatIfResponse's own type comment) so this can never silently drift
// from report_data_builder.py's actual formula.
function CompositeScoreBreakdown({
  result,
  scoring,
}: {
  result: WhatIfResult;
  scoring: NonNullable<WhatIfResponse["scoring"]>;
}) {
  if (result.composite_score === null) return null;
  const { composite_score, dcf_score, relative_score } = result;
  const { buy_threshold, sell_threshold, dcf_weight, relative_weight, score_cap } = scoring;

  // Position as a 0-100% offset along the -score_cap..+score_cap scale.
  const toPct = (v: number) => Math.min(100, Math.max(0, ((v + score_cap) / (2 * score_cap)) * 100));
  const scorePct = toPct(composite_score);
  const sellPct = toPct(sell_threshold);
  const buyPct = toPct(buy_threshold);

  const zoneColor =
    composite_score >= buy_threshold ? "text-accent" : composite_score <= sell_threshold ? "text-danger" : "text-warn";
  const zoneDotColor =
    composite_score >= buy_threshold ? "bg-accent" : composite_score <= sell_threshold ? "bg-danger" : "bg-warn";

  return (
    <div className="mt-4 rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">COMPOSITE SCORE BREAKDOWN</p>
        <span className={`font-mono text-sm font-bold ${zoneColor}`}>
          {composite_score >= 0 ? "+" : ""}
          {composite_score.toFixed(1)}
        </span>
      </div>

      <div className="relative mt-3 h-2 rounded-full bg-border-subtle">
        <div className="absolute inset-y-0 left-0 rounded-l-full bg-danger/30" style={{ width: `${sellPct}%` }} />
        <div className="absolute inset-y-0 bg-warn/30" style={{ left: `${sellPct}%`, width: `${buyPct - sellPct}%` }} />
        <div
          className="absolute inset-y-0 right-0 rounded-r-full bg-accent/30"
          style={{ left: `${buyPct}%` }}
        />
        <div
          className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-bg ${zoneDotColor}`}
          style={{ left: `${scorePct}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[9px] text-dim">
        <span>SELL (&le;{sell_threshold.toFixed(1)})</span>
        <span>HOLD</span>
        <span>BUY (&ge;{buy_threshold.toFixed(1)})</span>
      </div>

      <div className="mt-3 flex flex-col gap-1 border-t border-border-subtle pt-2">
        <div className="flex items-center justify-between font-mono text-[11px]">
          <span className="text-muted">
            DCF ({dcf_score >= 0 ? "+" : ""}
            {dcf_score.toFixed(1)}) &times; {relative_score !== null ? `${(dcf_weight * 100).toFixed(0)}%` : "100%"}
          </span>
          <span className="text-text">
            {((relative_score !== null ? dcf_weight : 1) * dcf_score).toFixed(1)}
          </span>
        </div>
        {relative_score !== null ? (
          <div className="flex items-center justify-between font-mono text-[11px]">
            <span className="text-muted">
              Relative ({relative_score >= 0 ? "+" : ""}
              {relative_score.toFixed(1)}) &times; {(relative_weight * 100).toFixed(0)}%
            </span>
            <span className="text-text">{(relative_weight * relative_score).toFixed(1)}</span>
          </div>
        ) : (
          <p className="font-mono text-[10px] text-dim">Relative valuation unavailable for this company -- DCF weighted at 100%.</p>
        )}
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
