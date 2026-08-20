"use client";

import { useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactNumber } from "@/lib/stockFormat";
import { useThemeColors } from "@/lib/useThemeColors";
import { cagr, fdMaturity, lumpsumFutureValue, ppfMaturity, sipFutureValue, type GrowthResult } from "@/lib/calculators";

const CALCULATORS = ["SIP", "LUMPSUM", "FD", "PPF", "CAGR"] as const;
type CalculatorKind = (typeof CALCULATORS)[number];

const COMPOUNDING_OPTIONS = [
  { label: "Yearly", value: 1 },
  { label: "Half-Yearly", value: 2 },
  { label: "Quarterly", value: 4 },
  { label: "Monthly", value: 12 },
];

function NumberField({
  label,
  value,
  onChange,
  min = 0,
  step = "any",
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  step?: number | "any";
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[9px] tracking-wide text-dim">{label}</span>
      <input
        type="number"
        inputMode="decimal"
        min={min}
        step={step}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))}
        className="rounded border border-border bg-bg px-2.5 py-2 font-mono text-xs text-text focus:outline-none focus:border-accent"
      />
    </label>
  );
}

function StatCard({ label, value, tone = "text" }: { label: string; value: string; tone?: "text" | "accent" }) {
  return (
    <div className="rounded-lg border border-border bg-card px-2.5 py-2 text-center">
      <div className="font-mono text-[9px] tracking-wide text-dim">{label}</div>
      <div className={`mt-0.5 font-mono text-xs font-bold ${tone === "accent" ? "text-accent" : "text-text"}`}>{value}</div>
    </div>
  );
}

function GrowthChartTooltip({ active, payload, symbol }: { active?: boolean; payload?: { name: string; value: number; payload: { color: string } }[]; symbol: string }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="rounded-lg border border-border bg-card px-2.5 py-1.5 font-mono text-[10px] shadow-lg" style={{ color: p.payload.color }}>
      {p.name}: {symbol}
      {fmtCompactNumber(p.value)}
    </div>
  );
}

// Shared result display for every growth calculator (SIP/Lumpsum/FD/PPF)
// -- same three numbers (invested, returns, total) and the same
// invested-vs-returns split, just fed a different GrowthResult.
function GrowthOutput({ result, symbol }: { result: GrowthResult; symbol: string }) {
  const colors = useThemeColors();
  const data = [
    { name: "Invested", value: result.investedAmount, color: colors.dim },
    { name: "Returns", value: Math.max(result.estimatedReturns, 0), color: colors.accent },
  ];
  const hasReturns = result.investedAmount > 0 && result.totalValue > 0;

  return (
    <div className="mt-4">
      <div className="grid grid-cols-3 gap-2">
        <StatCard label="INVESTED" value={`${symbol}${fmtCompactNumber(result.investedAmount)}`} />
        <StatCard label="EST. RETURNS" value={`${symbol}${fmtCompactNumber(result.estimatedReturns)}`} tone="accent" />
        <StatCard label="TOTAL VALUE" value={`${symbol}${fmtCompactNumber(result.totalValue)}`} />
      </div>
      {hasReturns && (
        <div className="mt-3 h-[140px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey="value" nameKey="name" innerRadius={38} outerRadius={58} paddingAngle={3} strokeWidth={0}>
                {data.map((d) => (
                  <Cell key={d.name} fill={d.color} />
                ))}
              </Pie>
              <Tooltip content={<GrowthChartTooltip symbol={symbol} />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function SipCalculator({ symbol }: { symbol: string }) {
  const [monthlyAmount, setMonthlyAmount] = useState(10000);
  const [annualRatePct, setAnnualRatePct] = useState(12);
  const [years, setYears] = useState(10);
  const result = sipFutureValue(monthlyAmount, annualRatePct, years);

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="MONTHLY INVESTMENT" value={monthlyAmount} onChange={setMonthlyAmount} />
        <NumberField label="EXPECTED RETURN % (ANNUAL)" value={annualRatePct} onChange={setAnnualRatePct} />
        <NumberField label="TIME PERIOD (YEARS)" value={years} onChange={setYears} min={1} step={1} />
      </div>
      <GrowthOutput result={result} symbol={symbol} />
    </div>
  );
}

function LumpsumCalculator({ symbol }: { symbol: string }) {
  const [principal, setPrincipal] = useState(100000);
  const [annualRatePct, setAnnualRatePct] = useState(12);
  const [years, setYears] = useState(10);
  const result = lumpsumFutureValue(principal, annualRatePct, years);

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="LUMPSUM AMOUNT" value={principal} onChange={setPrincipal} />
        <NumberField label="EXPECTED RETURN % (ANNUAL)" value={annualRatePct} onChange={setAnnualRatePct} />
        <NumberField label="TIME PERIOD (YEARS)" value={years} onChange={setYears} min={1} step={1} />
      </div>
      <GrowthOutput result={result} symbol={symbol} />
    </div>
  );
}

function FdCalculator({ symbol }: { symbol: string }) {
  const [principal, setPrincipal] = useState(100000);
  const [annualRatePct, setAnnualRatePct] = useState(7);
  const [years, setYears] = useState(5);
  const [compoundingPerYear, setCompoundingPerYear] = useState(4);
  const result = fdMaturity(principal, annualRatePct, years, compoundingPerYear);

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="DEPOSIT AMOUNT" value={principal} onChange={setPrincipal} />
        <NumberField label="INTEREST RATE % (ANNUAL)" value={annualRatePct} onChange={setAnnualRatePct} />
        <NumberField label="TENURE (YEARS)" value={years} onChange={setYears} min={1} step={1} />
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[9px] tracking-wide text-dim">COMPOUNDING</span>
          <select
            value={compoundingPerYear}
            onChange={(e) => setCompoundingPerYear(Number(e.target.value))}
            className="rounded border border-border bg-bg px-2.5 py-2 font-mono text-xs text-text focus:outline-none focus:border-accent"
          >
            {COMPOUNDING_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <GrowthOutput result={result} symbol={symbol} />
    </div>
  );
}

function PpfCalculator({ symbol }: { symbol: string }) {
  const [yearlyAmount, setYearlyAmount] = useState(150000);
  const [annualRatePct, setAnnualRatePct] = useState(7.1);
  const [years, setYears] = useState(15);
  const result = ppfMaturity(yearlyAmount, annualRatePct, years);

  return (
    <div>
      <p className="mb-2 text-[11px] leading-relaxed text-dim">
        Generic yearly-deposit, annually-compounded calculator -- does not enforce PPF&apos;s real contribution
        limits or lock-in. Check the current government-notified rate before relying on this.
      </p>
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="YEARLY DEPOSIT" value={yearlyAmount} onChange={setYearlyAmount} />
        <NumberField label="INTEREST RATE % (ANNUAL)" value={annualRatePct} onChange={setAnnualRatePct} />
        <NumberField label="TENURE (YEARS)" value={years} onChange={setYears} min={1} step={1} />
      </div>
      <GrowthOutput result={result} symbol={symbol} />
    </div>
  );
}

function CagrCalculator({ symbol }: { symbol: string }) {
  const [initialValue, setInitialValue] = useState(100000);
  const [finalValue, setFinalValue] = useState(200000);
  const [years, setYears] = useState(5);
  const rate = cagr(initialValue, finalValue, years);

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        <NumberField label="INITIAL VALUE" value={initialValue} onChange={setInitialValue} />
        <NumberField label="FINAL VALUE" value={finalValue} onChange={setFinalValue} />
        <NumberField label="TIME PERIOD (YEARS)" value={years} onChange={setYears} min={1} step={1} />
      </div>
      <div className="mt-4 rounded-lg border border-border bg-card px-4 py-4 text-center">
        <div className="font-mono text-[9px] tracking-wide text-dim">CAGR</div>
        <div className="mt-1 font-mono text-xl font-bold text-accent">
          {rate !== null ? `${rate.toFixed(2)}%` : "--"}
        </div>
        {rate === null && (
          <p className="mt-1 font-mono text-[10px] text-dim">Enter positive initial/final values and a positive time period.</p>
        )}
        {rate !== null && (
          <p className="mt-1 font-mono text-[10px] text-dim">
            {symbol}
            {fmtCompactNumber(initialValue)} &rarr; {symbol}
            {fmtCompactNumber(finalValue)} over {years} {years === 1 ? "year" : "years"}
          </p>
        )}
      </div>
    </div>
  );
}

// Pure client-side financial calculators -- no backend call, unlike
// every other page in this app, since none of this needs live data.
// Same "Multibagg AI gap" motivation as Screener.tsx: a quick,
// no-new-data-source engagement feature.
export default function Calculators() {
  const [kind, setKind] = useState<CalculatorKind>("SIP");
  const [currency, setCurrency] = useState<"INR" | "USD">("INR");
  const symbol = currencySymbol(currency);

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex flex-wrap gap-1 rounded-lg border border-border bg-card p-0.5">
          {CALCULATORS.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setKind(c)}
              className={`rounded px-2.5 py-1 font-mono text-[10px] font-bold ${
                kind === c ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {c}
            </button>
          ))}
        </div>
        <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
          {(["INR", "USD"] as const).map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setCurrency(c)}
              className={`rounded px-2 py-1 font-mono text-[10px] font-bold ${
                currency === c ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {currencySymbol(c)}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-3 rounded-lg border border-border bg-card p-3.5">
        {kind === "SIP" && <SipCalculator symbol={symbol} />}
        {kind === "LUMPSUM" && <LumpsumCalculator symbol={symbol} />}
        {kind === "FD" && <FdCalculator symbol={symbol} />}
        {kind === "PPF" && <PpfCalculator symbol={symbol} />}
        {kind === "CAGR" && <CagrCalculator symbol={symbol} />}
      </div>
    </div>
  );
}
