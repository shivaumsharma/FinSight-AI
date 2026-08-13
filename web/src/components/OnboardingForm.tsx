"use client";

import { useState } from "react";

const RISK_LEVELS = ["Conservative", "Moderate", "Aggressive"] as const;
const GOALS = ["Wealth Growth", "Retirement", "Income", "Capital Preservation"] as const;
const HORIZONS = ["Short-term (<3y)", "Medium (3-7y)", "Long-term (7y+)"] as const;

interface Answers {
  riskTolerance: string;
  investmentGoal: string;
  investmentHorizon: string;
  interestedInCrypto: boolean;
  interestedInRealEstate: boolean;
}

function OptionRow({ label, options, value, onChange }: { label: string; options: readonly string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">{label}</p>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={`rounded-lg border px-3 py-2 font-mono text-xs font-bold transition-colors ${
              value === opt ? "border-accent bg-accent text-bg" : "border-border bg-card text-muted hover:text-text"
            }`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

function ToggleRow({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <span className="font-mono text-xs text-text">{label}</span>
      <div className="flex gap-2">
        {[true, false].map((opt) => (
          <button
            key={String(opt)}
            type="button"
            onClick={() => onChange(opt)}
            className={`rounded-md border px-3 py-1.5 font-mono text-[10px] font-bold transition-colors ${
              value === opt ? "border-accent bg-accent text-bg" : "border-border text-muted hover:text-text"
            }`}
          >
            {opt ? "YES" : "NO"}
          </button>
        ))}
      </div>
    </div>
  );
}

// One-time gate rendered by AuthGate.tsx in place of the app when
// onboardingCompleted is false -- same "answer once, always-editable
// later from Profile" shape risk-tolerance already has, just gated on
// first login instead of always-optional (see the approved plan).
export default function OnboardingForm({
  onComplete,
  error,
}: {
  onComplete: (answers: Answers) => Promise<boolean>;
  error: string | null;
}) {
  const [riskTolerance, setRiskTolerance] = useState<string>("Moderate");
  const [investmentGoal, setInvestmentGoal] = useState<string>("");
  const [investmentHorizon, setInvestmentHorizon] = useState<string>("");
  const [interestedInCrypto, setInterestedInCrypto] = useState(false);
  const [interestedInRealEstate, setInterestedInRealEstate] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = riskTolerance && investmentGoal && investmentHorizon && !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    await onComplete({ riskTolerance, investmentGoal, investmentHorizon, interestedInCrypto, interestedInRealEstate });
    setSubmitting(false);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-5 py-10">
      <div className="w-full max-w-sm">
        <div className="pb-5 text-center">
          <p className="font-mono text-base font-bold tracking-wide text-text">A FEW QUESTIONS FIRST</p>
          <p className="mt-1.5 text-xs text-muted">
            Helps tailor what FinSight surfaces for you. Answer once now -- everything here stays editable later from your Profile.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <OptionRow label="RISK TOLERANCE" options={RISK_LEVELS} value={riskTolerance} onChange={setRiskTolerance} />
          <OptionRow label="PRIMARY GOAL" options={GOALS} value={investmentGoal} onChange={setInvestmentGoal} />
          <OptionRow label="TIME HORIZON" options={HORIZONS} value={investmentHorizon} onChange={setInvestmentHorizon} />
          <ToggleRow label="Interested in crypto?" value={interestedInCrypto} onChange={setInterestedInCrypto} />
          <ToggleRow label="Interested in real estate?" value={interestedInRealEstate} onChange={setInterestedInRealEstate} />

          {error && (
            <div className="rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-danger">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full rounded-lg bg-accent py-2.5 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "..." : "CONTINUE"}
          </button>
        </form>
      </div>
    </div>
  );
}
