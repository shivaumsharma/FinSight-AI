"use client";

import { useRef, useState } from "react";
import VoiceInputButton, { MicIcon, type VoiceInputHandle, type VoiceState } from "@/components/VoiceInputButton";

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

type FieldKey = "risk_tolerance" | "investment_goal" | "investment_horizon" | "interested_in_crypto" | "interested_in_real_estate";

// Spoken prompts, deliberately reading the option list aloud each time
// (unlike the click UI, which shows them as buttons) -- a spoken
// question with no visible options needs to say what's valid to answer.
const VOICE_QUESTIONS: { field: FieldKey; prompt: string }[] = [
  { field: "risk_tolerance", prompt: "First, what's your risk tolerance? Conservative, moderate, or aggressive?" },
  { field: "investment_goal", prompt: "What's your primary investment goal? Wealth growth, retirement, income, or capital preservation?" },
  { field: "investment_horizon", prompt: "What's your time horizon? Short-term, under three years; medium, three to seven years; or long-term, over seven years?" },
  { field: "interested_in_crypto", prompt: "Are you interested in cryptocurrency? Yes or no." },
  { field: "interested_in_real_estate", prompt: "Last one -- are you interested in real estate? Yes or no." },
];
const MAX_RETRIES_PER_QUESTION = 1;
const VOICE_INTRO = "Hi, I'm FinSight. I've got five quick questions to help tailor things for you.";

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

function voiceStatusLabel(index: number, phase: "speaking" | "listening" | "thinking" | "error", micState: VoiceState): string {
  if (phase === "error" || micState === "error") return "DIDN'T CATCH THAT -- ONE MORE TRY";
  if (phase === "speaking") return "ASKING...";
  if (phase === "thinking" || micState === "transcribing") return "THINKING...";
  if (phase === "listening" && micState === "recording") return "LISTENING...";
  return `QUESTION ${index + 1} OF ${VOICE_QUESTIONS.length}`;
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

  const [voiceActive, setVoiceActive] = useState(false);
  const [voiceStep, setVoiceStep] = useState(0);
  const [voicePhase, setVoicePhase] = useState<"speaking" | "listening" | "thinking" | "error">("speaking");
  const [micState, setMicState] = useState<VoiceState>("idle");
  const voiceInputRef = useRef<VoiceInputHandle>(null);
  const retryCountRef = useRef(0);
  // Voice mode fills these via ref (not the individual useState setters
  // above) while it's running, so the final onComplete() call always
  // has the freshly-collected value even though setState updates
  // haven't necessarily flushed to a re-render yet.
  const voiceAnswersRef = useRef<Partial<Record<FieldKey, string>>>({});

  const canSubmit = riskTolerance && investmentGoal && investmentHorizon && !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    await onComplete({ riskTolerance, investmentGoal, investmentHorizon, interestedInCrypto, interestedInRealEstate });
    setSubmitting(false);
  }

  async function speak(text: string): Promise<void> {
    try {
      // AbortSignal.timeout -- without it, a hung/slow backend leaves
      // this await pending forever, which blocks askVoiceQuestion()
      // from ever reaching voiceInputRef.current?.start() below: the
      // exact same class of stuck-forever bug as VoiceInputButton.tsx's
      // own transcribe() call, just on the speaking side instead of
      // the listening side of the same voice loop.
      const resp = await fetch("/api/voice/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: AbortSignal.timeout(20_000),
      });
      if (!resp.ok) return; // silent no-op -- the question can still be answered unheard is unlikely but not fatal
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      await new Promise<void>((resolve) => {
        audio.onended = () => resolve();
        audio.onerror = () => resolve();
        audio.play().catch(() => resolve());
      });
      URL.revokeObjectURL(url);
    } catch {
      // fall through -- proceed to listening regardless
    }
  }

  function applyVoiceAnswer(field: FieldKey, value: string) {
    voiceAnswersRef.current[field] = value;
    switch (field) {
      case "risk_tolerance": setRiskTolerance(value); break;
      case "investment_goal": setInvestmentGoal(value); break;
      case "investment_horizon": setInvestmentHorizon(value); break;
      case "interested_in_crypto": setInterestedInCrypto(value === "Yes"); break;
      case "interested_in_real_estate": setInterestedInRealEstate(value === "Yes"); break;
    }
  }

  async function askVoiceQuestion(index: number) {
    setVoicePhase("speaking");
    retryCountRef.current = 0;
    await speak(VOICE_QUESTIONS[index].prompt);
    setVoicePhase("listening");
    voiceInputRef.current?.start();
  }

  async function handleVoiceTranscript(text: string) {
    setVoicePhase("thinking");
    const { field } = VOICE_QUESTIONS[voiceStep];

    let value: string | null = null;
    try {
      // AbortSignal.timeout -- same reasoning as speak()'s own fix
      // above: without it, a hung backend leaves this stuck on
      // "THINKING..." forever instead of falling through to the
      // retry-then-manual-fallback path below.
      const resp = await fetch("/api/onboarding/classify-answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ field, answer: text }),
        signal: AbortSignal.timeout(20_000),
      });
      const data = await resp.json();
      value = resp.ok ? data.value : null;
    } catch {
      value = null;
    }

    if (value) {
      applyVoiceAnswer(field, value);
      const next = voiceStep + 1;
      if (next < VOICE_QUESTIONS.length) {
        setVoiceStep(next);
        void askVoiceQuestion(next);
      } else {
        // All five answered by voice -- submit immediately, same as
        // the manual form's own submit button, using the ref (not
        // React state, which may not have flushed this last answer
        // into a render yet).
        const a = voiceAnswersRef.current;
        setSubmitting(true);
        await onComplete({
          riskTolerance: a.risk_tolerance!,
          investmentGoal: a.investment_goal!,
          investmentHorizon: a.investment_horizon!,
          interestedInCrypto: a.interested_in_crypto === "Yes",
          interestedInRealEstate: a.interested_in_real_estate === "Yes",
        });
        setSubmitting(false);
        setVoiceActive(false);
      }
      return;
    }

    // Ambiguous answer -- retry once with a clarifying re-prompt, then
    // give up on voice for THIS question and drop back to the manual
    // click form (with everything collected so far already filled in)
    // rather than leaving the user stuck in a loop with no way out.
    if (retryCountRef.current < MAX_RETRIES_PER_QUESTION) {
      retryCountRef.current += 1;
      setVoicePhase("error");
      await speak(`Sorry, I didn't catch that. ${VOICE_QUESTIONS[voiceStep].prompt}`);
      setVoicePhase("listening");
      voiceInputRef.current?.start();
    } else {
      setVoiceActive(false);
    }
  }

  async function startVoiceOnboarding() {
    setVoiceActive(true);
    setVoiceStep(0);
    voiceAnswersRef.current = {};
    setVoicePhase("speaking");
    await speak(VOICE_INTRO);
    void askVoiceQuestion(0);
  }

  function stopVoiceOnboarding() {
    voiceInputRef.current?.stop();
    setVoiceActive(false);
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

        {voiceActive ? (
          <div className="space-y-4">
            <button
              type="button"
              onClick={stopVoiceOnboarding}
              className="flex w-full flex-col items-center gap-2 rounded-lg border border-accent bg-accent/10 px-4 py-6 font-mono text-xs font-bold text-accent"
            >
              <MicIcon active={micState === "recording"} />
              {voiceStatusLabel(voiceStep, voicePhase, micState)}
              <span className="mt-1 text-[10px] font-normal text-muted">tap to cancel and answer manually</span>
            </button>
            {/* Mounted off-screen, not unmounted -- keeps its recording/
                silence-detection state machine alive across the whole
                voice flow; this component's own imperative start/stop
                is the only thing driving it here, never a click. */}
            <div className="sr-only">
              <VoiceInputButton
                ref={voiceInputRef}
                onTranscript={(text) => void handleVoiceTranscript(text)}
                onStateChange={setMicState}
              />
            </div>
          </div>
        ) : (
          <>
            <button
              type="button"
              onClick={() => void startVoiceOnboarding()}
              className="mb-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-card py-2.5 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent"
            >
              <MicIcon active={false} />
              ANSWER BY VOICE INSTEAD
            </button>

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
          </>
        )}
      </div>
    </div>
  );
}
