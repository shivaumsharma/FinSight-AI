"use client";

import { useEffect, useRef, useState } from "react";
import { MicIcon } from "@/components/VoiceInputButton";
import type { useWakeWordListener, WakeWordState } from "@/lib/useWakeWordListener";

function labelFor(state: WakeWordState): string {
  if (state === "listening") return 'Listening for "Hey FinSight"';
  if (state === "connecting") return "Connecting...";
  if (state === "error") return "Voice agent error -- tap to retry";
  return "Voice agent";
}

// Lives in the page header (page.tsx), not buried in the HomeAssistant
// card -- moved here specifically so it's visible without first
// scrolling to or interacting with the assistant widget. Enabling it
// opens a brief confirmation popup rather than connecting on the first
// click: this is a persistent mic stream to a cloud STT provider, not
// a one-shot tap-to-talk, so it deserves a beat of "yes, actually turn
// this on" before it starts. Turning it back OFF is a single click,
// no popup -- there's nothing to confirm about stopping.
export default function VoiceAgentToggle({
  wakeWord,
}: {
  wakeWord: ReturnType<typeof useWakeWordListener>;
}) {
  const [showConfirm, setShowConfirm] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showConfirm) return;
    function onPointerDown(e: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setShowConfirm(false);
      }
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [showConfirm]);

  function handleClick() {
    if (wakeWord.state === "idle") {
      setShowConfirm((v) => !v);
    } else if (wakeWord.state === "error") {
      void wakeWord.start(); // already consented once -- just retry
    } else {
      wakeWord.stop();
    }
  }

  function confirmEnable() {
    setShowConfirm(false);
    void wakeWord.start();
  }

  const active = wakeWord.state === "listening" || wakeWord.state === "connecting";

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={handleClick}
        aria-label={labelFor(wakeWord.state)}
        title={labelFor(wakeWord.state)}
        className={`flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[9px] font-bold transition-colors ${
          wakeWord.state === "listening"
            ? "border-accent text-accent animate-pulse"
            : wakeWord.state === "connecting"
              ? "border-accent/60 text-accent/70"
              : wakeWord.state === "error"
                ? "border-danger/60 text-danger"
                : "border-border text-dim hover:text-muted"
        }`}
      >
        <MicIcon active={active} />
        {wakeWord.state === "listening" && <span className="hidden sm:inline">LISTENING</span>}
        {wakeWord.state === "connecting" && <span className="hidden sm:inline">CONNECTING</span>}
      </button>

      {showConfirm && (
        <div className="absolute right-0 z-20 mt-2 w-64 rounded-lg border border-border bg-card p-3 shadow-lg">
          <p className="font-mono text-[11px] leading-relaxed text-text">
            Enable voice agent mode? FinSight will listen in the background for &ldquo;Hey
            FinSight&rdquo; and start a voice session the moment it hears it.
          </p>
          <div className="mt-2.5 flex gap-2">
            <button
              type="button"
              onClick={confirmEnable}
              className="flex-1 rounded-md bg-accent py-1.5 font-mono text-[10px] font-bold text-bg"
            >
              ENABLE
            </button>
            <button
              type="button"
              onClick={() => setShowConfirm(false)}
              className="flex-1 rounded-md border border-border py-1.5 font-mono text-[10px] font-bold text-muted hover:text-text"
            >
              NOT NOW
            </button>
          </div>
        </div>
      )}

      {wakeWord.state === "error" && wakeWord.errorMessage && (
        <div className="absolute right-0 z-20 mt-2 w-64 rounded-lg border border-danger/40 bg-card p-2.5">
          <p className="font-mono text-[10px] text-danger">{wakeWord.errorMessage}</p>
        </div>
      )}
    </div>
  );
}
