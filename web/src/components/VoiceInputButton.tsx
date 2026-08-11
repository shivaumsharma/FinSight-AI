"use client";

import { useEffect, useRef, useState } from "react";

// Hard recording ceiling, not voice-activity-detection -- margin under
// Sarvam's synchronous STT endpoint's own 30s cap (see
// app/data/sarvam_client.py's module docstring). Hitting this auto-stops
// and transcribes normally, it is not treated as an error.
const MAX_RECORDING_MS = 25_000;

type VoiceState = "idle" | "recording" | "transcribing" | "error";

function MicIcon({ active }: { active: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" strokeLinecap="round" />
      <path d="M12 18v4" strokeLinecap="round" />
      {active && <path d="M8 22h8" strokeLinecap="round" />}
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth={2} strokeOpacity={0.25} />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
}

// Peer of the question <input> in page.tsx -- only ever calls
// onTranscript (page.tsx's setQuery), never submits/runs anything
// itself. A misheard word should cost nothing more than the Sarvam
// call; the user reviews/edits the filled-in text before hitting the
// existing submit button themselves.
export default function VoiceInputButton({
  onTranscript,
  disabled,
}: {
  onTranscript: (text: string) => void;
  disabled?: boolean;
}) {
  const [state, setState] = useState<VoiceState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setSupported(
      typeof window !== "undefined" &&
        !!navigator.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== "undefined"
    );
  }, []);

  async function startRecording() {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (stopTimerRef.current) {
          clearTimeout(stopTimerRef.current);
          stopTimerRef.current = null;
        }
        void transcribe(new Blob(chunksRef.current, { type: mimeType }));
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setState("recording");
      stopTimerRef.current = setTimeout(() => stopRecording(), MAX_RECORDING_MS);
    } catch {
      setState("error");
      setErrorMessage("Microphone permission denied. Enable it in your browser settings to use voice input.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  async function transcribe(blob: Blob) {
    setState("transcribing");
    try {
      const formData = new FormData();
      formData.set("file", blob, "recording.webm");
      const resp = await fetch("/api/voice/transcribe", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.message || "Voice transcription failed. Please try again or type your question.");
      onTranscript(data.transcript);
      setState("idle");
    } catch (e) {
      setState("error");
      setErrorMessage(e instanceof Error ? e.message : "Voice transcription failed. Please try again or type your question.");
    }
  }

  function handleClick() {
    if (state === "recording") stopRecording();
    else if (state === "idle" || state === "error") void startRecording();
  }

  if (!supported) {
    return (
      <span
        title="Voice input isn't supported in this browser."
        className="shrink-0 cursor-not-allowed text-dim opacity-50"
      >
        <MicIcon active={false} />
      </span>
    );
  }

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled || state === "transcribing"}
        title={state === "recording" ? "Stop recording" : "Speak your question"}
        className={`flex items-center justify-center rounded p-1 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
          state === "recording"
            ? "animate-pulse text-danger"
            : state === "error"
            ? "text-danger hover:text-text"
            : "text-dim hover:text-accent"
        }`}
      >
        {state === "transcribing" ? <SpinnerIcon /> : <MicIcon active={state === "recording"} />}
      </button>
      {errorMessage && (
        <p className="absolute right-0 top-full z-10 mt-1 w-56 rounded border border-red-900/60 bg-red-950/40 px-2 py-1.5 font-mono text-[10px] leading-snug text-danger">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
