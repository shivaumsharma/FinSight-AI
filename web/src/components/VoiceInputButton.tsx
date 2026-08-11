"use client";

import { useEffect, useRef, useState } from "react";
import fixWebmDuration from "fix-webm-duration";

// Hard recording ceiling, not a substitute for silence detection below --
// margin under Sarvam's synchronous STT endpoint's own 30s cap (see
// app/data/sarvam_client.py's module docstring). Hitting this auto-stops
// and transcribes normally, it is not treated as an error.
const MAX_RECORDING_MS = 25_000;

// Silence-based auto-stop -- no more manual "click again to stop".
// getByteTimeDomainData returns 0-255 centered at 128; SILENCE_RMS
// is the RMS deviation from that midpoint, roughly 0-1 scale. 0.02 is
// an empirical middle ground: sensitive enough to catch a genuine
// trailing pause, tolerant enough of normal mic noise floor not to
// false-trigger mid-sentence. Auto-stop only arms AFTER the user has
// actually spoken at least once (hasSpokenRef) -- otherwise thinking
// silently for a moment before starting would cut the recording off
// before they've said anything.
const SILENCE_RMS_THRESHOLD = 0.02;
const SILENCE_HOLD_MS = 1600;

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
  const recordingStartRef = useRef(0);

  // Silence-detection plumbing -- torn down in stopSilenceWatch, called
  // from both the normal onstop path and the error path, so a denied
  // permission or a mid-recording failure can never leave an
  // AudioContext or rAF loop running in the background.
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const hasSpokenRef = useRef(false);
  const lastLoudAtRef = useRef(0);

  useEffect(() => {
    setSupported(
      typeof window !== "undefined" &&
        !!navigator.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== "undefined"
    );
    return () => stopSilenceWatch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopSilenceWatch() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
    analyserRef.current = null;
  }

  function watchForSilence() {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const deviation = (data[i] - 128) / 128;
        sumSquares += deviation * deviation;
      }
      const rms = Math.sqrt(sumSquares / data.length);
      const now = performance.now();

      if (rms > SILENCE_RMS_THRESHOLD) {
        hasSpokenRef.current = true;
        lastLoudAtRef.current = now;
      } else if (hasSpokenRef.current && now - lastLoudAtRef.current > SILENCE_HOLD_MS) {
        stopRecording();
        return; // stopRecording() -> onstop tears this loop down
      }

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  async function startRecording() {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      // Separate from MediaRecorder -- Web Audio API taps the same
      // MediaStream in parallel purely to measure volume for
      // auto-stop, it never touches the recorded bytes themselves.
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      hasSpokenRef.current = false;
      lastLoudAtRef.current = performance.now();

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        stopSilenceWatch();
        if (stopTimerRef.current) {
          clearTimeout(stopTimerRef.current);
          stopTimerRef.current = null;
        }
        const rawBlob = new Blob(chunksRef.current, { type: mimeType });
        const durationMs = performance.now() - recordingStartRef.current;
        // MediaRecorder's webm output has no duration in its header
        // (browsers write it incrementally and never know the final
        // length) -- Sarvam's API rejects that with a 400 indistinguishable
        // from genuinely malformed audio. fixWebmDuration patches the
        // real elapsed time into the blob's EBML header before upload.
        void fixWebmDuration(rawBlob, durationMs, { logger: false })
          .then((fixedBlob) => transcribe(fixedBlob))
          .catch(() => transcribe(rawBlob)); // fall back to the unfixed blob rather than losing the recording entirely
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      recordingStartRef.current = performance.now();
      setState("recording");
      stopTimerRef.current = setTimeout(() => stopRecording(), MAX_RECORDING_MS);
      watchForSilence();
    } catch {
      stopSilenceWatch();
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
        title={state === "recording" ? "Listening -- stops automatically, or click to stop now" : "Speak your question"}
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
