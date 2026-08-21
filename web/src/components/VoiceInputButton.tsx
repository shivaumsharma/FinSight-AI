"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { GetUserMediaTimeoutError, getUserMediaWithTimeout } from "@/lib/getUserMediaWithTimeout";

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
// 2600ms, not 1600ms -- found live in a real voice session: 1.6s cut
// a user off mid-thought after "Okay, so..." (a natural pause while
// forming the rest of a sentence), sending an incomplete utterance as
// its own turn instead of waiting for the actual question. A longer
// hold costs a bit of latency on a genuinely finished sentence, which
// is a much smaller cost than losing the rest of what someone meant
// to say.
const SILENCE_HOLD_MS = 2600;
// A single loud sample (a mic bump, a cough, ambient noise) must not
// latch hasSpokenRef on its own -- found live in a real voice session:
// a session's auto-re-armed mic picked up a brief noise blip with no
// one actually speaking, and Sarvam hallucinated a plausible-sounding
// word ("Hello") out of what was functionally silence. Requiring the
// signal to stay continuously above SILENCE_RMS_THRESHOLD for this
// long filters out a blip while still being far under normal
// word-length, so real speech onset is still caught quickly.
const MIN_VOICED_MS = 300;

export type VoiceState = "idle" | "recording" | "transcribing" | "error";

export interface VoiceInputHandle {
  // Imperative start, for a caller that needs to arm the mic without a
  // click -- the voice-session loop (chat/page.tsx) re-arms it itself
  // right after a spoken reply finishes, which is never a click event.
  start: () => void;
  // Imperative stop, for a caller ending a session mid-recording (the
  // user hit "end session" while the mic was still listening).
  stop: () => void;
}

// Sarvam's /speech-to-text endpoint rejects webm/opus outright (only
// wav/mp3/aac/... are accepted) -- but MediaRecorder can't record
// directly to any of those, webm/opus is the only format every browser
// actually supports recording to. So: decode the recorded blob via the
// Web Audio API and re-encode it as a plain 16-bit PCM WAV file before
// upload. This also sidesteps the webm-duration-header problem
// (fixWebmDuration existed only to patch that) since a WAV header
// always carries its own correct length.
async function blobToWavBlob(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const audioContext = new AudioContextCtor();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return encodeWav(audioBuffer);
  } finally {
    void audioContext.close();
  }
}

// Minimal RIFF/PCM encoder -- mono/stereo 16-bit little-endian samples,
// downmixed to mono (Sarvam's STT has no use for stereo and it halves
// upload size). Standard 44-byte WAV header, no compression, no
// external library needed for this well-known format.
function encodeWav(audioBuffer: AudioBuffer): Blob {
  const numFrames = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;

  const mono = new Float32Array(numFrames);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < numFrames; i++) mono[i] += channelData[i] / numChannels;
  }

  const bytesPerSample = 2;
  const dataSize = numFrames * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset: number, s: string) {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    const sample = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export function MicIcon({ active }: { active: boolean }) {
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

// Peer of the question <input> in page.tsx -- by default only ever
// calls onTranscript (page.tsx's setQuery), never submits/runs
// anything itself, so a misheard word costs nothing more than the
// Sarvam call and the user reviews/edits before hitting submit
// themselves. The voice-session loop (chat/page.tsx) is the one
// caller that treats onTranscript as an auto-submit signal instead --
// that's a decision the caller makes, this component's own contract
// doesn't change.
const VoiceInputButton = forwardRef<VoiceInputHandle, {
  onTranscript: (text: string) => void;
  disabled?: boolean;
  // Fired the instant recording actually starts (mic permission
  // granted, MediaRecorder live) -- lets a caller with its own audio
  // playback (e.g. spoken chat replies) implement barge-in: opening
  // the mic cuts off whatever's currently speaking. Optional so every
  // existing caller (page.tsx, chat/page.tsx's text-fill usage) is
  // unaffected.
  onRecordingStart?: () => void;
  // Fired whenever the internal idle/recording/transcribing/error
  // state changes -- lets a caller (the voice-session loop) render one
  // unified status indicator that spans both this component's own
  // recording state and its own TTS-playback state, without
  // duplicating this component's state machine.
  onStateChange?: (state: VoiceState) => void;
}>(function VoiceInputButton({ onTranscript, disabled, onRecordingStart, onStateChange }, ref) {
  const [state, setState] = useState<VoiceState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Silence-detection plumbing -- torn down in stopSilenceWatch, called
  // from both the normal onstop path and the error path, so a denied
  // permission or a mid-recording failure can never leave an
  // AudioContext or rAF loop running in the background.
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const hasSpokenRef = useRef(false);
  const lastLoudAtRef = useRef(0);
  // Tracks how long the signal has been continuously above threshold,
  // reset on any dip below it -- hasSpokenRef only latches true once
  // this streak clears MIN_VOICED_MS, not on a single loud sample.
  const voicedStreakMsRef = useRef(0);
  const lastTickAtRef = useRef(0);

  useImperativeHandle(ref, () => ({ start: () => void startRecording(), stop: stopRecording }), []);

  useEffect(() => {
    onStateChange?.(state);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

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
    lastTickAtRef.current = performance.now();
    voicedStreakMsRef.current = 0;
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const deviation = (data[i] - 128) / 128;
        sumSquares += deviation * deviation;
      }
      const rms = Math.sqrt(sumSquares / data.length);
      const now = performance.now();
      const deltaMs = now - lastTickAtRef.current;
      lastTickAtRef.current = now;

      if (rms > SILENCE_RMS_THRESHOLD) {
        lastLoudAtRef.current = now;
        voicedStreakMsRef.current += deltaMs;
        if (voicedStreakMsRef.current >= MIN_VOICED_MS) hasSpokenRef.current = true;
      } else {
        voicedStreakMsRef.current = 0;
        if (hasSpokenRef.current && now - lastLoudAtRef.current > SILENCE_HOLD_MS) {
          stopRecording();
          return; // stopRecording() -> onstop tears this loop down
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  async function startRecording() {
    setErrorMessage(null);
    try {
      const stream = await getUserMediaWithTimeout();
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
        // Sarvam's STT endpoint rejects webm/opus outright -- decode
        // and re-encode as WAV (see blobToWavBlob above) before upload.
        void blobToWavBlob(rawBlob)
          .then((wavBlob) => transcribe(wavBlob))
          .catch(() => {
            setState("error");
            setErrorMessage("Couldn't process the recording. Please try again or type your question.");
          });
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setState("recording");
      onRecordingStart?.();
      stopTimerRef.current = setTimeout(() => {
        stopRecording();
        // Escape hatch: onstop should fire within milliseconds of a
        // successful stop() call. If the recorder is somehow still not
        // "inactive" shortly after (a swallowed exception from a
        // double-stop race with the silence-detector's own stopRecording()
        // call, or a stuck driver), there was previously no way out of
        // this except the user manually tapping cancel -- the UI would
        // otherwise sit on "LISTENING..." forever with the mic still live.
        window.setTimeout(() => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
            forceStopCleanup();
          }
        }, 3000);
      }, MAX_RECORDING_MS);
      watchForSilence();
    } catch (err) {
      stopSilenceWatch();
      setState("error");
      setErrorMessage(
        err instanceof GetUserMediaTimeoutError
          ? "Didn't get a response to the microphone permission prompt. Check for a popup near your browser's address bar, allow access, and try again."
          : "Microphone permission denied. Enable it in your browser settings to use voice input."
      );
    }
  }

  function forceStopCleanup() {
    stopSilenceWatch();
    if (stopTimerRef.current) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    mediaRecorderRef.current?.stream?.getTracks().forEach((t) => t.stop());
    setState("error");
    setErrorMessage("Voice recording didn't finish cleanly. Please try again or type your question.");
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    try {
      recorder.stop();
    } catch {
      // stop() threw synchronously -- onstop will never fire for this
      // call, so run the same cleanup onstop would have done rather than
      // leaving the UI stuck showing "LISTENING..." indefinitely.
      forceStopCleanup();
    }
  }

  async function transcribe(blob: Blob) {
    setState("transcribing");
    try {
      const formData = new FormData();
      formData.set("file", blob, "recording.wav");
      // AbortSignal.timeout, not a bare fetch -- confirmed live: this
      // request has NO timeout otherwise, and if the backend is ever
      // slow or unresponsive (a real thing that happened during a live
      // AWS deploy issue), the UI hangs on "THINKING..." forever with
      // no recovery at all -- the exact same class of stuck-forever bug
      // MAX_RECORDING_MS's own watchdog fixes for the recording phase,
      // just one step later in the pipeline. 35s, not 25s: comfortably
      // past Sarvam's own documented 30s synchronous STT cap (see
      // app/data/sarvam_client.py) so a merely-slow-but-working
      // transcription isn't cut off right as it would have succeeded.
      const resp = await fetch("/api/voice/transcribe", {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(35_000),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.message || "Voice transcription failed. Please try again or type your question.");
      onTranscript(data.transcript);
      setState("idle");
    } catch (e) {
      setState("error");
      const timedOut = e instanceof Error && e.name === "TimeoutError";
      setErrorMessage(timedOut
        ? "Voice transcription timed out. Please try again or type your question."
        : e instanceof Error ? e.message : "Voice transcription failed. Please try again or type your question.");
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
});

export default VoiceInputButton;
