"use client";

import { useCallback, useRef, useState } from "react";

const SAMPLE_RATE = 16000; // must match app/data/sarvam_realtime_client.py's connect() call exactly
const CHUNK_SAMPLES = 4096; // ScriptProcessorNode buffer size, in samples at SAMPLE_RATE (~256ms/chunk)
// Cloud Run's own request timeout is 300s (5 minutes) for every
// endpoint on this backend, not just this one -- raising it service-
// wide to accommodate one feature would mean a genuinely stuck HTTP
// request elsewhere could now hang for much longer before Cloud Run
// kills it. Safer to keep that shared limit alone and make just this
// listener resilient to it: reconnect a good margin before the cutoff
// so it's never silently severed mid-listen with no visible sign why.
// The mic stream itself stays open across a reconnect (only the
// WebSocket is torn down and rebuilt), so this is invisible to the user.
const RECONNECT_BEFORE_MS = 4 * 60 * 1000;
const RECONNECT_AFTER_ERROR_MS = 2000;

export type WakeWordState = "idle" | "listening" | "error";

// Continuously streams the mic to the backend's /v1/voice/wake-listen
// relay (app/data/sarvam_realtime_client.py), watching for "Hey
// FinSight" without requiring a tap for every listen -- the one tap
// this DOES still require is start() itself, since no browser can
// grant microphone access without one (see the chat explanation this
// feature grew out of). Deliberately only detects the phrase; the
// caller decides what to do next (typically: start a normal voice
// session), same "detect only, hand off to the already-hardened batch
// pipeline for the actual command" split as the backend relay.
export function useWakeWordListener(onWakeDetected: (text: string) => void) {
  const [state, setState] = useState<WakeWordState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Not React state -- read synchronously inside the audio-processing
  // callback (fires many times/sec) and the WS close handler, where a
  // stale render's captured state value would be wrong.
  const stoppedRef = useRef(true);
  // True while a voice session (VoiceInputButton's own separate mic
  // stream) is actively recording/replying -- the listener stays
  // connected but stops SENDING audio, so it isn't burning Sarvam
  // streaming cost and can't hear itself over the session's own audio.
  const pausedRef = useRef(false);

  function teardownWs() {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = null; // about to close it ourselves -- don't let the reconnect-on-close handler fire
      ws.close();
    }
  }

  const openWs = useCallback(async () => {
    if (stoppedRef.current) return;
    try {
      const resp = await fetch("/api/voice/wake-listen-token");
      if (!resp.ok) throw new Error("Couldn't get a wake-listen token.");
      const { token, ws_url: wsUrl } = await resp.json();

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ wake_token: token }));
        setState("listening");
        setErrorMessage(null);
      };
      ws.onmessage = (event) => {
        if (typeof event.data !== "string") return; // audio never arrives this direction, only JSON
        try {
          const data = JSON.parse(event.data);
          if (data.event === "wake_detected") onWakeDetected(data.text ?? "");
        } catch {
          // ignore anything that isn't the JSON shape we expect
        }
      };
      ws.onclose = () => {
        if (stoppedRef.current) return;
        reconnectTimerRef.current = setTimeout(() => void openWs(), RECONNECT_AFTER_ERROR_MS);
      };

      reconnectTimerRef.current = setTimeout(() => {
        teardownWs();
        void openWs();
      }, RECONNECT_BEFORE_MS);
    } catch {
      if (stoppedRef.current) return;
      setState("error");
      setErrorMessage("Couldn't connect the wake-word listener. Retrying...");
      reconnectTimerRef.current = setTimeout(() => void openWs(), RECONNECT_AFTER_ERROR_MS);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onWakeDetected]);

  const start = useCallback(async () => {
    if (!stoppedRef.current) return; // already running
    stoppedRef.current = false;
    pausedRef.current = false;
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioContext = new AudioContextCtor({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // ScriptProcessorNode, not an AudioWorklet -- deprecated but
      // still universally supported and, unlike a worklet, needs no
      // separate module file served as a static asset. Good enough
      // for a mono 16kHz PCM tap; revisit only if main-thread jank
      // from onaudioprocess ever becomes a measurable problem.
      const processor = audioContext.createScriptProcessor(CHUNK_SAMPLES, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
        if (pausedRef.current) return;
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const sample = Math.max(-1, Math.min(1, input[i]));
          pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }
        ws.send(pcm.buffer);
      };
      source.connect(processor);
      // onaudioprocess only fires while the node is connected all the
      // way to a destination -- routing through a zero-gain node keeps
      // it running without playing the user's own mic back to them.
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      await openWs();
    } catch {
      stoppedRef.current = true;
      setState("error");
      setErrorMessage("Microphone permission denied. Enable it in your browser settings to use the wake-word listener.");
    }
  }, [openWs]);

  const stop = useCallback(() => {
    stoppedRef.current = true;
    teardownWs();
    processorRef.current?.disconnect();
    processorRef.current = null;
    silentGainRef.current?.disconnect();
    silentGainRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setState("idle");
  }, []);

  const pause = useCallback(() => {
    pausedRef.current = true;
  }, []);

  const resume = useCallback(() => {
    pausedRef.current = false;
  }, []);

  return { state, errorMessage, start, stop, pause, resume };
}
