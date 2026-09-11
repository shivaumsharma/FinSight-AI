"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { VoiceState } from "@/components/VoiceInputButton";
import { GetUserMediaTimeoutError, getUserMediaWithTimeout } from "@/lib/getUserMediaWithTimeout";

// Streams the mic directly to app/api/main.py's /v1/voice/conversation-listen
// relay for one conversational turn, using Sarvam's real-time STT
// instead of VoiceInputButton.tsx's record-a-whole-clip-then-upload
// batch flow. Backend half of this was built and unit-tested a while
// ago (app/data/sarvam_realtime_client.py's relay_for_conversation) but
// never had a frontend caller -- see this hook's own use in
// useConversationalAssistant.ts for where it's actually wired in.
//
// Deliberately a hook, not a rendered component like VoiceInputButton:
// it only ever needs to be driven programmatically by the hands-free
// session loop (start a turn, get the final transcript, repeat), never
// by a user's direct tap -- tap-to-talk stays on VoiceInputButton's own
// proven batch path unchanged, this is purely additive.
//
// Same PCM16/16kHz encoding as useWakeWordListener.ts (must match
// app/data/sarvam_realtime_client.py's connect() call exactly), and the
// same "send {wake_token: token} as the first message" contract even
// though this token was minted for the conversation purpose
// specifically -- see conversation_listen()'s own docstring in
// app/api/main.py for why the field name doesn't change.
const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 4096;

// One turn's ceiling if Sarvam's own VAD endpointing never fires a
// transcript.final at all -- e.g. the user never actually speaks after
// starting the turn. Same safety-net philosophy as VoiceInputButton's
// own MAX_RECORDING_MS, just via a stop-and-report-empty instead of a
// forced stop-and-transcribe (there's nothing to transcribe here, the
// backend does that as audio arrives).
const MAX_TURN_MS = 30_000;

export interface RealtimeVoiceInputHandle {
  start: () => void;
  stop: () => void;
}

export function useRealtimeVoiceInput(
  onTranscript: (text: string) => void,
  onRecordingStart?: () => void,
  onStateChange?: (state: VoiceState) => void
): RealtimeVoiceInputHandle & { state: VoiceState; errorMessage: string | null } {
  const [state, setStateRaw] = useState<VoiceState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const setState = useCallback(
    (next: VoiceState) => {
      setStateRaw(next);
      onStateChange?.(next);
    },
    [onStateChange]
  );

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const maxTurnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Guards against a turn already being torn down (final transcript
  // received, error, or explicit stop()) racing a late-arriving WS
  // event -- every teardown path is idempotent, but only the FIRST one
  // should actually fire onTranscript/state changes for this turn.
  const turnEndedRef = useRef(true);

  function teardownAudio() {
    if (maxTurnTimerRef.current) {
      clearTimeout(maxTurnTimerRef.current);
      maxTurnTimerRef.current = null;
    }
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
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = null; // tearing it down ourselves -- the close handler below shouldn't also react
      ws.close();
    }
  }

  function endTurn(finalText: string | null, errorMsg: string | null) {
    if (turnEndedRef.current) return;
    turnEndedRef.current = true;
    teardownAudio();
    if (errorMsg !== null) {
      setErrorMessage(errorMsg);
      setState("error");
    } else {
      setErrorMessage(null);
      setState("idle");
      if (finalText) onTranscript(finalText);
    }
  }

  const start = useCallback(async () => {
    if (!turnEndedRef.current) return; // a turn is already in progress
    turnEndedRef.current = false;
    setErrorMessage(null);

    try {
      const stream = await getUserMediaWithTimeout();
      streamRef.current = stream;

      const resp = await fetch("/api/voice/conversation-listen-token");
      if (!resp.ok) throw new Error("Couldn't get a voice-session token.");
      const { token, ws_url: wsUrl } = await resp.json();

      const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioContext = new AudioContextCtor({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // ScriptProcessorNode, not an AudioWorklet -- same tradeoff
      // useWakeWordListener.ts's own comment makes: deprecated but
      // universally supported, no separate module file to serve.
      const processor = audioContext.createScriptProcessor(CHUNK_SAMPLES, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
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
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ wake_token: token }));
      };
      ws.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        let data: { event?: string; text?: string; message?: string };
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        if (data.event === "ready") {
          setState("recording");
          onRecordingStart?.();
        } else if (data.event === "transcript.final") {
          endTurn((data.text ?? "").trim() || null, null);
        }
        // transcript.partial is intentionally not surfaced here -- this
        // hook's contract is "one turn in, one final transcript out",
        // matching VoiceInputButton's own onTranscript shape so the
        // session loop that drives both needs no special-casing. A
        // live partial-text readout is a real future enhancement, not
        // needed for wiring this endpoint up for the first time.
      };
      ws.onclose = () => {
        // A close with no transcript.final ever received (backend
        // hiccup, Sarvam session dropped) -- report it as a normal mic
        // error, same recovery path (auto-retry, capped) that
        // useConversationalAssistant.ts's handleMicStateChange already
        // has for VoiceInputButton's own errors.
        endTurn(null, "Voice connection dropped. Retrying...");
      };

      maxTurnTimerRef.current = setTimeout(() => {
        endTurn(null, "Didn't catch anything that time.");
      }, MAX_TURN_MS);
    } catch (err) {
      endTurn(
        null,
        err instanceof GetUserMediaTimeoutError
          ? "Didn't get a response to the microphone permission prompt. Check for a popup near your browser's address bar, allow access, and try again."
          : "Microphone permission denied. Enable it in your browser settings to use voice input."
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onTranscript, onRecordingStart]);

  const stop = useCallback(() => {
    endTurn(null, null);
  }, []);

  // Unmount cleanup -- nothing previously released the mic/AudioContext/
  // WebSocket if the owning component unmounted mid-turn (e.g. the user
  // navigates away via BottomNav while a voice session is live) rather
  // than calling stop() itself. Deliberately calls teardownAudio()
  // directly rather than endTurn(null, null): endTurn also calls
  // setState/setErrorMessage/onTranscript, which are pointless (and in
  // some React versions warn) on an already-unmounting component --
  // this only needs to release the real resources, not update state
  // nobody will see.
  useEffect(() => {
    return () => {
      if (!turnEndedRef.current) {
        turnEndedRef.current = true;
        teardownAudio();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { state, errorMessage, start, stop };
}
