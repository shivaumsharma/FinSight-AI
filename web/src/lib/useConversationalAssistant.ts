"use client";

import { useEffect, useRef, useState } from "react";
import type { VoiceInputHandle, VoiceState } from "@/components/VoiceInputButton";
import type { ChatMessage } from "@/lib/types";

const VOICE_MODE_STORAGE_KEY = "finsight-voice-mode";
const MAX_MIC_ERROR_RETRIES = 2;

// All the state and behavior behind the conversational assistant --
// text chat, spoken replies, and the hands-free voice-session loop --
// with zero JSX of its own. Extracted so the same logic can back two
// different layouts: the full-page /chat view (message thread, sticky
// input bar) and a compact card embedded on Home (active the instant
// the app opens, not buried behind a tab switch). Every quirk fixed
// live in a real voice session lives here once, not duplicated per
// caller: the barge-in promise-resolution fix, the mic-error auto-
// retry cap, the "re-arm only after playback truly finishes" ordering.
export function useConversationalAssistant() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  // Distinct from "loaded, zero messages" -- a failed fetch must not
  // look identical to a genuinely new conversation (see the /chat
  // page's own empty-state copy, which reads this to decide between
  // "couldn't load your history" and "try asking...").
  const [historyError, setHistoryError] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [micState, setMicState] = useState<VoiceState>("idle");
  const voiceInputRef = useRef<VoiceInputHandle>(null);
  // Mirrors sessionActive for use inside async callbacks (audio.onended,
  // the post-reply re-arm) that would otherwise close over a stale
  // `false` from whichever render scheduled them.
  const sessionActiveRef = useRef(false);
  // Not React state -- swapped/torn down imperatively by
  // synthesizeAndSpeak/stopSpeaking below, a render on every play/pause
  // would be wasted work here.
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  // The pending synthesizeAndSpeak() promise's own resolve -- stored so
  // stopSpeaking() can settle it when playback is cut short externally
  // (barge-in), not just when audio.onended fires naturally. Without
  // this, an interrupted session's await never resolves and the mic
  // never re-arms.
  const speakResolveRef = useRef<(() => void) | null>(null);
  // Consecutive failed-transcription count within the current session --
  // found live: VoiceInputButton's onTranscript only fires on SUCCESS,
  // so a failed transcribe() call (network blip, Sarvam hiccup) never
  // reached handleTranscript/sendMessage at all, meaning the mic
  // re-arm logic (which lives inside sendMessage) never ran either --
  // the session just silently stalled in "error" state until someone
  // manually tapped to end it. Retried automatically now, capped so a
  // genuinely denied mic permission doesn't spin forever.
  const micErrorStreakRef = useRef(0);

  useEffect(() => {
    fetch("/api/chat/history")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => setMessages(data.messages ?? []))
      .catch(() => setHistoryError(true))
      .finally(() => setHistoryLoaded(true));
    // Voice mode is a standing preference, not per-message -- read once
    // on mount (deliberately after the initial render, not via
    // useState's initializer, so server-rendered and first-client-
    // rendered markup match; localStorage doesn't exist during SSR).
    setVoiceMode(localStorage.getItem(VOICE_MODE_STORAGE_KEY) === "1");
    return () => stopSpeaking();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function stopSpeaking() {
    audioRef.current?.pause();
    audioRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setSpeaking(false);
    // Settle synthesizeAndSpeak()'s pending await if there is one --
    // covers both natural end (already resolved, this is a no-op) and
    // an external interruption (barge-in) that would otherwise leave
    // it pending forever and the mic never re-arming.
    if (speakResolveRef.current) {
      speakResolveRef.current();
      speakResolveRef.current = null;
    }
  }

  function toggleVoiceMode() {
    const next = !voiceMode;
    setVoiceMode(next);
    localStorage.setItem(VOICE_MODE_STORAGE_KEY, next ? "1" : "0");
    if (!next) stopSpeaking();
  }

  function startSession() {
    sessionActiveRef.current = true;
    setSessionActive(true);
    setError(null);
    micErrorStreakRef.current = 0;
    voiceInputRef.current?.start();
  }

  function endSession() {
    sessionActiveRef.current = false;
    setSessionActive(false);
    voiceInputRef.current?.stop();
    stopSpeaking();
  }

  // VoiceInputButton's own state, surfaced here for both the status
  // readout AND automatic recovery: a failed transcription would
  // otherwise strand an active session with nothing to re-arm the mic
  // (see micErrorStreakRef's own comment above). A short delay before
  // retrying keeps a genuinely denied permission from hammering
  // getUserMedia in a tight loop.
  function handleMicStateChange(newState: VoiceState) {
    setMicState(newState);
    if (newState !== "error" || !sessionActiveRef.current) return;

    micErrorStreakRef.current += 1;
    if (micErrorStreakRef.current <= MAX_MIC_ERROR_RETRIES) {
      setTimeout(() => {
        if (sessionActiveRef.current) voiceInputRef.current?.start();
      }, 1200);
    } else {
      setError("Voice session paused -- couldn't hear you a few times in a row. Tap Start Voice Session to try again.");
      endSession();
    }
  }

  // Returns only once playback has genuinely finished (ended, errored,
  // or play() itself was rejected/blocked) -- the session loop awaits
  // this before re-arming the mic, so listening never starts while a
  // reply is still audibly playing.
  async function synthesizeAndSpeak(text: string): Promise<void> {
    stopSpeaking();
    try {
      // AbortSignal.timeout -- without it, a hung/slow backend leaves
      // this await pending forever, which blocks sendMessage()'s own
      // mic re-arm (voiceInputRef.current?.start() below it) for the
      // rest of the session. Same fix as VoiceInputButton.tsx's own
      // transcribe() call and OnboardingForm.tsx's speak(), just for
      // this third caller of the same synthesize endpoint.
      const resp = await fetch("/api/voice/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: AbortSignal.timeout(20_000),
      });
      // Silent no-op on failure -- the text reply is already on screen
      // and fully usable; a broken TTS call should never block or
      // visibly disrupt the actual conversation, same non-blocking
      // convention as every other best-effort feature in this app
      // (news fetch, sector lookup, etc).
      if (!resp.ok) return;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      setSpeaking(true);
      await new Promise<void>((resolve) => {
        speakResolveRef.current = resolve;
        audio.onended = () => stopSpeaking();
        audio.onerror = () => stopSpeaking();
        audio.play().catch(() => stopSpeaking());
      });
    } catch {
      // fall through to the cleanup below regardless
    } finally {
      stopSpeaking();
    }
  }

  async function sendMessage(message: string) {
    if (!message || sending) return;

    setError(null);
    setSending(true);
    // Optimistic user bubble -- the server persists the real row, but
    // waiting for the round-trip before showing what was just typed/
    // said would make every message feel laggy.
    const optimisticUser: ChatMessage = {
      id: `local-${Date.now()}`, role: "user", content: message, intent: null, ticker: null, created_at: Date.now() / 1000,
    };
    setMessages((prev) => [...prev, optimisticUser]);

    try {
      // AbortSignal.timeout -- same reasoning as synthesizeAndSpeak()'s
      // own fix above, applied to the actual chat call itself: without
      // it, a hung backend leaves `sending` true forever, and the mic
      // re-arm at the bottom of this function never runs, stranding a
      // hands-free session with no way to continue except manually
      // ending it. 45s, not 20s -- this call runs real classification
      // AND reply-generation LLM calls back to back (see chat_router.py),
      // not just one short call like synthesize/classify-answer above.
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal: AbortSignal.timeout(45_000),
      });
      if (!resp.ok) throw new Error("chat request failed");
      const data = await resp.json();
      const assistantMessage: ChatMessage = {
        id: `local-${Date.now()}-reply`, role: "assistant", content: data.reply,
        intent: data.intent, ticker: data.ticker, created_at: Date.now() / 1000,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      if (voiceMode || sessionActiveRef.current) await synthesizeAndSpeak(data.reply);
    } catch {
      setError("Couldn't reach the assistant. Try again.");
    } finally {
      setSending(false);
    }

    // Re-arm the mic for the next turn -- only after the reply has
    // fully finished speaking (synthesizeAndSpeak above already
    // awaited that), and only if the user hasn't ended the session in
    // the meantime.
    if (sessionActiveRef.current) voiceInputRef.current?.start();
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    const message = input.trim();
    if (!message) return;
    setInput("");
    await sendMessage(message);
  }

  // In a normal (non-session) tap, a transcript just fills the input
  // for review -- the existing "misheard word costs nothing" contract.
  // Inside a voice session, the whole point is hands-free, so it
  // auto-submits instead.
  function handleTranscript(text: string) {
    micErrorStreakRef.current = 0; // a successful transcription -- the session is healthy again
    if (sessionActiveRef.current) void sendMessage(text);
    else setInput(text);
  }

  return {
    messages, historyLoaded, historyError, input, setInput, sending, error,
    voiceMode, toggleVoiceMode, speaking, sessionActive, micState,
    voiceInputRef, startSession, endSession, handleMicStateChange,
    stopSpeaking, handleSend, handleTranscript, synthesizeAndSpeak,
  };
}
