"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { VoiceInputHandle, VoiceState } from "@/components/VoiceInputButton";
import type { ChatMessage } from "@/lib/types";
import { getNavigationTarget } from "@/lib/chatNavigation";
import { splitSentences } from "@/lib/splitSentences";

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
  const router = useRouter();
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
  // (barge-in), not just when the whole sentence pipeline finishes
  // naturally. Without this, an interrupted session's await never
  // resolves and the mic never re-arms.
  const speakResolveRef = useRef<(() => void) | null>(null);
  // Bumped on every stopSpeaking() call (natural end or external
  // interrupt alike) -- the sentence-pipelining loop below checks this
  // before every step and bails immediately if it's stale, so a
  // barge-in mid-reply can't have an old pipeline keep fetching/queuing
  // sentences after the user has already moved on.
  const speakSessionRef = useRef(0);
  // In-flight /api/voice/synthesize fetches, one per sentence currently
  // being synthesized -- stopSpeaking() aborts every one of these so a
  // barge-in doesn't keep burning Sarvam TTS calls for audio that will
  // never play.
  const pendingSynthesisRef = useRef<Set<AbortController>>(new Set());
  // Resolves the CURRENTLY PLAYING sentence's own wait -- stopSpeaking()
  // calls this too (alongside pausing the <audio> element) so an
  // external interrupt unblocks the pipeline loop immediately. pause()
  // alone never fires 'ended', so without this the loop would otherwise
  // hang forever waiting on an event that isn't coming.
  const sentenceDoneResolveRef = useRef<(() => void) | null>(null);
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
    speakSessionRef.current += 1; // invalidate any in-flight pipeline work, see its own comment above
    audioRef.current?.pause();
    audioRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    for (const controller of pendingSynthesisRef.current) controller.abort();
    pendingSynthesisRef.current.clear();
    if (sentenceDoneResolveRef.current) {
      sentenceDoneResolveRef.current();
      sentenceDoneResolveRef.current = null;
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

  // Fetches one sentence's audio -- AbortSignal.any combines a real
  // timeout (same 20s convention as every other synthesize caller) with
  // stopSpeaking()'s own abort-on-interrupt above, so a barge-in cancels
  // the actual network request instead of just ignoring its result.
  async function fetchSentenceAudio(sentence: string): Promise<Blob | null> {
    const controller = new AbortController();
    pendingSynthesisRef.current.add(controller);
    try {
      const resp = await fetch("/api/voice/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentence }),
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20_000)]),
      });
      if (!resp.ok) return null;
      return await resp.blob();
    } catch {
      return null; // network failure or abort -- caller skips this sentence, same non-blocking philosophy as the rest of this file
    } finally {
      pendingSynthesisRef.current.delete(controller);
    }
  }

  // Plays one sentence's audio to completion (or until stopSpeaking()
  // cuts it short via sentenceDoneResolveRef). Never rejects -- a
  // playback error degrades the same way a failed fetch does, by moving
  // on rather than breaking the whole reply over one bad clip.
  function playSentenceAudio(blob: Blob): Promise<void> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      const finish = () => {
        URL.revokeObjectURL(url);
        if (audioUrlRef.current === url) audioUrlRef.current = null;
        sentenceDoneResolveRef.current = null;
        resolve();
      };
      sentenceDoneResolveRef.current = finish;
      audio.onended = finish;
      audio.onerror = finish;
      audio.play().catch(finish);
    });
  }

  // Synthesizes and plays each sentence with lookahead=1: the instant
  // sentence i's audio is ready, sentence i+1's synthesis starts in the
  // background WHILE sentence i plays, instead of waiting for the
  // entire reply to become one clip before any of it is audible. Checks
  // speakSessionRef before every step so a mid-reply barge-in (which
  // bumps it via stopSpeaking()) stops this loop from queuing any more
  // work immediately, not just after the current sentence finishes.
  async function runSentencePipeline(sentences: string[], mySession: number) {
    let nextAudio = fetchSentenceAudio(sentences[0]);
    for (let i = 0; i < sentences.length; i++) {
      if (speakSessionRef.current !== mySession) return;
      const blob = await nextAudio;
      if (speakSessionRef.current !== mySession) return;
      if (i + 1 < sentences.length) nextAudio = fetchSentenceAudio(sentences[i + 1]);
      if (blob) await playSentenceAudio(blob);
      if (speakSessionRef.current !== mySession) return;
    }
    if (speakSessionRef.current === mySession) stopSpeaking(); // natural end of the whole reply -- resolves synthesizeAndSpeak()'s own await
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

  // Returns only once the whole reply has genuinely finished speaking
  // (naturally, or cut short by a barge-in) -- the session loop awaits
  // this before re-arming the mic, so listening never starts while a
  // reply is still audibly playing. Internally pipelines sentence-by-
  // sentence (see runSentencePipeline above) rather than waiting for
  // the entire reply to synthesize as one clip before anything plays --
  // a single-sentence reply (most short confirmations) behaves
  // identically to the old one-shot version, since there's only one
  // sentence to pipeline.
  async function synthesizeAndSpeak(text: string): Promise<void> {
    stopSpeaking();
    const mySession = speakSessionRef.current;
    const sentences = splitSentences(text);
    if (sentences.length === 0) return; // nothing to speak -- same silent no-op as the old empty/failed-fetch case

    setSpeaking(true);
    await new Promise<void>((resolve) => {
      speakResolveRef.current = resolve;
      void runSentencePipeline(sentences, mySession);
    });
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

    // Set inside the try block below (from the reply's own intent) and
    // read after it -- lets the end-of-function re-arm logic know
    // whether THIS turn was "end the session" without threading intent
    // through another ref, matching sessionActiveRef's existing
    // stale-closure-avoidance pattern.
    let endingSession = false;
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

      // Navigate to wherever this reply's result actually lives (e.g.
      // watchlist_add -> /watchlist) -- confirmed live: adding a stock
      // by voice succeeded server-side but never visibly landed
      // anywhere, reading as "did that even work?". end_voice_session
      // is handled below instead (it stops the mic, not the router),
      // so it's deliberately excluded from getNavigationTarget's map.
      if (data.intent === "end_voice_session") {
        endingSession = true;
      } else {
        const target = getNavigationTarget(data.intent, data.ticker);
        if (target) router.push(target);
      }

      if (voiceMode || sessionActiveRef.current) await synthesizeAndSpeak(data.reply);
    } catch {
      setError("Couldn't reach the assistant. Try again.");
    } finally {
      setSending(false);
    }

    // Re-arm the mic for the next turn -- only after the reply has
    // fully finished speaking (synthesizeAndSpeak above already
    // awaited that), and only if the user hasn't ended the session in
    // the meantime (either by tapping End, or by just asking the
    // assistant to end it -- same teardown either way).
    if (endingSession) {
      sessionActiveRef.current = false;
      setSessionActive(false);
      voiceInputRef.current?.stop();
    } else if (sessionActiveRef.current) {
      voiceInputRef.current?.start();
    }
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
