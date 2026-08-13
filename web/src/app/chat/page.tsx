"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import VoiceInputButton, { MicIcon, type VoiceInputHandle, type VoiceState } from "@/components/VoiceInputButton";
import type { ChatMessage } from "@/lib/types";

const VOICE_MODE_STORAGE_KEY = "finsight-voice-mode";

function SpeakerIcon({ speaking }: { speaking: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <path d="M4 9v6h4l5 5V4L8 9H4z" strokeLinejoin="round" />
      {speaking && <path d="M17 8a6 6 0 0 1 0 8M20 5a10 10 0 0 1 0 14" strokeLinecap="round" />}
    </svg>
  );
}

// Status readout for an active voice session -- spans both
// VoiceInputButton's own recording state (via onStateChange) and this
// page's own TTS-playback state, since a session cycles through both.
function sessionStatusLabel(micState: VoiceState, sending: boolean, speaking: boolean): string {
  if (speaking) return "SPEAKING...";
  if (sending || micState === "transcribing") return "THINKING...";
  if (micState === "recording") return "LISTENING...";
  if (micState === "error") return "MIC ERROR";
  return "STARTING...";
}

// The fast conversational assistant -- deliberately separate from the
// existing one-shot "Run Full Research Report" flow on the home page,
// which stays untouched (see app/reasoning/chat_router.py's own
// docstring for why: the full pipeline is fundamentally serialized,
// MAX_CONCURRENT_JOBS=1, so chat can't reuse it and stay fast). A
// full_report_request reply hands off to that existing flow via the
// same `?q=` pre-fill pattern the stock detail page's own link uses,
// rather than duplicating the job/poll logic here.
export default function ChatPage() {
  return (
    <AuthGate>
      {() => <ChatContent />}
    </AuthGate>
  );
}

function ChatContent() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [micState, setMicState] = useState<VoiceState>("idle");
  const bottomRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    fetch("/api/chat/history")
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((data) => setMessages(data.messages ?? []))
      .catch(() => {})
      .finally(() => setHistoryLoaded(true));
    // Voice mode is a standing preference, not per-message -- read once
    // on mount (deliberately after the initial render, not via
    // useState's initializer, so server-rendered and first-client-
    // rendered markup match; localStorage doesn't exist during SSR).
    setVoiceMode(localStorage.getItem(VOICE_MODE_STORAGE_KEY) === "1");
    return () => stopSpeaking();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

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
    voiceInputRef.current?.start();
  }

  function endSession() {
    sessionActiveRef.current = false;
    setSessionActive(false);
    voiceInputRef.current?.stop();
    stopSpeaking();
  }

  // Returns only once playback has genuinely finished (ended, errored,
  // or play() itself was rejected/blocked) -- the session loop awaits
  // this before re-arming the mic, so listening never starts while a
  // reply is still audibly playing.
  async function synthesizeAndSpeak(text: string): Promise<void> {
    stopSpeaking();
    try {
      const resp = await fetch("/api/voice/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
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
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
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
    if (sessionActiveRef.current) void sendMessage(text);
    else setInput(text);
  }

  return (
    <div className="min-h-screen bg-bg pb-36">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <div className="flex items-center justify-between">
          <h1 className="font-mono text-lg font-bold text-text">Chat</h1>
          <button
            type="button"
            onClick={toggleVoiceMode}
            aria-label={voiceMode ? "Turn off spoken replies" : "Turn on spoken replies"}
            title={voiceMode ? "Spoken replies on -- tap to turn off" : "Spoken replies off -- tap to turn on"}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[10px] font-bold transition-colors ${
              voiceMode ? "border-accent text-accent" : "border-border text-dim hover:text-muted"
            }`}
          >
            <SpeakerIcon speaking={speaking} />
            {speaking ? "SPEAKING" : voiceMode ? "VOICE ON" : "VOICE OFF"}
          </button>
        </div>
        <p className="mt-1 font-mono text-[10px] text-dim">
          Ask about your portfolio or a specific ticker. Informational only, not personalized investment advice.
        </p>

        {sessionActive ? (
          <button
            type="button"
            onClick={endSession}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-accent bg-accent/10 py-3 font-mono text-xs font-bold text-accent"
          >
            <MicIcon active={micState === "recording"} />
            {sessionStatusLabel(micState, sending, speaking)} -- TAP TO END SESSION
          </button>
        ) : (
          <button
            type="button"
            onClick={startSession}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-card py-3 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent"
          >
            <MicIcon active={false} />
            START VOICE SESSION
          </button>
        )}

        <div className="mt-6 flex flex-col gap-3">
          {historyLoaded && messages.length === 0 && (
            <p className="mt-4 text-center font-mono text-xs text-dim">
              Try &quot;what&apos;s my portfolio look like&quot; or &quot;what&apos;s going on with AAPL&quot;.
            </p>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {sending && (
            <div className="self-start rounded-lg border border-border bg-card px-3.5 py-2.5 font-mono text-xs text-muted">
              thinking&hellip;
            </div>
          )}
          {error && <p className="font-mono text-[10px] text-danger">{error}</p>}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="fixed inset-x-0 bottom-16 z-10 border-t border-border bg-bg/95 px-5 py-3 backdrop-blur">
        <form onSubmit={handleSend} className="mx-auto flex max-w-2xl items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={sessionActive ? "voice session active..." : "ask finsight..."}
            disabled={sending || sessionActive}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3.5 py-2.5 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <VoiceInputButton
            ref={voiceInputRef}
            onTranscript={handleTranscript}
            // Not disabled while merely "speaking" (even though
            // `sending` is still true for that whole window, since
            // synthesizeAndSpeak is awaited before sending flips back
            // to false) -- a tap during playback is barge-in, and
            // disabling the button would make that unreachable.
            disabled={sending && !speaking}
            onRecordingStart={stopSpeaking}
            onStateChange={setMicState}
          />
          <button
            type="submit"
            disabled={sending || sessionActive || !input.trim()}
            className="rounded-lg bg-accent px-4 py-2.5 font-mono text-xs font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            SEND
          </button>
        </form>
      </div>

      <BottomNav />
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[85%] whitespace-pre-line rounded-lg px-3.5 py-2.5 font-mono text-xs ${
          isUser ? "bg-accent text-bg" : "border border-border bg-card text-text/90"
        }`}
      >
        {message.content}
      </div>
      {message.intent === "full_report_request" && message.ticker && (
        <Link
          href={`/?q=${encodeURIComponent(`Should I invest in ${message.ticker}?`)}`}
          className="mt-1.5 font-mono text-[10px] font-bold text-accent hover:underline"
        >
          RUN FULL RESEARCH REPORT &rarr;
        </Link>
      )}
    </div>
  );
}
