"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import VoiceInputButton from "@/components/VoiceInputButton";
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
  const bottomRef = useRef<HTMLDivElement>(null);
  // Not React state -- swapped/torn down imperatively by
  // synthesizeAndSpeak/stopSpeaking below, a render on every play/pause
  // would be wasted work here.
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);

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
  }

  function toggleVoiceMode() {
    const next = !voiceMode;
    setVoiceMode(next);
    localStorage.setItem(VOICE_MODE_STORAGE_KEY, next ? "1" : "0");
    if (!next) stopSpeaking();
  }

  async function synthesizeAndSpeak(text: string) {
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
      audio.onended = stopSpeaking;
      audio.onerror = stopSpeaking;
      setSpeaking(true);
      await audio.play();
    } catch {
      stopSpeaking();
    }
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    const message = input.trim();
    if (!message || sending) return;

    setInput("");
    setError(null);
    setSending(true);
    // Optimistic user bubble -- the server persists the real row, but
    // waiting for the round-trip before showing what was just typed
    // would make every message feel laggy.
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
      if (voiceMode) void synthesizeAndSpeak(data.reply);
    } catch {
      setError("Couldn't reach the assistant. Try again.");
    } finally {
      setSending(false);
    }
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
            placeholder="ask finsight..."
            disabled={sending}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3.5 py-2.5 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <VoiceInputButton onTranscript={setInput} disabled={sending} onRecordingStart={stopSpeaking} />
          <button
            type="submit"
            disabled={sending || !input.trim()}
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
