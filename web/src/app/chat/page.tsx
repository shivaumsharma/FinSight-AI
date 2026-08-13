"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import VoiceInputButton from "@/components/VoiceInputButton";
import type { ChatMessage } from "@/lib/types";

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
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/chat/history")
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((data) => setMessages(data.messages ?? []))
      .catch(() => {})
      .finally(() => setHistoryLoaded(true));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

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
    } catch {
      setError("Couldn't reach the assistant. Try again.");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="min-h-screen bg-bg pb-36">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <h1 className="font-mono text-lg font-bold text-text">Chat</h1>
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
          <VoiceInputButton onTranscript={setInput} disabled={sending} />
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
