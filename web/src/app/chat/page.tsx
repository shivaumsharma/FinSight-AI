"use client";

import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import VoiceInputButton, { MicIcon } from "@/components/VoiceInputButton";
import { useConversationalAssistant } from "@/lib/useConversationalAssistant";
import { sessionStatusLabel, SpeakerIcon } from "@/components/ConversationalAssistantUI";
import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/lib/types";

// The fast conversational assistant -- deliberately separate from the
// existing one-shot "Run Full Research Report" flow on the home page,
// which stays untouched (see app/reasoning/chat_router.py's own
// docstring for why: the full pipeline is fundamentally serialized,
// MAX_CONCURRENT_JOBS=1, so chat can't reuse it and stay fast). A
// full_report_request reply hands off to that existing flow via the
// same `?q=` pre-fill pattern the stock detail page's own link uses,
// rather than duplicating the job/poll logic here. All the state and
// session-loop logic lives in useConversationalAssistant -- the same
// hook backs the compact widget embedded on Home, this is just the
// full-page layout (full message thread, sticky input bar).
export default function ChatPage() {
  return (
    <AuthGate>
      {() => <ChatContent />}
    </AuthGate>
  );
}

function ChatContent() {
  const a = useConversationalAssistant();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [a.messages, a.sending]);

  return (
    <div className="min-h-screen bg-bg pb-36">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <div className="flex items-center justify-between">
          <h1 className="font-mono text-lg font-bold text-text">Chat</h1>
          <button
            type="button"
            onClick={a.toggleVoiceMode}
            aria-label={a.voiceMode ? "Turn off spoken replies" : "Turn on spoken replies"}
            title={a.voiceMode ? "Spoken replies on -- tap to turn off" : "Spoken replies off -- tap to turn on"}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[10px] font-bold transition-colors ${
              a.voiceMode ? "border-accent text-accent" : "border-border text-dim hover:text-muted"
            }`}
          >
            <SpeakerIcon speaking={a.speaking} />
            {a.speaking ? "SPEAKING" : a.voiceMode ? "VOICE ON" : "VOICE OFF"}
          </button>
        </div>
        <p className="mt-1 font-mono text-[10px] text-dim">
          Ask about your portfolio or a specific ticker. Informational only, not personalized investment advice.
        </p>

        {a.sessionActive ? (
          <button
            type="button"
            onClick={a.endSession}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-accent bg-accent/10 py-3 font-mono text-xs font-bold text-accent"
          >
            <MicIcon active={a.micState === "recording"} />
            {sessionStatusLabel(a.micState, a.sending, a.speaking)} -- TAP TO END SESSION
          </button>
        ) : (
          <button
            type="button"
            onClick={a.startSession}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-card py-3 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent"
          >
            <MicIcon active={false} />
            START VOICE SESSION
          </button>
        )}

        <div className="mt-6 flex flex-col gap-3">
          {a.historyLoaded && a.messages.length === 0 && (
            <p className="mt-4 text-center font-mono text-xs text-dim">
              Try &quot;what&apos;s my portfolio look like&quot; or &quot;what&apos;s going on with AAPL&quot;.
            </p>
          )}
          {a.messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {a.sending && (
            <div className="self-start rounded-lg border border-border bg-card px-3.5 py-2.5 font-mono text-xs text-muted">
              thinking&hellip;
            </div>
          )}
          {a.error && <p className="font-mono text-[10px] text-danger">{a.error}</p>}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="fixed inset-x-0 bottom-16 z-10 border-t border-border bg-bg/95 px-5 py-3 backdrop-blur">
        <form onSubmit={a.handleSend} className="mx-auto flex max-w-2xl items-center gap-2">
          <input
            type="text"
            value={a.input}
            onChange={(e) => a.setInput(e.target.value)}
            placeholder={a.sessionActive ? "voice session active..." : "ask finsight..."}
            disabled={a.sending || a.sessionActive}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-lg border border-border bg-card px-3.5 py-2.5 font-mono text-sm text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <VoiceInputButton
            ref={a.voiceInputRef}
            onTranscript={a.handleTranscript}
            // Not disabled while merely "speaking" (even though
            // `sending` is still true for that whole window, since
            // synthesizeAndSpeak is awaited before sending flips back
            // to false) -- a tap during playback is barge-in, and
            // disabling the button would make that unreachable.
            disabled={a.sending && !a.speaking}
            onRecordingStart={a.stopSpeaking}
            onStateChange={a.handleMicStateChange}
          />
          <button
            type="submit"
            disabled={a.sending || a.sessionActive || !a.input.trim()}
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
