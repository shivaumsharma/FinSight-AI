"use client";

import Link from "next/link";
import VoiceInputButton, { MicIcon } from "@/components/VoiceInputButton";
import { useConversationalAssistant } from "@/lib/useConversationalAssistant";
import { sessionStatusLabel, SpeakerIcon } from "@/components/ConversationalAssistantUI";

// Compact conversational assistant, embedded on Home so talking to
// FinSight (text or voice) is active the instant the app opens --
// previously only reachable by first switching to the separate Chat
// tab. Same useConversationalAssistant hook as the full-page /chat
// view (identical session-loop behavior, no duplicated logic), just a
// smaller footprint: only the latest exchange shown inline, with a
// link to /chat for the full thread, since Home already has several
// other cards below this one.
export default function HomeAssistant() {
  const a = useConversationalAssistant();
  const lastUser = [...a.messages].reverse().find((m) => m.role === "user");
  const lastAssistant = [...a.messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="mt-4 rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] tracking-wide text-dim">TALK TO FINSIGHT</p>
        <div className="flex items-center gap-2">
          <Link href="/chat" className="font-mono text-[10px] font-bold text-muted hover:text-accent">
            FULL CONVERSATION &rarr;
          </Link>
          <button
            type="button"
            onClick={a.toggleVoiceMode}
            aria-label={a.voiceMode ? "Turn off spoken replies" : "Turn on spoken replies"}
            title={a.voiceMode ? "Spoken replies on -- tap to turn off" : "Spoken replies off -- tap to turn on"}
            className={`flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[9px] font-bold transition-colors ${
              a.voiceMode ? "border-accent text-accent" : "border-border text-dim hover:text-muted"
            }`}
          >
            <SpeakerIcon speaking={a.speaking} />
            {a.speaking ? "SPEAKING" : a.voiceMode ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {lastUser && lastAssistant && !a.sessionActive && (
        <div className="mt-2.5 flex flex-col gap-1.5 border-t border-border-subtle pt-2.5">
          <p className="font-mono text-[11px] text-muted">&gt; {lastUser.content}</p>
          <p className="line-clamp-3 font-mono text-xs text-text/90">{lastAssistant.content}</p>
        </div>
      )}

      {a.sessionActive ? (
        <button
          type="button"
          onClick={a.endSession}
          className="mt-2.5 flex w-full items-center justify-center gap-2 rounded-lg border border-accent bg-accent/10 py-2.5 font-mono text-[11px] font-bold text-accent"
        >
          <MicIcon active={a.micState === "recording"} />
          {sessionStatusLabel(a.micState, a.sending, a.speaking)} -- TAP TO END
        </button>
      ) : (
        <>
          <button
            type="button"
            onClick={a.startSession}
            className="mt-2.5 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg py-2 font-mono text-[11px] font-bold text-muted hover:border-accent hover:text-accent"
          >
            <MicIcon active={false} />
            START VOICE SESSION
          </button>

          <form onSubmit={a.handleSend} className="mt-2 flex items-center gap-2">
            <input
              type="text"
              value={a.input}
              onChange={(e) => a.setInput(e.target.value)}
              placeholder="or type a question..."
              disabled={a.sending}
              autoComplete="off"
              className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
            />
            <VoiceInputButton
              ref={a.voiceInputRef}
              onTranscript={a.handleTranscript}
              disabled={a.sending && !a.speaking}
              onRecordingStart={a.stopSpeaking}
              onStateChange={a.handleMicStateChange}
            />
            <button
              type="submit"
              disabled={a.sending || !a.input.trim()}
              className="shrink-0 rounded-lg bg-accent px-3 py-2 font-mono text-[11px] font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
            >
              SEND
            </button>
          </form>
        </>
      )}

      {a.error && <p className="mt-1.5 font-mono text-[10px] text-danger">{a.error}</p>}
    </div>
  );
}
