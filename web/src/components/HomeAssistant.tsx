"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import VoiceInputButton, { MicIcon } from "@/components/VoiceInputButton";
import { useConversationalAssistant } from "@/lib/useConversationalAssistant";
import { useWakeWordListener } from "@/lib/useWakeWordListener";
import { sessionStatusLabel, SpeakerIcon } from "@/components/ConversationalAssistantUI";

const GREETED_SESSION_KEY = "finsight-greeted";

function wakeWordLabel(state: "idle" | "listening" | "error"): string {
  if (state === "listening") return 'LISTENING FOR "HEY FINSIGHT"';
  if (state === "error") return "WAKE WORD ERROR -- TAP TO RETRY";
  return 'ENABLE "HEY FINSIGHT"';
}

// Compact conversational assistant, embedded on Home so talking to
// FinSight (text or voice) is active the instant the app opens --
// previously only reachable by first switching to the separate Chat
// tab. Same useConversationalAssistant hook as the full-page /chat
// view (identical session-loop behavior, no duplicated logic).
//
// Two states, on purpose:
// - Before any interaction this session: a minimal spoken greeting
//   ("Hi <name>, what would you like to do today?") plus ONE big
//   tap-to-talk control -- no visible text box, no message history,
//   nothing to read before you can act. A real platform limit shapes
//   this: a browser can never start listening on its own (mic access
//   requires an actual tap, no exception, on every browser), so
//   "voice-activated on open" in practice means "the greeting speaks
//   itself, and the very next tap starts a full session" -- the
//   closest thing to hands-free a web page can offer.
// - After the first message either way: the fuller card (latest
//   exchange, text input, link to the full thread) -- once there's
//   something to review, hiding it would be worse, not more minimal.
export default function HomeAssistant({ userName }: { userName: string }) {
  const a = useConversationalAssistant();
  const greetedRef = useRef(false);
  const lastUser = [...a.messages].reverse().find((m) => m.role === "user");
  const lastAssistant = [...a.messages].reverse().find((m) => m.role === "assistant");
  // Deliberately NOT derived from a.messages.length > 0 -- that would
  // include history from EARLIER visits (fetched on mount), so a
  // returning user with any past conversation would never see the
  // greeting screen at all. This tracks only "did something happen
  // THIS page visit" -- the greeting should show every time the app
  // opens, not just the very first time ever.
  const [justInteracted, setJustInteracted] = useState(false);
  const hasInteracted = justInteracted || a.sessionActive;

  // Detecting the phrase only hands off to the EXISTING tap-to-start
  // session loop for the actual command -- same reasoning as the
  // backend relay's own docstring: keep the new, riskier code (a
  // persistent mic stream + WebSocket) doing as little as possible,
  // and reuse the already-hardened batch pipeline for everything else.
  const wakeWord = useWakeWordListener(() => {
    setJustInteracted(true);
    wakeWord.pause();
    void a.startSession();
  });

  // Resume listening for the NEXT wake phrase once the triggered
  // session ends -- but only if the listener is still meant to be
  // running (the user might have tapped "stop" mid-session, or a mic
  // error might have already torn it down).
  const prevSessionActiveRef = useRef(false);
  useEffect(() => {
    if (prevSessionActiveRef.current && !a.sessionActive && wakeWord.state === "listening") {
      wakeWord.resume();
    }
    prevSessionActiveRef.current = a.sessionActive;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a.sessionActive]);

  async function handleGreetingSend(e: React.FormEvent) {
    e.preventDefault();
    if (!a.input.trim()) return;
    setJustInteracted(true);
    await a.handleSend(e);
  }

  function toggleWakeWord() {
    if (wakeWord.state === "listening") wakeWord.stop();
    else void wakeWord.start();
  }

  useEffect(() => {
    if (greetedRef.current) return;
    if (sessionStorage.getItem(GREETED_SESSION_KEY)) return;
    greetedRef.current = true;
    sessionStorage.setItem(GREETED_SESSION_KEY, "1");
    // Best-effort: browsers commonly block audio.play() this soon
    // after a page load with no direct click on THIS page (autoplay
    // policy) -- synthesizeAndSpeak already swallows a rejected
    // play() silently, so a blocked greeting just never plays instead
    // of erroring. The greeting text itself still renders either way.
    void a.synthesizeAndSpeak(`Hi ${userName}, what would you like to do today?`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!hasInteracted) {
    return (
      <div className="mt-4 rounded-lg border border-border bg-card px-4 py-5 text-center">
        <p className="font-mono text-sm text-text">
          Hi {userName}, what would you like to do today?
        </p>
        <button
          type="button"
          onClick={a.startSession}
          aria-label="Start talking to FinSight"
          className="mx-auto mt-3 flex h-14 w-14 items-center justify-center rounded-full border border-accent text-accent hover:bg-accent/10"
        >
          <MicIcon active={false} />
        </button>
        <p className="mt-2 font-mono text-[10px] text-dim">tap to talk, or type instead</p>
        <form
          onSubmit={handleGreetingSend}
          className="mx-auto mt-3 flex max-w-sm items-center gap-2"
        >
          <input
            type="text"
            value={a.input}
            onChange={(e) => a.setInput(e.target.value)}
            placeholder="or type a question..."
            disabled={a.sending}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={a.sending || !a.input.trim()}
            className="shrink-0 rounded-lg bg-accent px-3 py-2 font-mono text-[11px] font-bold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            SEND
          </button>
        </form>
        <button
          type="button"
          onClick={toggleWakeWord}
          className={`mx-auto mt-3 block font-mono text-[10px] font-bold ${
            wakeWord.state === "listening" ? "text-accent" : "text-dim hover:text-muted"
          }`}
        >
          {wakeWordLabel(wakeWord.state)}
        </button>
        {a.error && <p className="mt-1.5 font-mono text-[10px] text-danger">{a.error}</p>}
        {wakeWord.errorMessage && <p className="mt-1.5 font-mono text-[10px] text-danger">{wakeWord.errorMessage}</p>}
      </div>
    );
  }

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
            onClick={toggleWakeWord}
            title={wakeWordLabel(wakeWord.state)}
            className={`flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[9px] font-bold transition-colors ${
              wakeWord.state === "listening" ? "border-accent text-accent animate-pulse" : "border-border text-dim hover:text-muted"
            }`}
          >
            <MicIcon active={wakeWord.state === "listening"} />
          </button>
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
