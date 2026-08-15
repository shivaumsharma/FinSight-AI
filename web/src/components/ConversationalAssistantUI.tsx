import type { VoiceState } from "@/components/VoiceInputButton";

export function SpeakerIcon({ speaking }: { speaking: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
      <path d="M4 9v6h4l5 5V4L8 9H4z" strokeLinejoin="round" />
      {speaking && <path d="M17 8a6 6 0 0 1 0 8M20 5a10 10 0 0 1 0 14" strokeLinecap="round" />}
    </svg>
  );
}

// Status readout for an active voice session -- spans both
// VoiceInputButton's own recording state (via onStateChange) and the
// page's own TTS-playback state, since a session cycles through both.
// Shared between the full-page Chat view and the compact Home widget
// so a session started in one place reads identically in the other.
export function sessionStatusLabel(micState: VoiceState, sending: boolean, speaking: boolean): string {
  if (speaking) return "SPEAKING...";
  if (sending || micState === "transcribing") return "THINKING...";
  if (micState === "recording") return "LISTENING...";
  if (micState === "error") return "DIDN'T CATCH THAT -- RETRYING...";
  return "STARTING...";
}
