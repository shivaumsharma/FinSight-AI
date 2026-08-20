// Splits an assistant reply into sentence-shaped chunks for pipelined
// TTS playback (see useConversationalAssistant.ts's synthesizeAndSpeak)
// -- speaking sentence 1 while sentence 2 is still being synthesized,
// instead of waiting for the whole reply to become one audio clip.
//
// The real risk here isn't general English prose, it's THIS app's own
// reply register: chat_router.py's f-strings are full of
// currency-formatted prices ("$180.00", "₹4,200.00", always two
// decimal places via `:,.2f`). A naive split on every "." would
// fracture "$180.00" into "$180" and "00" -- an audible broken pause
// mid-number. The fix doesn't need to special-case digits at all: a
// decimal point inside a price is always immediately followed by
// another digit, never whitespace, so requiring the boundary itself
// to be followed by whitespace-or-end-of-string already excludes it
// naturally. A run of punctuation ("...", "?!") is matched as one
// unit so it isn't fragmented into separate one-character sentences.
const SENTENCE_BOUNDARY = /[.!?]+(?=\s|$)/g;

export function splitSentences(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  const sentences: string[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  SENTENCE_BOUNDARY.lastIndex = 0;
  while ((match = SENTENCE_BOUNDARY.exec(trimmed)) !== null) {
    const end = match.index + match[0].length;
    const sentence = trimmed.slice(lastIndex, end).trim();
    if (sentence) sentences.push(sentence);
    lastIndex = end;
  }
  const remainder = trimmed.slice(lastIndex).trim();
  if (remainder) sentences.push(remainder);

  return sentences;
}
