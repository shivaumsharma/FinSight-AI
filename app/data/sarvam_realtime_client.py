"""
sarvam_realtime_client.py

The wake-word listener's backend half: relays raw PCM16 audio from a
browser WebSocket (app/api/main.py's /v1/voice/wake-listen) to
Sarvam's realtime streaming STT (`saaras:v3-realtime`), watching every
partial/final transcript for a wake phrase and signaling the browser
the moment one is heard.

Uses Sarvam's own official SDK (`sarvamai`) for this connection,
unlike sarvam_client.py/sarvam_tts_client.py's plain `requests` calls
-- those are simple request/response, but the realtime WebSocket
protocol's exact message shapes turned out to differ between Sarvam's
own doc pages when researched directly; the SDK (which Sarvam
maintains against their real, current wire protocol) is a materially
more reliable source of truth than re-implementing that protocol by
hand against inconsistent docs.

Deliberate design choice: this module's only job is detecting the
wake phrase, not transcribing the actual command that follows it. The
existing tap-to-start voice session (VoiceInputButton.tsx's batch
record -> WAV -> sarvam_client.transcribe() pipeline) already handles
capturing and transcribing a real command reliably -- reusing it for
the actual command means the risky new code here (a persistent
bidirectional relay) stays as small and low-stakes as possible: detect
the phrase, then hand off to code that's already been hardened through
several rounds of live debugging this session.
"""

import asyncio
import base64
import logging
import os
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from sarvamai import AsyncSarvamAI, RealtimeAudioInput

logger = logging.getLogger(__name__)

# Matched as a case-insensitive substring of the (lowercased) transcript
# -- deliberately loose rather than an exact-phrase match, since Sarvam
# may transcribe "Hey FinSight" with different punctuation/spacing, and
# a false negative (missed wake phrase) is a worse failure mode here
# than an occasional false positive (which just opens a voice session
# the user can immediately end).
WAKE_PHRASES = ("hey finsight", "hi finsight", "hey fin sight", "hi fin sight")

_REALTIME_MODEL = "saaras:v3-realtime"


class SarvamRealtimeError(Exception):
    """Raised only when the realtime session can't be established at
    all (missing/invalid SARVAM_API_KEY, connection rejected). A
    mid-session Sarvam-side error is logged and handled inline instead
    of raised -- the wake-word listener should keep listening through
    a transient hiccup, not tear the whole connection down over it."""


def _contains_wake_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in WAKE_PHRASES)


async def relay_for_wake_word(browser_ws: WebSocket, language_code: str = "en-IN") -> None:
    """
    Proxies raw PCM16 (linear16, 16kHz, mono) audio from `browser_ws`
    (already accept()-ed by the caller) to Sarvam's realtime STT.
    Sends {"event": "wake_detected", "text": <transcript>} back to
    browser_ws the instant a wake phrase is heard in either a partial
    or final transcript -- the browser decides what to do next (this
    function keeps relaying regardless, so it also works if the caller
    wants to detect a second wake phrase later in the same session).

    Returns when browser_ws disconnects or the Sarvam connection ends
    fatally. Raises SarvamRealtimeError only for a failure to
    establish the Sarvam connection in the first place.
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SarvamRealtimeError("SARVAM_API_KEY is not configured.")

    client = AsyncSarvamAI(api_subscription_key=api_key)

    try:
        async with client.speech_to_text_realtime_streaming.connect(
            language_code=language_code,
            model=_REALTIME_MODEL,
            encoding="linear16",
            sample_rate="16000",
            endpointing="vad",
            # "fast" over the default "balanced": a wake-word check only
            # needs a quick, good-enough transcript to substring-match
            # against four fixed phrases, not best-possible accuracy --
            # trading a little WER for lower latency is the right call
            # here specifically (unlike the batch command transcription
            # in VoiceInputButton.tsx, which stays on Sarvam's default).
            stream_type="fast",
        ) as sarvam_ws:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_pump_browser_audio(browser_ws, sarvam_ws))
                tg.create_task(_pump_sarvam_transcripts(sarvam_ws, browser_ws))
    except* WebSocketDisconnect:
        pass  # the browser closing the tab/ending the session is expected, not an error
    except* Exception as eg:
        raise SarvamRealtimeError(f"Sarvam realtime connection failed: {eg.exceptions}") from eg


async def _pump_browser_audio(browser_ws: WebSocket, sarvam_ws) -> None:
    """Browser -> Sarvam: forward each raw binary audio chunk as-is."""
    while True:
        chunk = await browser_ws.receive_bytes()
        await sarvam_ws.send_realtime_audio_input(
            RealtimeAudioInput(audio=base64.b64encode(chunk).decode("ascii"))
        )


async def _pump_sarvam_transcripts(sarvam_ws, browser_ws: WebSocket) -> None:
    """Sarvam -> browser: watch every transcript for the wake phrase."""
    async for message in sarvam_ws:
        event = getattr(message, "event", None)
        if event == "session.begin":
            # The browser's own WebSocket to us opens well before Sarvam's
            # session is actually up -- flipping the UI to "listening" on
            # that first handshake (rather than this) meant a user could
            # say the wake phrase into a connection that wasn't actually
            # capturing yet, which read as the whole feature being slow
            # to respond. This is the real "ready" signal.
            await browser_ws.send_json({"event": "ready"})
        elif event in ("transcript.partial", "transcript.final"):
            text = getattr(message, "text", "") or ""
            if _contains_wake_phrase(text):
                await browser_ws.send_json({"event": "wake_detected", "text": text})
        elif event == "error":
            is_fatal = getattr(message, "is_fatal", False)
            logger.warning(f"Sarvam realtime STT error (fatal={is_fatal}): {getattr(message, 'message', message)}")
            if is_fatal:
                return
