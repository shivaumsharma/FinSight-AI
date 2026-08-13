"""
sarvam_tts_client.py

Text-to-speech for spoken chat replies (voice-conversation feature) via
Sarvam AI's synchronous REST API (https://api.sarvam.ai/text-to-speech)
-- the same paid, user-supplied SARVAM_API_KEY sarvam_client.py already
uses for speech-to-text (one Sarvam account, two capabilities).

Mirrors sarvam_client.py's shape (module-level functions, one shared
requests.Session(), network calls wrapped in retry_on_transient_error,
raises on failure rather than degrading to None -- a spoken reply the
user is actively waiting on should fail loudly, same reasoning
sarvam_client.py's own docstring gives for STT).

Confirmed against Sarvam's live REST API docs, not guessed:
- JSON POST (not multipart -- STT uploads a file, TTS submits text),
  auth via the same `api-subscription-key` header (NOT Bearer-prefixed).
- Response is JSON, not raw audio bytes: {"request_id": str,
  "audios": [base64_encoded_wav_string]} -- must be base64-decoded
  before it's usable audio.
- 2500-character cap on `text` for bulbul:v3 (1500 for v2) -- callers
  must truncate/summarize before calling synthesize(), this client
  does not chunk long text into multiple requests.
- `audio_format` defaults to WAV server-side; requested explicitly here
  for clarity rather than relying on the default.
"""

import base64
import os
from typing import Optional

import requests

from app.core.retry import retry_on_transient_error

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "Shubh"
DEFAULT_LANGUAGE_CODE = "en-IN"
MAX_TEXT_CHARS = 2500


class SarvamSynthesisError(Exception):
    """Raised for any failure synthesizing speech via Sarvam AI --
    missing/invalid SARVAM_API_KEY, text over MAX_TEXT_CHARS, a Sarvam
    request that still fails after retry_on_transient_error's retries
    are exhausted, or a response with no usable audio. Never propagates
    raw past app/api/main.py's synthesize_voice endpoint."""


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _post_to_sarvam(text: str, language_code: str, speaker: str, pace: float, api_key: str):
    """The network-call seam -- tests monkeypatch this directly,
    same convention as sarvam_client.py's _post_to_sarvam."""
    def _do():
        resp = _get_session().post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "language_code": language_code,
                "model": DEFAULT_MODEL,
                "speaker": speaker,
                "pace": pace,
                "audio_format": "wav",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp

    return retry_on_transient_error(_do, max_retries=2)


def synthesize(text: str, language_code: str = DEFAULT_LANGUAGE_CODE,
               speaker: str = DEFAULT_SPEAKER, pace: float = 1.0) -> bytes:
    """
    Returns raw WAV audio bytes for `text`. Raises SarvamSynthesisError
    -- never returns None, never returns empty bytes -- on: a missing
    SARVAM_API_KEY, text over MAX_TEXT_CHARS, any network/HTTP failure
    surviving retry, or a response with no usable audio.

    language_code defaults to "en-IN" (unlike sarvam_client.transcribe's
    "unknown" auto-detect default) because TTS has no text to detect a
    language FROM -- the target language must be stated up front. This
    app's chat replies are English today; a caller in another language
    context should pass its own language_code explicitly.
    """
    if not text or not text.strip():
        raise SarvamSynthesisError("No text provided to synthesize.")
    if len(text) > MAX_TEXT_CHARS:
        raise SarvamSynthesisError(f"Text is {len(text)} characters, over the {MAX_TEXT_CHARS}-character limit.")

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SarvamSynthesisError("SARVAM_API_KEY is not configured.")

    try:
        resp = _post_to_sarvam(text, language_code, speaker, pace, api_key)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError) as e:
        # Same reasoning as sarvam_client.py's transcribe(): surface
        # Sarvam's actual error body rather than a bare "400 Bad
        # Request" that gives no clue which field was rejected.
        detail = str(e)
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            body_snippet = (e.response.text or "").strip()[:300]
            if body_snippet:
                detail = f"{e} -- {body_snippet}"
        raise SarvamSynthesisError(f"Sarvam TTS request failed: {detail}") from e

    audios = resp.json().get("audios") or []
    if not audios or not audios[0]:
        raise SarvamSynthesisError("Sarvam returned no audio.")

    try:
        return base64.b64decode(audios[0])
    except (ValueError, TypeError) as e:
        raise SarvamSynthesisError(f"Sarvam returned unparseable audio: {e}") from e
