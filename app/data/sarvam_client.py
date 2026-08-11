"""
sarvam_client.py

Speech-to-text for the voice-input mic button (web/src/components/
VoiceInputButton.tsx) via Sarvam AI's synchronous REST API
(https://api.sarvam.ai/speech-to-text) -- a paid, user-supplied
SARVAM_API_KEY (see .env.example).

Mirrors app/data/nse_filings_client.py's shape (module-level functions,
one shared requests.Session(), network calls wrapped in
retry_on_transient_error) with one deliberate divergence: THIS client
raises on failure, it does not degrade to None. NSE/SEC filing evidence
is optional context a report can complete without; a voice
transcription is a synchronous action the user is actively waiting on
-- silently doing nothing would be worse than a clear error. app/api/
main.py catches SarvamTranscriptionError and translates it into the
structured STT_UNAVAILABLE API error (see app/api/errors.py).

Confirmed against Sarvam's live REST API docs, not guessed:
- multipart/form-data POST, auth via `api-subscription-key` header
  (NOT Bearer-prefixed -- different convention from this app's own
  session tokens).
- WebM is a natively supported input format, which matters because
  browsers' MediaRecorder defaults to audio/webm;codecs=opus -- no
  client-side transcoding needed anywhere in this pipeline.
- The synchronous endpoint used here caps input at 30 seconds of audio
  (a separate batch API exists for longer clips, not needed for a
  spoken question -- VoiceInputButton enforces a 25s recording ceiling
  client-side to stay safely under this).
"""

import os
from typing import Optional

import requests

from app.core.retry import retry_on_transient_error

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v3"


class SarvamTranscriptionError(Exception):
    """Raised for any failure transcribing audio via Sarvam AI --
    missing/invalid SARVAM_API_KEY, a Sarvam request that still fails
    after retry_on_transient_error's retries are exhausted, or a
    response with no usable transcript. Never propagates raw past
    app/api/main.py's transcribe_voice endpoint."""


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _post_to_sarvam(audio_bytes: bytes, filename: str, content_type: str,
                     language_code: str, api_key: str):
    """The network-call seam -- tests monkeypatch this directly
    (matching nse_filings_client.py's _fetch_json/_fetch_pdf_bytes
    convention) rather than mocking `requests` itself."""
    def _do():
        resp = _get_session().post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": (filename, audio_bytes, content_type)},
            data={"model": DEFAULT_MODEL, "mode": "transcribe", "language_code": language_code},
            timeout=30,
        )
        resp.raise_for_status()
        return resp

    return retry_on_transient_error(_do, max_retries=2)


def transcribe(audio_bytes: bytes, filename: str, content_type: str,
               language_code: str = "unknown") -> str:
    """
    Returns the transcript string for `audio_bytes` (already validated
    non-empty and under the size ceiling by main.py's caller). Raises
    SarvamTranscriptionError -- never returns None, never returns an
    empty/whitespace-only string -- on: a missing SARVAM_API_KEY, any
    network/HTTP failure surviving retry, or Sarvam returning a blank
    transcript (e.g. a silent recording).

    language_code defaults to "unknown" (Sarvam's own auto-detect
    value) rather than hardcoding "en-IN" -- this app's users are not
    assumed to be speaking a single fixed language, and Sarvam's own
    docs recommend "unknown" specifically for this case.
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SarvamTranscriptionError("SARVAM_API_KEY is not configured.")

    try:
        resp = _post_to_sarvam(audio_bytes, filename, content_type, language_code, api_key)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError) as e:
        raise SarvamTranscriptionError(f"Sarvam STT request failed: {e}") from e

    transcript = (resp.json().get("transcript") or "").strip()
    if not transcript:
        raise SarvamTranscriptionError("Sarvam returned an empty transcript.")
    return transcript
