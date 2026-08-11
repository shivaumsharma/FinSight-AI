"""
Unit tests for app/data/sarvam_client.py -- monkeypatched at the
_get_session() seam (a fake session whose .post() returns pre-built
requests.Response objects), no live network. Exercises the actual
retry_on_transient_error wiring (not bypassed), since this client's
whole point is behaving correctly under Sarvam's transient failures --
retry.time.sleep is monkeypatched to keep this fast, matching
test_retry.py's own convention.
"""

import json

import pytest
import requests

from app.core import retry as retry_module
from app.data import sarvam_client as sarvam


def _response(status_code, json_body=None):
    resp = requests.Response()
    resp.status_code = status_code
    if json_body is not None:
        resp._content = json.dumps(json_body).encode("utf-8")
    return resp


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda s: None)


@pytest.fixture(autouse=True)
def _reset_session(monkeypatch):
    # sarvam_client's _session is a module-level singleton -- reset it
    # per test so an earlier test's fake session never leaks into a
    # later one via _get_session()'s "if None, create" caching.
    monkeypatch.setattr(sarvam, "_session", None)


def test_transcribe_success(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(200, {"transcript": "Should I buy Apple?"})])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    result = sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")

    assert result == "Should I buy Apple?"
    assert fake.calls == 1


def test_transcribe_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(500), _response(500), _response(200, {"transcript": "recovered"})])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    result = sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")

    assert result == "recovered"
    assert fake.calls == 3


def test_transcribe_raises_after_retries_exhausted(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(500), _response(500), _response(500)])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    with pytest.raises(sarvam.SarvamTranscriptionError):
        sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")

    assert fake.calls == 3  # 1 initial attempt + 2 retries, matching _post_to_sarvam's max_retries=2


def test_transcribe_does_not_retry_a_non_retryable_4xx(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(401)])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    with pytest.raises(sarvam.SarvamTranscriptionError):
        sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")

    assert fake.calls == 1  # no retry burned on a certain (bad-key) failure


def test_transcribe_error_message_includes_sarvams_response_body(monkeypatch):
    # A bare "400 Client Error: Bad Request" (requests' own default
    # HTTPError message) throws away exactly the information needed to
    # tell "malformed audio" apart from every other 4xx -- Sarvam's own
    # response body is the real diagnostic and must survive into
    # SarvamTranscriptionError's message.
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(400, {"error": "Invalid audio: missing duration metadata"})])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    with pytest.raises(sarvam.SarvamTranscriptionError, match="missing duration metadata"):
        sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")


def test_transcribe_raises_before_any_network_call_when_key_missing(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    def _boom():
        raise AssertionError("_get_session should never be called without an API key")

    monkeypatch.setattr(sarvam, "_get_session", _boom)

    with pytest.raises(sarvam.SarvamTranscriptionError, match="SARVAM_API_KEY"):
        sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")


def test_transcribe_raises_on_empty_transcript(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(200, {"transcript": "   "})])
    monkeypatch.setattr(sarvam, "_get_session", lambda: fake)

    with pytest.raises(sarvam.SarvamTranscriptionError, match="empty transcript"):
        sarvam.transcribe(b"audio-bytes", "recording.webm", "audio/webm")
