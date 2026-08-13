"""
Unit tests for app/data/sarvam_tts_client.py -- same monkeypatched-
_get_session() seam and retry-exercising convention as
test_sarvam_client.py (no live network, retry.time.sleep patched out).
"""

import base64
import json

import pytest
import requests

from app.core import retry as retry_module
from app.data import sarvam_tts_client as tts


def _response(status_code, json_body=None):
    resp = requests.Response()
    resp.status_code = status_code
    if json_body is not None:
        resp._content = json.dumps(json_body).encode("utf-8")
    return resp


def _audios_response(status_code, audio_bytes=b"RIFF...wav-bytes"):
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return _response(status_code, {"request_id": "abc", "audios": [encoded]})


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
    monkeypatch.setattr(tts, "_session", None)


def test_synthesize_success(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_audios_response(200, b"real-wav-bytes")])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    result = tts.synthesize("Hello there")

    assert result == b"real-wav-bytes"
    assert fake.calls == 1


def test_synthesize_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(500), _response(500), _audios_response(200, b"recovered-bytes")])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    result = tts.synthesize("Hello there")

    assert result == b"recovered-bytes"
    assert fake.calls == 3


def test_synthesize_raises_after_retries_exhausted(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(500), _response(500), _response(500)])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    with pytest.raises(tts.SarvamSynthesisError):
        tts.synthesize("Hello there")

    assert fake.calls == 3  # 1 initial attempt + 2 retries, matching _post_to_sarvam's max_retries=2


def test_synthesize_does_not_retry_a_non_retryable_4xx(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(401)])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    with pytest.raises(tts.SarvamSynthesisError):
        tts.synthesize("Hello there")

    assert fake.calls == 1  # no retry burned on a certain (bad-key) failure


def test_synthesize_error_message_includes_sarvams_response_body(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(400, {"error": "Invalid speaker name"})])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    with pytest.raises(tts.SarvamSynthesisError, match="Invalid speaker name"):
        tts.synthesize("Hello there")


def test_synthesize_raises_before_any_network_call_when_key_missing(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    def _boom():
        raise AssertionError("_get_session should never be called without an API key")

    monkeypatch.setattr(tts, "_get_session", _boom)

    with pytest.raises(tts.SarvamSynthesisError, match="SARVAM_API_KEY"):
        tts.synthesize("Hello there")


def test_synthesize_raises_before_any_network_call_on_blank_text(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    def _boom():
        raise AssertionError("_get_session should never be called for blank text")

    monkeypatch.setattr(tts, "_get_session", _boom)

    with pytest.raises(tts.SarvamSynthesisError, match="No text"):
        tts.synthesize("   ")


def test_synthesize_raises_before_any_network_call_when_text_too_long(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    def _boom():
        raise AssertionError("_get_session should never be called for over-limit text")

    monkeypatch.setattr(tts, "_get_session", _boom)

    with pytest.raises(tts.SarvamSynthesisError, match="2500-character limit"):
        tts.synthesize("x" * (tts.MAX_TEXT_CHARS + 1))


def test_synthesize_raises_when_no_audio_returned(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    fake = _FakeSession([_response(200, {"request_id": "abc", "audios": []})])
    monkeypatch.setattr(tts, "_get_session", lambda: fake)

    with pytest.raises(tts.SarvamSynthesisError, match="no audio"):
        tts.synthesize("Hello there")


def test_synthesize_passes_speaker_and_pace_through(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    captured = {}

    class _CapturingSession:
        def post(self, url, headers=None, json=None, timeout=None):
            captured["json"] = json
            captured["headers"] = headers
            return _audios_response(200)

    monkeypatch.setattr(tts, "_get_session", lambda: _CapturingSession())

    tts.synthesize("Hello there", language_code="hi-IN", speaker="priya", pace=1.5)

    assert captured["json"]["language_code"] == "hi-IN"
    assert captured["json"]["speaker"] == "priya"
    assert captured["json"]["pace"] == 1.5
    assert captured["json"]["model"] == "bulbul:v3"
    assert captured["headers"]["api-subscription-key"] == "test-key"
