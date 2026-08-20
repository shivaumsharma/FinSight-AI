"""
Unit tests for app/data/sarvam_realtime_client.py. The Sarvam side is
faked with simple async iterables/namespaces (not a real WebSocket or
network call) -- these test the relay/wake-detection LOGIC, not the
live Sarvam wire protocol itself (that's only verifiable against
production, same as sarvam_client.py/sarvam_tts_client.py's own
"confirmed against the live API" notes).

No pytest-asyncio dependency -- this project has no existing async
test infrastructure, so each async body is run via plain
asyncio.run() from an ordinary sync test function rather than adding
a new test-only dependency and its config for a single module.
"""

import asyncio
import base64
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

from app.data import sarvam_realtime_client as src


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- _contains_wake_phrase

def test_contains_wake_phrase_matches_case_insensitively():
    assert src._contains_wake_phrase("okay so Hey FinSight what's up") is True


def test_contains_wake_phrase_matches_hi_variant():
    assert src._contains_wake_phrase("hi finsight can you help") is True


def test_contains_wake_phrase_false_for_unrelated_text():
    assert src._contains_wake_phrase("what's my portfolio doing") is False


def test_contains_wake_phrase_false_for_empty_string():
    assert src._contains_wake_phrase("") is False


# ---------------------------------------------------------------- relay_for_wake_word

def test_relay_raises_before_any_connection_when_key_missing(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(src.SarvamRealtimeError, match="SARVAM_API_KEY"):
        _run(src.relay_for_wake_word(browser_ws=None))  # never reached -- fails before touching it


# ---------------------------------------------------------------- _pump_sarvam_transcripts

class _FakeBrowserWs:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class _FakeSarvamSocket:
    """Async-iterable over a fixed list of fake Sarvam messages."""

    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m


def test_pump_sarvam_transcripts_forwards_session_begin_as_ready():
    messages = [SimpleNamespace(event="session.begin", request_id="abc123")]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [{"event": "ready"}]


def test_pump_sarvam_transcripts_forwards_wake_detection():
    messages = [
        SimpleNamespace(event="transcript.partial", text="hows my portfolio"),
        SimpleNamespace(event="transcript.partial", text="hey finsight whats up"),
        SimpleNamespace(event="transcript.final", text="hey finsight whats up"),
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [
        {"event": "wake_detected", "text": "hey finsight whats up"},
        {"event": "wake_detected", "text": "hey finsight whats up"},
    ]


def test_pump_sarvam_transcripts_ignores_non_matching_transcripts():
    messages = [SimpleNamespace(event="transcript.final", text="whats going on with apple")]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == []


def test_pump_sarvam_transcripts_stops_on_fatal_error():
    messages = [
        SimpleNamespace(event="error", is_fatal=True, message="quota exceeded"),
        SimpleNamespace(event="transcript.final", text="hey finsight"),  # never reached
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == []  # the loop returned before the wake phrase after the fatal error


def test_pump_sarvam_transcripts_continues_past_a_non_fatal_error():
    messages = [
        SimpleNamespace(event="error", is_fatal=False, message="config rejected"),
        SimpleNamespace(event="transcript.final", text="hey finsight"),
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [{"event": "wake_detected", "text": "hey finsight"}]


# ---------------------------------------------------------------- relay_for_conversation

def test_relay_for_conversation_raises_before_any_connection_when_key_missing(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(src.SarvamRealtimeError, match="SARVAM_API_KEY"):
        _run(src.relay_for_conversation(browser_ws=None))  # never reached -- fails before touching it


# ---------------------------------------------------------------- _pump_sarvam_transcripts_verbatim

def test_pump_sarvam_transcripts_verbatim_forwards_session_begin_as_ready():
    messages = [SimpleNamespace(event="session.begin", request_id="abc123")]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts_verbatim(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [{"event": "ready"}]


def test_pump_sarvam_transcripts_verbatim_forwards_every_transcript_unfiltered():
    """Unlike _pump_sarvam_transcripts (wake-word only), this forwards
    ordinary speech too -- it's the actual command being transcribed
    here, not a phrase to substring-match."""
    messages = [
        SimpleNamespace(event="transcript.partial", text="whats going on"),
        SimpleNamespace(event="transcript.partial", text="whats going on with apple"),
        SimpleNamespace(event="transcript.final", text="whats going on with apple"),
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts_verbatim(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [
        {"event": "transcript.partial", "text": "whats going on"},
        {"event": "transcript.partial", "text": "whats going on with apple"},
        {"event": "transcript.final", "text": "whats going on with apple"},
    ]


def test_pump_sarvam_transcripts_verbatim_stops_on_fatal_error():
    messages = [
        SimpleNamespace(event="error", is_fatal=True, message="quota exceeded"),
        SimpleNamespace(event="transcript.final", text="never reached"),
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts_verbatim(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == []


def test_pump_sarvam_transcripts_verbatim_continues_past_a_non_fatal_error():
    messages = [
        SimpleNamespace(event="error", is_fatal=False, message="config rejected"),
        SimpleNamespace(event="transcript.final", text="still works"),
    ]
    browser_ws = _FakeBrowserWs()

    _run(src._pump_sarvam_transcripts_verbatim(_FakeSarvamSocket(messages), browser_ws))

    assert browser_ws.sent == [{"event": "transcript.final", "text": "still works"}]


# ---------------------------------------------------------------- _pump_browser_audio

class _FakeSarvamSendClient:
    def __init__(self):
        self.sent_audio = []

    async def send_realtime_audio_input(self, message):
        self.sent_audio.append(message)


class _FakeBrowserWsWithAudio:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def receive_bytes(self):
        if not self._chunks:
            raise WebSocketDisconnect()
        return self._chunks.pop(0)


def test_pump_browser_audio_forwards_base64_encoded_chunks():
    browser_ws = _FakeBrowserWsWithAudio([b"chunk-one", b"chunk-two"])
    sarvam_ws = _FakeSarvamSendClient()

    with pytest.raises(WebSocketDisconnect):
        _run(src._pump_browser_audio(browser_ws, sarvam_ws))

    assert [m.audio for m in sarvam_ws.sent_audio] == [
        base64.b64encode(b"chunk-one").decode("ascii"),
        base64.b64encode(b"chunk-two").decode("ascii"),
    ]
