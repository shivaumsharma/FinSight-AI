"""Tests for the live price hub (app/data/price_hub.py) and GET/WS /v1/prices/stream."""

import asyncio

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.api import auth, db
from app.api.main import app
from app.data import price_hub
from app.data.price_hub import PriceHub
from app.tests.test_api import _signup


def _quotes(prices):
    """A fetch function reading from a mutable dict; a ticker missing from it behaves like an unknown symbol."""
    def fetch(ticker):
        if ticker not in prices:
            raise KeyError(ticker)
        price, change = prices[ticker]
        return {"price": price, "change_pct": change, "currency": "USD"}
    return fetch


def _manual_hub(prices):
    """A hub whose background loop never starts, so a test drives polling itself with poll_once()."""
    hub = PriceHub(fetch=_quotes(prices), interval=0.01)
    hub._ensure_running = lambda: None
    return hub


# ---------------------------------------------------------------- hub

def test_poll_forwards_only_prices_that_changed():
    async def go():
        prices = {"AAPL": (100.0, 1.0), "MSFT": (200.0, -0.5)}
        hub = _manual_hub(prices)
        client = hub.register()
        hub.set_subscriptions(client, ["AAPL", "MSFT"])

        assert await hub.poll_once() == 2
        first = {t["t"]: t["p"] for t in client.take()}
        assert first == {"AAPL": 100.0, "MSFT": 200.0}

        prices["AAPL"] = (101.0, 2.0)  # only AAPL moves
        assert await hub.poll_once() == 1
        assert [t["t"] for t in client.take()] == ["AAPL"]

        assert await hub.poll_once() == 0  # nothing moved: nothing sent
        assert client.take() == []
    asyncio.run(go())


def test_a_slow_client_gets_only_the_latest_price_per_ticker():
    async def go():
        prices = {"AAPL": (100.0, 0.0)}
        hub = _manual_hub(prices)
        client = hub.register()
        hub.set_subscriptions(client, ["AAPL"])
        for p in (100.0, 101.0, 102.0, 103.0):
            prices["AAPL"] = (p, 0.0)
            await hub.poll_once()
        ticks = client.take()  # client never drained in between
        assert [t["p"] for t in ticks] == [103.0]
    asyncio.run(go())


def test_clients_only_receive_their_own_subscriptions():
    async def go():
        prices = {"AAPL": (100.0, 0.0), "MSFT": (200.0, 0.0)}
        hub = _manual_hub(prices)
        a, b = hub.register(), hub.register()
        hub.set_subscriptions(a, ["AAPL"])
        hub.set_subscriptions(b, ["MSFT"])
        await hub.poll_once()
        assert [t["t"] for t in a.take()] == ["AAPL"]
        assert [t["t"] for t in b.take()] == ["MSFT"]
    asyncio.run(go())


def test_a_ticker_that_fails_does_not_stop_the_others():
    async def go():
        hub = _manual_hub({"AAPL": (100.0, 0.0)})
        client = hub.register()
        hub.set_subscriptions(client, ["AAPL", "NOPE"])
        assert await hub.poll_once() == 1
        assert [t["t"] for t in client.take()] == ["AAPL"]
    asyncio.run(go())


def test_a_late_subscriber_gets_the_known_price_immediately():
    async def go():
        hub = _manual_hub({"AAPL": (100.0, 0.0)})
        first = hub.register()
        hub.set_subscriptions(first, ["AAPL"])
        await hub.poll_once()
        late = hub.register()
        hub.set_subscriptions(late, ["AAPL"])
        assert [t["p"] for t in late.take()] == [100.0]
    asyncio.run(go())


def test_subscriptions_are_normalised_deduplicated_and_capped():
    async def go():
        hub = _manual_hub({})
        client = hub.register()
        accepted = hub.set_subscriptions(client, [" aapl", "AAPL", "", "x" * 30] + [f"T{i}" for i in range(80)])
        assert accepted[0] == "AAPL" and accepted.count("AAPL") == 1
        assert "X" * 30 not in accepted and len(accepted) == price_hub.MAX_TICKERS_PER_CLIENT
    asyncio.run(go())


def test_unsubscribing_drops_pending_ticks_for_that_ticker():
    async def go():
        hub = _manual_hub({"AAPL": (100.0, 0.0), "MSFT": (200.0, 0.0)})
        client = hub.register()
        hub.set_subscriptions(client, ["AAPL", "MSFT"])
        await hub.poll_once()
        hub.set_subscriptions(client, ["MSFT"])
        assert [t["t"] for t in client.take()] == ["MSFT"]
    asyncio.run(go())


def test_wait_for_event_reports_set_and_timeout():
    async def go():
        event = asyncio.Event()
        assert await price_hub.wait_for_event(event, 0.01) is False
        asyncio.get_running_loop().call_later(0.01, event.set)
        assert await price_hub.wait_for_event(event, 1) is True
    asyncio.run(go())


def test_a_cancelled_polling_loop_always_stops():
    """Cancels a tight wait loop at many moments around its timeout; on Python 3.11 asyncio.wait_for can swallow one."""
    async def go():
        async def poll_forever():
            event = asyncio.Event()
            while True:
                await price_hub.wait_for_event(event, 0.001)

        for i in range(150):
            task = asyncio.ensure_future(poll_forever())
            await asyncio.sleep(0.0004 * (i % 9))
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=1)
            assert done, f"task {i} kept running after cancel()"
    asyncio.run(go())


# ---------------------------------------------------------------- endpoints

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def hub(monkeypatch):
    prices = {"AAPL": (233.1, 1.4), "MSFT": (410.0, -0.2)}
    h = PriceHub(fetch=_quotes(prices), interval=0.05)
    h.prices = prices
    price_hub.reset_hub(h)
    yield h
    price_hub.reset_hub(None)


def _token(client, email="stream_user@example.com"):
    headers = _signup(client, email=email)
    resp = client.get("/v1/prices/stream-token", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def test_stream_token_requires_a_session(client):
    assert client.get("/v1/prices/stream-token").status_code == 401


def test_stream_closes_on_a_malformed_first_message(client, hub):
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_text("not json")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4001


def test_stream_rejects_a_bad_token(client, hub):
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_json({"token": "garbage"})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_a_voice_token_cannot_be_replayed_on_the_price_stream(client, hub):
    headers = _signup(client, email="replay@example.com")
    voice = client.get("/v1/voice/wake-listen-token", headers=headers).json()["token"]
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_json({"token": voice})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_stream_rejects_an_expired_token(client, hub, monkeypatch):
    monkeypatch.setattr(auth, "REALTIME_VOICE_TOKEN_TTL_SECONDS", -1)
    token = _token(client, email="expired_stream@example.com")
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_json({"token": token})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_stream_delivers_ticks_for_subscribed_tickers(client, hub):
    token = _token(client)
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_json({"token": token})
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "subscribe", "tickers": ["aapl", "MSFT"]})

        seen = {}
        subscribed = False
        for _ in range(10):
            msg = ws.receive_json()
            if msg["type"] == "subscribed":
                subscribed = True
                assert msg["tickers"] == ["AAPL", "MSFT"]
            elif msg["type"] == "ticks":
                for t in msg["ticks"]:
                    seen[t["t"]] = t["p"]
            if subscribed and len(seen) == 2:
                break
        assert seen == {"AAPL": 233.1, "MSFT": 410.0}

        hub.prices["AAPL"] = (234.0, 1.8)
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "ticks":
                for t in msg["ticks"]:
                    seen[t["t"]] = t["p"]
            if seen["AAPL"] == 234.0:
                break
        assert seen["AAPL"] == 234.0


def test_stream_answers_ping(client, hub):
    token = _token(client, email="ping@example.com")
    with client.websocket_connect("/v1/prices/stream") as ws:
        ws.send_json({"token": token})
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
