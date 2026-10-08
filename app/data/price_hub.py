"""
price_hub.py

Fan-out of live quotes to WebSocket clients (GET /v1/prices/stream in app/api/main.py).

The source is the same Yahoo quote lookup the rest of the app uses (market_data.get_quote), polled on an interval: the
prices are near-real-time and can lag the exchange, not an exchange tick feed. The hub polls each subscribed ticker
once per interval no matter how many clients watch it, and only forwards prices that changed. Each client keeps at most
one pending tick per ticker, so a slow client always receives the latest price rather than a growing backlog.
"""

import asyncio
import logging
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = float(os.environ.get("PRICE_STREAM_INTERVAL_SECONDS", "5"))
MAX_TICKERS_PER_CLIENT = 50
MAX_TICKERS_TOTAL = 200
FETCH_CONCURRENCY = 8
IDLE_SHUTDOWN_SECONDS = 30


async def wait_for_event(event: asyncio.Event, timeout: float) -> bool:
    """True if the event was set, False if the timeout passed first.

    Not asyncio.wait_for: on Python 3.11 (CI and the Docker image) it can swallow a cancellation that lands at the
    moment its timeout fires, so a cancelled polling loop keeps running and asyncio.run never returns.
    """
    waiter = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait({waiter}, timeout=timeout)
        return waiter.done()
    finally:
        waiter.cancel()


def _default_fetch(ticker: str) -> dict:
    from app.data.market_data import get_quote

    return get_quote(ticker, max_age_seconds=0)


class Client:
    def __init__(self) -> None:
        self.tickers: Set[str] = set()
        self.pending: Dict[str, dict] = {}
        self.ready = asyncio.Event()

    def push(self, tick: dict) -> None:
        self.pending[tick["t"]] = tick  # latest wins
        self.ready.set()

    def take(self) -> List[dict]:
        ticks = list(self.pending.values())
        self.pending.clear()
        self.ready.clear()
        return ticks


class PriceHub:
    def __init__(self, fetch: Optional[Callable[[str], dict]] = None, interval: Optional[float] = None) -> None:
        self._fetch = fetch or _default_fetch
        self.interval = POLL_INTERVAL_SECONDS if interval is None else interval
        self.clients: Set[Client] = set()
        self.last: Dict[str, dict] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------ clients
    def register(self) -> Client:
        client = Client()
        self.clients.add(client)
        self._ensure_running()
        return client

    def unregister(self, client: Client) -> None:
        self.clients.discard(client)

    def set_subscriptions(self, client: Client, tickers: Iterable[str]) -> List[str]:
        """Replaces the client's ticker set (normalised, de-duplicated, capped). Known prices go out immediately; new
        tickers are fetched on the next cycle, which is brought forward. Returns the accepted tickers."""
        cleaned: List[str] = []
        for raw in tickers:
            t = str(raw).strip().upper()
            if t and len(t) <= 20 and t not in cleaned:
                cleaned.append(t)
        cleaned = cleaned[:MAX_TICKERS_PER_CLIENT]
        client.tickers = set(cleaned)
        client.pending = {t: v for t, v in client.pending.items() if t in client.tickers}
        for t in cleaned:
            if t in self.last:
                client.push(self.last[t])
        self._wake.set()
        return cleaned

    def _wanted(self) -> List[str]:
        wanted: Set[str] = set()
        for c in self.clients:
            wanted |= c.tickers
        return sorted(wanted)[:MAX_TICKERS_TOTAL]

    # ------------------------------------------------------------ polling
    def _ensure_running(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._loop is not loop:
            self._loop = loop
            self._wake = asyncio.Event()
            self._task = loop.create_task(self._run())

    async def _fetch_one(self, ticker: str, sem: asyncio.Semaphore) -> Optional[dict]:
        async with sem:
            try:
                q = await asyncio.to_thread(self._fetch, ticker)
            except Exception:
                return None  # one bad ticker never stops the others
        price = q.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            return None
        change = q.get("change_pct")
        return {"t": ticker, "p": float(price), "c": float(change) if isinstance(change, (int, float)) else None,
                "cur": q.get("currency") or "USD", "ts": int(time.time())}

    async def poll_once(self) -> int:
        """One cycle: fetch every wanted ticker, forward the ones whose price changed. Returns how many changed."""
        tickers = self._wanted()
        if not tickers:
            return 0
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)
        results = await asyncio.gather(*(self._fetch_one(t, sem) for t in tickers))
        changed = 0
        for tick in results:
            if tick is None:
                continue
            previous = self.last.get(tick["t"])
            self.last[tick["t"]] = tick
            if previous is not None and previous["p"] == tick["p"] and previous["c"] == tick["c"]:
                continue
            changed += 1
            for c in self.clients:
                if tick["t"] in c.tickers:
                    c.push(tick)
        return changed

    async def _run(self) -> None:
        idle_since: Optional[float] = None
        while True:
            if not self.clients:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > IDLE_SHUTDOWN_SECONDS:
                    return
            else:
                idle_since = None
                try:
                    await self.poll_once()
                except Exception:
                    logger.exception("price poll failed")
            await wait_for_event(self._wake, self.interval)
            self._wake.clear()


_hub: Optional[PriceHub] = None


def get_hub() -> PriceHub:
    global _hub
    if _hub is None:
        _hub = PriceHub()
    return _hub


def reset_hub(hub: Optional[PriceHub] = None) -> None:
    """Test hook: replace (or clear) the process-wide hub."""
    global _hub
    _hub = hub
