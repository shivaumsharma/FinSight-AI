// WebSocket client for GET /v1/prices/stream: connect with a short-lived token, subscribe, deliver ticks, and reconnect
// on its own when the connection drops or goes silent. No React in here, so it can be tested with fake timers.

export interface Tick {
  t: string; // ticker
  p: number; // price
  c: number | null; // change vs previous close, percent
  cur: string; // currency code
  ts: number; // unix seconds when the backend read the quote
}

// connecting: first attempt. live: receiving. reconnecting: dropped after having been live (or retrying early on).
// unavailable: repeated failures without ever connecting (feed not reachable); still retries, slowly.
export type StreamStatus = "idle" | "connecting" | "live" | "reconnecting" | "unavailable";

export interface SocketLike {
  readyState: number;
  onopen: (() => void) | null;
  onmessage: ((e: { data: unknown }) => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  send(data: string): void;
  close(): void;
}

export interface PriceStreamOptions {
  getEndpoint: () => Promise<{ token: string; ws_url: string }>;
  onTicks: (ticks: Tick[]) => void;
  onStatus: (status: StreamStatus) => void;
  createSocket?: (url: string) => SocketLike;
  // The backend sends a heartbeat every 15s when idle, so silence this long means the link is dead.
  staleAfterMs?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  unavailableAfterFailures?: number;
  random?: () => number;
}

export class PriceStream {
  private o: Required<Omit<PriceStreamOptions, "createSocket">> & { createSocket: (url: string) => SocketLike };
  private socket: SocketLike | null = null;
  private generation = 0; // handlers from an older socket must not act
  private started = false;
  private everLive = false;
  private failures = 0;
  private tickers: string[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setTimeout> | null = null;
  private status: StreamStatus = "idle";

  constructor(options: PriceStreamOptions) {
    this.o = {
      staleAfterMs: 35_000,
      baseDelayMs: 1_000,
      maxDelayMs: 30_000,
      unavailableAfterFailures: 5,
      random: Math.random,
      createSocket: (url) => new WebSocket(url) as unknown as SocketLike,
      ...options,
    };
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.setStatus("connecting");
    void this.connect();
  }

  stop(): void {
    this.started = false;
    this.generation++;
    this.clearTimers();
    const s = this.socket;
    this.socket = null;
    if (s) {
      s.onclose = s.onerror = s.onmessage = s.onopen = null;
      s.close();
    }
    this.setStatus("idle");
  }

  setTickers(tickers: string[]): void {
    this.tickers = tickers;
    this.sendSubscribe();
  }

  // For the browser's "online" event: don't wait out the backoff.
  reconnectNow(): void {
    if (!this.started || this.status === "live") return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.failures = 0;
    void this.connect();
  }

  getStatus(): StreamStatus {
    return this.status;
  }

  // ------------------------------------------------------------------ internals
  private setStatus(next: StreamStatus): void {
    if (next === this.status) return;
    this.status = next;
    this.o.onStatus(next);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.watchdog) clearTimeout(this.watchdog);
    this.reconnectTimer = this.watchdog = null;
  }

  private sendSubscribe(): void {
    if (this.status === "live" && this.socket) {
      this.socket.send(JSON.stringify({ type: "subscribe", tickers: this.tickers }));
    }
  }

  private armWatchdog(generation: number): void {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = setTimeout(() => {
      if (generation !== this.generation) return;
      this.drop(generation); // silent connection: treat as dropped
    }, this.o.staleAfterMs);
  }

  private async connect(): Promise<void> {
    if (!this.started) return;
    const generation = ++this.generation;
    let endpoint: { token: string; ws_url: string };
    try {
      endpoint = await this.o.getEndpoint();
    } catch {
      if (generation === this.generation) this.scheduleReconnect();
      return;
    }
    if (generation !== this.generation || !this.started) return;

    const socket = this.o.createSocket(endpoint.ws_url);
    this.socket = socket;
    socket.onopen = () => {
      if (generation === this.generation) socket.send(JSON.stringify({ token: endpoint.token }));
    };
    socket.onmessage = (event) => {
      if (generation !== this.generation || typeof event.data !== "string") return;
      this.armWatchdog(generation);
      let msg: { type?: string; ticks?: Tick[] };
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === "hello") {
        this.everLive = true;
        this.failures = 0;
        this.setStatus("live");
        this.sendSubscribe();
      } else if (msg.type === "ticks" && Array.isArray(msg.ticks)) {
        this.o.onTicks(msg.ticks);
      }
    };
    socket.onclose = socket.onerror = () => this.drop(generation);
  }

  private drop(generation: number): void {
    if (generation !== this.generation || !this.started) return;
    this.generation++; // ignore anything else this socket does
    if (this.watchdog) clearTimeout(this.watchdog);
    const s = this.socket;
    this.socket = null;
    if (s) {
      s.onclose = s.onerror = s.onmessage = s.onopen = null;
      try {
        s.close();
      } catch {
        // already closed
      }
    }
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (!this.started) return;
    this.failures++;
    this.setStatus(!this.everLive && this.failures >= this.o.unavailableAfterFailures ? "unavailable" : this.everLive ? "reconnecting" : "connecting");
    const exp = Math.min(this.o.maxDelayMs, this.o.baseDelayMs * 2 ** (this.failures - 1));
    const delay = Math.round(exp * (0.5 + this.o.random() * 0.5));
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect();
    }, delay);
  }
}
