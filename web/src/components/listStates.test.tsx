import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Watchlist from "./Watchlist";
import IndicesCarousel from "./IndicesCarousel";
import MarketNews from "./MarketNews";

type Reply = { status: number; body: unknown } | "hang" | "network-error";

// Routes by URL prefix; a "hang" reply never resolves so the loading state stays on screen.
function mockFetch(routes: Record<string, Reply | (() => Reply)>) {
  const fn = vi.fn((url: string) => {
    const key = Object.keys(routes).find((k) => url.startsWith(k));
    const reply = key ? (typeof routes[key] === "function" ? (routes[key] as () => Reply)() : (routes[key] as Reply)) : { status: 404, body: {} };
    if (reply === "hang") return new Promise<Response>(() => {});
    if (reply === "network-error") return Promise.reject(new Error("offline"));
    return Promise.resolve({ ok: reply.status < 300, status: reply.status, json: async () => reply.body } as Response);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("Watchlist states", () => {
  it("shows skeleton rows while loading, not a blank gap", () => {
    mockFetch({ "/api/watchlist": "hang" });
    render(<Watchlist />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
  });

  it("shows the empty state with sample rows when the list is empty", async () => {
    mockFetch({ "/api/watchlist": { status: 200, body: { items: [] } } });
    render(<Watchlist />);
    expect(await screen.findByText(/Empty for now/)).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows an error with a working retry, never the empty state, when the load fails", async () => {
    let attempts = 0;
    mockFetch({
      "/api/watchlist": () => (++attempts === 1 ? { status: 500, body: {} } : { status: 200, body: { items: [] } }),
    });
    render(<Watchlist />);
    expect(await screen.findByText(/Couldn't load your watchlist/)).toBeInTheDocument();
    expect(screen.queryByText(/Empty for now/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "RETRY" }));
    expect(await screen.findByText(/Empty for now/)).toBeInTheDocument();
  });
});

describe("IndicesCarousel states", () => {
  it("shows a skeleton, then an error with retry when the request fails (not a silent blank)", async () => {
    mockFetch({ "/api/market/indices": { status: 502, body: {} } });
    render(<IndicesCarousel />);
    expect(await screen.findByText(/Couldn't load indices/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "RETRY" })).toBeInTheDocument();
  });

  it("recovers after a retry", async () => {
    let attempts = 0;
    mockFetch({
      "/api/market/indices": () =>
        ++attempts === 1
          ? "network-error"
          : { status: 200, body: { indices: [{ ticker: "^GSPC", name: "S&P 500", price: 5000.5, change_pct: 0.5, region: "global" }] } },
    });
    render(<IndicesCarousel />);
    fireEvent.click(await screen.findByRole("button", { name: "RETRY" }));
    expect(await screen.findByText("5,000.50")).toBeInTheDocument();
    expect(screen.getByText("+0.50%")).toBeInTheDocument();
  });
});

describe("MarketNews states", () => {
  it("shows an error with retry when the trending feed fails", async () => {
    mockFetch({ "/api/news/market": { status: 500, body: {} } });
    render(<MarketNews />);
    await waitFor(() => expect(screen.getByText(/Couldn't load market news/)).toBeInTheDocument());
  });

  it("renders nothing, not an error, when the feed works but is empty", async () => {
    mockFetch({ "/api/news/market": { status: 200, body: { articles: [] } } });
    const { container } = render(<MarketNews />);
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
    expect(container).toBeEmptyDOMElement();
  });
});
