import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import WatchlistRow from "./WatchlistRow";
import type { WatchlistItem } from "@/lib/types";

const item = (over: Partial<WatchlistItem> = {}): WatchlistItem => ({
  ticker: "AAPL",
  price: 233.1,
  change_pct: 1.4,
  currency: "USD",
  rating: "Buy",
  added_at: 0,
  next_earnings_date: null,
  next_ex_dividend_date: null,
  last_dividend_amount: null,
  last_split: null,
  ...over,
});

describe("WatchlistRow", () => {
  it("shows the ticker, a formatted price and a signed percent", () => {
    render(<WatchlistRow item={item()} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("$233.10")).toBeInTheDocument();
    expect(screen.getByText("+1.40%")).toBeInTheDocument();
  });

  it("colours a falling stock as a loss and keeps the minus sign", () => {
    render(<WatchlistRow item={item({ change_pct: -0.86 })} />);
    const pct = screen.getByText("-0.86%");
    expect(pct.className).toContain("text-danger");
  });

  it("never shows -0.00% or +0.00% for a flat day", () => {
    render(<WatchlistRow item={item({ change_pct: -0.004 })} />);
    expect(screen.getByText("0.00%")).toBeInTheDocument();
  });

  it("uses the rupee symbol for INR quotes", () => {
    render(<WatchlistRow item={item({ ticker: "TCS.NS", price: 3500, currency: "INR" })} />);
    expect(screen.getByText("₹3,500.00")).toBeInTheDocument();
  });

  it("omits the price block when the quote is missing, instead of showing a wrong number", () => {
    render(<WatchlistRow item={item({ price: null, change_pct: null })} />);
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("says a ticker has not been researched when there is no rating", () => {
    render(<WatchlistRow item={item({ rating: null })} />);
    expect(screen.getByText("not yet researched")).toBeInTheDocument();
  });

  it("links to the stock page", () => {
    render(<WatchlistRow item={item()} />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "/stock/AAPL");
  });

  it("removes by ticker, and a demo row has no remove button", () => {
    const onRemove = vi.fn();
    const { rerender } = render(<WatchlistRow item={item()} onRemove={onRemove} />);
    fireEvent.click(screen.getByRole("button", { name: "Remove AAPL from watchlist" }));
    expect(onRemove).toHaveBeenCalledWith("AAPL");
    rerender(<WatchlistRow item={item()} isDemo onRemove={onRemove} />);
    expect(screen.queryByRole("button", { name: "Remove AAPL from watchlist" })).not.toBeInTheDocument();
  });

  it("prefers a live quote over the loaded price and change", () => {
    render(<WatchlistRow item={item()} live={{ price: 240.5, changePct: -0.25, currency: "USD", ts: 1 }} />);
    expect(screen.getByText("$240.50")).toBeInTheDocument();
    expect(screen.getByText("-0.25%")).toBeInTheDocument();
    expect(screen.queryByText("$233.10")).not.toBeInTheDocument();
  });

  it("flashes green when the price rises and red when it falls, and not on first paint", () => {
    const quote = (price: number) => ({ price, changePct: 0, currency: "USD", ts: 1 });
    const { rerender } = render(<WatchlistRow item={item()} live={quote(233.1)} />);
    expect(screen.getByText("$233.10")).not.toHaveAttribute("data-flash");

    rerender(<WatchlistRow item={item()} live={quote(234)} />);
    expect(screen.getByText("$234.00")).toHaveAttribute("data-flash", "up");
    expect(screen.getByText("$234.00").className).toContain("flash-up");

    rerender(<WatchlistRow item={item()} live={quote(233)} />);
    expect(screen.getByText("$233.00")).toHaveAttribute("data-flash", "down");
  });

  it("does not flash when a re-render brings the same price", () => {
    const live = { price: 233.1, changePct: 0, currency: "USD", ts: 1 };
    const { rerender } = render(<WatchlistRow item={item()} live={live} />);
    rerender(<WatchlistRow item={item({ rating: "Hold" })} live={{ ...live }} />);
    expect(screen.getByText("$233.10")).not.toHaveAttribute("data-flash");
  });
});
