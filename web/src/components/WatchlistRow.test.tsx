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
});
