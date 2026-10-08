import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OrderTicket from "./OrderTicket";

type Handler = (url: string, init?: RequestInit) => { status: number; body: unknown };

let calls: Array<{ url: string; init?: RequestInit }>;

function mockBackend(onOrder: Handler) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      const { status, body } =
        url.startsWith("/api/orders") && init?.method === "POST"
          ? onOrder(url, init)
          : url.startsWith("/api/orders")
            ? { status: 200, body: { orders: [] } }
            : { status: 200, body: { suggestions: [] } };
      return { ok: status >= 200 && status < 300, status, json: async () => body } as Response;
    }),
  );
}

async function openForm() {
  render(<OrderTicket />);
  fireEvent.click(await screen.findByText("+ NEW ORDER"));
}

function fill(ticker: string, quantity: string) {
  fireEvent.change(screen.getByLabelText("Ticker or company"), { target: { value: ticker } });
  fireEvent.change(screen.getByLabelText("Quantity (shares)"), { target: { value: quantity } });
}

const posts = () => calls.filter((c) => c.init?.method === "POST");

beforeEach(() => mockBackend(() => ({ status: 200, body: { ok: true } })));
afterEach(() => vi.unstubAllGlobals());

describe("OrderTicket", () => {
  it("blocks an empty order and says what is missing, without calling the backend", async () => {
    await openForm();
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    expect(screen.getByText("Enter a ticker or company name.")).toBeInTheDocument();
    expect(screen.getByText("Enter a quantity.")).toBeInTheDocument();
    expect(posts()).toHaveLength(0);
  });

  it.each(["0", "-3", "abc", "1.1234567"])("rejects quantity %j before any request", async (qty) => {
    await openForm();
    fill("AAPL", qty);
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    expect(screen.queryByTestId("order-review")).not.toBeInTheDocument();
    expect(screen.getAllByRole("alert").length).toBeGreaterThan(0);
    expect(posts()).toHaveLength(0);
  });

  it("shows a confirmation first and only submits after Confirm", async () => {
    await openForm();
    fill("aapl", "10");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    expect(screen.getByTestId("order-review")).toHaveTextContent("BUY 10 AAPL");
    expect(posts()).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Confirm buy (simulated)" }));
    await waitFor(() => expect(posts()).toHaveLength(1));
    expect(JSON.parse(posts()[0].init!.body as string)).toEqual({ ticker: "aapl", side: "BUY", quantity: 10 });
  });

  it("lets the user go back and edit from the confirmation", async () => {
    await openForm();
    fill("AAPL", "10");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit order" }));
    expect(screen.queryByTestId("order-review")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Review order" })).toBeInTheDocument();
  });

  it("drops back to edit if the quantity changes after review", async () => {
    await openForm();
    fill("AAPL", "10");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    fireEvent.change(screen.getByLabelText("Quantity (shares)"), { target: { value: "11" } });
    expect(screen.queryByTestId("order-review")).not.toBeInTheDocument();
  });

  it("shows the backend's reason when an order is rejected, and keeps the form open", async () => {
    mockBackend(() => ({ status: 400, body: { message: "You only hold 3 shares of AAPL." } }));
    await openForm();
    fireEvent.click(screen.getByRole("button", { name: "SELL" }));
    fill("AAPL", "50");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm sell (simulated)" }));
    expect(await screen.findByText("You only hold 3 shares of AAPL.")).toBeInTheDocument();
    expect(screen.getByLabelText("Quantity (shares)")).toHaveValue("50");
  });

  it("explains an outage plainly and says nothing was placed", async () => {
    mockBackend(() => ({ status: 503, body: { code: "BACKEND_UNREACHABLE" } }));
    await openForm();
    fill("AAPL", "1");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm buy (simulated)" }));
    expect(await screen.findByText(/Nothing was placed/)).toBeInTheDocument();
  });

  it("closes and clears the form after a successful order", async () => {
    await openForm();
    fill("AAPL", "2");
    fireEvent.click(screen.getByRole("button", { name: "Review order" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm buy (simulated)" }));
    await waitFor(() => expect(screen.queryByLabelText("Quantity (shares)")).not.toBeInTheDocument());
    expect(screen.getByText("+ NEW ORDER")).toBeInTheDocument();
  });
});
