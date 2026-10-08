import { describe, expect, it } from "vitest";
import { MAX_QUANTITY, orderErrorMessage, validateOrder } from "./orderValidation";

describe("validateOrder", () => {
  it("accepts a normal order and returns the parsed quantity", () => {
    expect(validateOrder({ ticker: " aapl ", quantity: "10" })).toEqual({ ok: true, errors: {}, quantity: 10 });
    expect(validateOrder({ ticker: "Reliance Industries", quantity: "2.5" }).quantity).toBe(2.5);
    expect(validateOrder({ ticker: "BRK-B", quantity: "0.000001" }).ok).toBe(true);
  });

  it.each(["", "   "])("rejects a blank ticker %j", (ticker) => {
    expect(validateOrder({ ticker, quantity: "1" }).errors.ticker).toMatch(/Enter a ticker/);
  });

  it("rejects ticker characters that cannot be a symbol or name", () => {
    expect(validateOrder({ ticker: "AAPL;DROP", quantity: "1" }).errors.ticker).toBeDefined();
    expect(validateOrder({ ticker: "<b>", quantity: "1" }).errors.ticker).toBeDefined();
    expect(validateOrder({ ticker: "A".repeat(41), quantity: "1" }).errors.ticker).toBeDefined();
  });

  it.each(["", " "])("rejects an empty quantity %j", (quantity) => {
    expect(validateOrder({ ticker: "AAPL", quantity }).errors.quantity).toMatch(/Enter a quantity/);
  });

  it.each(["0", "0.0", "-5", "abc", "1e3", "1,000", "--1", ".5", "5."])("rejects quantity %j", (quantity) => {
    const r = validateOrder({ ticker: "AAPL", quantity });
    expect(r.ok).toBe(false);
    expect(r.quantity).toBeUndefined();
  });

  it("rejects more than six decimals and anything above the cap", () => {
    expect(validateOrder({ ticker: "AAPL", quantity: "1.1234567" }).errors.quantity).toMatch(/decimal/);
    expect(validateOrder({ ticker: "AAPL", quantity: String(MAX_QUANTITY) }).ok).toBe(true);
    expect(validateOrder({ ticker: "AAPL", quantity: String(MAX_QUANTITY + 1) }).errors.quantity).toMatch(/exceed/);
  });

  it("reports ticker and quantity problems together", () => {
    const r = validateOrder({ ticker: "", quantity: "0" });
    expect(Object.keys(r.errors).sort()).toEqual(["quantity", "ticker"]);
  });
});

describe("orderErrorMessage", () => {
  it("uses the backend's own message for a rejected order", () => {
    expect(orderErrorMessage(400, { message: "You only hold 3 shares of AAPL." })).toBe("You only hold 3 shares of AAPL.");
  });

  it("explains expired sessions, validation failures, unknown tickers and outages without blaming the user", () => {
    expect(orderErrorMessage(401, {})).toMatch(/Sign in again/);
    expect(orderErrorMessage(422, { detail: [{ msg: "x" }] })).toMatch(/Check the ticker and quantity/);
    expect(orderErrorMessage(404, null)).toMatch(/couldn't find that ticker/);
    expect(orderErrorMessage(503, { code: "BACKEND_UNREACHABLE" })).toMatch(/Nothing was placed/);
  });

  it("falls back to a safe generic message", () => {
    expect(orderErrorMessage(400, undefined)).toBe("Couldn't place that order. Nothing was placed.");
    expect(orderErrorMessage(400, { message: 5 })).toBe("Couldn't place that order. Nothing was placed.");
  });
});
