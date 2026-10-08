// Order-ticket input rules and error wording, kept out of the component so they can be tested on their own.

import { formatQuantity } from "./numberFormat";

export const MAX_QUANTITY = 1_000_000;
export const MAX_QUANTITY_DECIMALS = 6;

export interface OrderInput {
  ticker: string;
  quantity: string;
}

export interface OrderValidation {
  ok: boolean;
  errors: { ticker?: string; quantity?: string };
  // The parsed quantity, present only when it is valid.
  quantity?: number;
}

const TICKER_OK = /^[A-Za-z0-9][A-Za-z0-9 .&'^-]{0,39}$/;
const QUANTITY_OK = /^\d+(\.\d+)?$/;

export function validateOrder({ ticker, quantity }: OrderInput): OrderValidation {
  const errors: OrderValidation["errors"] = {};
  const t = ticker.trim();
  const q = quantity.trim();

  if (!t) errors.ticker = "Enter a ticker or company name.";
  else if (!TICKER_OK.test(t)) errors.ticker = "Use letters, numbers and . - & only (up to 40 characters).";

  let parsed: number | undefined;
  if (!q) errors.quantity = "Enter a quantity.";
  else if (!QUANTITY_OK.test(q)) errors.quantity = "Enter a plain positive number, like 10 or 2.5.";
  else {
    const n = Number(q);
    const decimals = q.includes(".") ? q.split(".")[1].length : 0;
    if (!Number.isFinite(n) || n <= 0) errors.quantity = "Quantity must be greater than zero.";
    else if (decimals > MAX_QUANTITY_DECIMALS) errors.quantity = `Use at most ${MAX_QUANTITY_DECIMALS} decimal places.`;
    else if (n > MAX_QUANTITY) errors.quantity = `Quantity can't exceed ${formatQuantity(MAX_QUANTITY)}.`;
    else parsed = n;
  }

  return { ok: Object.keys(errors).length === 0, errors, quantity: parsed };
}

// Turns a failed order response into one readable sentence. The backend sends {message} for rejections it understands
// (for example selling more shares than held); validation failures and outages arrive in other shapes.
export function orderErrorMessage(status: number, body: unknown): string {
  const message = typeof (body as { message?: unknown })?.message === "string" ? (body as { message: string }).message : "";
  if (message) return message;
  if (status === 401) return "Your session has expired. Sign in again to place orders.";
  if (status === 422) return "That order wasn't accepted. Check the ticker and quantity and try again.";
  if (status === 404) return "We couldn't find that ticker. Check the symbol and try again.";
  if (status >= 500) return "The order service is unavailable right now. Nothing was placed; try again in a moment.";
  return "Couldn't place that order. Nothing was placed.";
}
