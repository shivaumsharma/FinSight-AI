import { expect, test, type WebSocketRoute } from "@playwright/test";

// Drives the real watchlist in a real browser against a scripted WebSocket server: live ticks update the row and
// flash, a dropped connection shows the reconnecting banner, and the client reconnects and resubscribes on its own.

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

test("watchlist prices update live, flash, and recover from a dropped connection", async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem("finsight:onboarding-seen-v1", "1"));
  await page.route("**/api/**", (route) => route.fulfill(json({ message: "not mocked" }, 404)));
  await page.route("**/api/auth/me", (route) =>
    route.fulfill(json({ user_id: "u1", email: "trader@example.com", onboarding_completed: true })),
  );
  await page.route("**/api/watchlist", (route) =>
    route.fulfill(
      json({
        items: [
          { ticker: "AAPL", price: 233.1, change_pct: 1.4, currency: "USD", rating: "Buy", added_at: 1, next_earnings_date: null, next_ex_dividend_date: null, last_dividend_amount: null, last_split: null },
        ],
      }),
    ),
  );
  await page.route("**/api/prices/stream-token", (route) =>
    route.fulfill(json({ token: "tok", expires_in: 60, ws_url: "ws://localhost:3217/mock-prices" })),
  );

  const connections: WebSocketRoute[] = [];
  const received: unknown[][] = [];
  await page.routeWebSocket("**/mock-prices", (ws) => {
    const index = connections.length;
    connections.push(ws);
    received[index] = [];
    ws.onMessage((raw) => {
      const msg = JSON.parse(String(raw));
      received[index].push(msg);
      if (msg.token) ws.send(JSON.stringify({ type: "hello", interval: 5, source: "yahoo-poll" }));
    });
  });

  const tick = (price: number, change = 1.4) =>
    JSON.stringify({ type: "ticks", ticks: [{ t: "AAPL", p: price, c: change, cur: "USD", ts: Math.floor(Date.now() / 1000) }] });

  await page.goto("/watchlist");
  const row = page.getByTestId("watchlist-row-AAPL");
  await expect(row).toContainText("$233.10");

  // Connected: authenticated with the token and subscribed to the watchlist's tickers.
  await expect.poll(() => received[0]?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(received[0][0]).toEqual({ token: "tok" });
  expect(received[0][1]).toEqual({ type: "subscribe", tickers: ["AAPL"] });
  await expect(page.getByTestId("connection-banner")).toHaveCount(0);

  // A rising tick updates the price and the percent, and flashes green.
  connections[0].send(tick(234.5, 2.0));
  await expect(row).toContainText("$234.50");
  await expect(row).toContainText("+2.00%");
  await expect(row.locator("[data-flash]")).toHaveAttribute("data-flash", "up");

  // A falling tick flashes red.
  connections[0].send(tick(233.9, 1.7));
  await expect(row).toContainText("$233.90");
  await expect(row.locator("[data-flash]")).toHaveAttribute("data-flash", "down");

  // Drop the connection: the banner appears, the last price stays on screen.
  await connections[0].close({ code: 1006, reason: "network lost" });
  await expect(page.getByTestId("connection-banner")).toContainText("Reconnecting");
  await expect(row).toContainText("$233.90");

  // The client reconnects by itself, re-authenticates, resubscribes, and the banner clears.
  await expect.poll(() => connections.length, { timeout: 10_000 }).toBe(2);
  await expect.poll(() => received[1]?.length ?? 0).toBeGreaterThanOrEqual(2);
  expect(received[1][0]).toEqual({ token: "tok" });
  expect(received[1][1]).toEqual({ type: "subscribe", tickers: ["AAPL"] });
  await expect(page.getByTestId("connection-banner")).toHaveCount(0);

  connections[1].send(tick(236, 2.5));
  await expect(row).toContainText("$236.00");
});

test("the render benchmark page is not served in a normal build", async ({ request }) => {
  expect((await request.get("/bench")).status()).toBe(404);
});
