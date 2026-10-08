import { expect, test, type Page } from "@playwright/test";

// Log in, add a stock to the watchlist, place a simulated order. The backend is mocked in the browser so the test is
// deterministic; it exercises the real UI, validation, confirmation step and refresh logic.

interface State {
  loggedIn: boolean;
  watchlist: Array<Record<string, unknown>>;
  orders: Array<Record<string, unknown>>;
  orderBodies: Array<Record<string, unknown>>;
}

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

async function mockBackend(page: Page): Promise<State> {
  const state: State = { loggedIn: false, watchlist: [], orders: [], orderBodies: [] };

  // The first-visit tour overlay would block clicks; mark it as already seen.
  await page.addInitScript(() => window.localStorage.setItem("finsight:onboarding-seen-v1", "1"));

  // Anything not mocked below fails cleanly instead of hitting a real backend; components show their error states.
  await page.route("**/api/**", (route) => route.fulfill(json({ message: "not mocked" }, 404)));

  await page.route("**/api/auth/me", (route) =>
    state.loggedIn
      ? route.fulfill(json({ user_id: "u1", email: "trader@example.com", onboarding_completed: true, risk_tolerance: "Moderate" }))
      : route.fulfill(json({ message: "Not signed in" }, 401)),
  );
  await page.route("**/api/auth/login", (route) => {
    state.loggedIn = true;
    return route.fulfill(json({ ok: true }));
  });
  await page.route("**/api/companies/suggest**", (route) => route.fulfill(json({ suggestions: [] })));

  await page.route("**/api/watchlist", async (route) => {
    if (route.request().method() === "POST") {
      const { ticker } = route.request().postDataJSON();
      state.watchlist.push({
        ticker: String(ticker).toUpperCase(), price: 233.1, change_pct: 1.4, currency: "USD", rating: "Buy", added_at: 1,
        next_earnings_date: null, next_ex_dividend_date: null, last_dividend_amount: null, last_split: null,
      });
      return route.fulfill(json({ ok: true }));
    }
    return route.fulfill(json({ items: state.watchlist }));
  });

  await page.route("**/api/orders**", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      state.orderBodies.push(body);
      state.orders.unshift({
        order_id: "o1", ticker: String(body.ticker).toUpperCase(), side: body.side, quantity: body.quantity,
        execution_price: 233.1, currency: "USD", executed_at: 1, rationale: null,
      });
      return route.fulfill(json({ ok: true }));
    }
    return route.fulfill(json({ orders: state.orders }));
  });

  await page.route("**/api/portfolio", (route) => route.fulfill(json({ holdings: [], summary: null })));
  return state;
}

test("log in, add to the watchlist, place a simulated order", async ({ page }) => {
  const state = await mockBackend(page);

  await page.goto("/");
  await page.getByPlaceholder("email").fill("trader@example.com");
  await page.getByPlaceholder("password").fill("correct-horse-battery");
  await page.getByRole("button", { name: "LOG IN" }).last().click();
  await expect(page.getByText("TRADE (SIMULATED)")).toBeVisible();

  // Watchlist
  await page.goto("/watchlist");
  await page.getByPlaceholder("add ticker or company name...").fill("aapl");
  await page.getByRole("button", { name: "ADD" }).click();
  const row = page.getByTestId("watchlist-row-AAPL");
  await expect(row).toBeVisible();
  await expect(row).toContainText("$233.10");
  await expect(row).toContainText("+1.40%");

  // Order: invalid input is stopped, then review, then confirm
  await page.goto("/");
  await page.getByText("+ NEW ORDER").click();
  await page.getByRole("button", { name: "Review order" }).click();
  await expect(page.getByText("Enter a quantity.")).toBeVisible();

  await page.getByLabel("Ticker or company").fill("AAPL");
  await page.getByLabel("Quantity (shares)").fill("10");
  await page.getByRole("button", { name: "Review order" }).click();
  await expect(page.getByTestId("order-review")).toContainText("BUY 10 AAPL");
  expect(state.orderBodies).toHaveLength(0);

  await page.getByRole("button", { name: "Confirm buy (simulated)" }).click();
  await expect(page.getByText("+ NEW ORDER")).toBeVisible();
  await expect(page.getByText("BUY").first()).toBeVisible();
  expect(state.orderBodies).toEqual([{ ticker: "AAPL", side: "BUY", quantity: 10 }]);
});
