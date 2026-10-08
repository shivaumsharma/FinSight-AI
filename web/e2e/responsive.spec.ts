import { expect, test, type Page } from "@playwright/test";

// At phone width no page may scroll sideways, and tappable controls should not be tiny. The backend is mocked with
// realistic, awkward data (long names, big numbers) so layout problems show up.

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

const LONG_NAME = "International Business Machines Corporation Depositary Shares";

async function mock(page: Page) {
  await page.addInitScript(() => window.localStorage.setItem("finsight:onboarding-seen-v1", "1"));
  await page.route("**/api/**", (route) => route.fulfill(json({ message: "not mocked" }, 404)));
  await page.route("**/api/auth/me", (route) =>
    route.fulfill(json({ user_id: "u1", email: "trader@example.com", onboarding_completed: true, risk_tolerance: "Moderate" })),
  );
  await page.route("**/api/watchlist", (route) =>
    route.fulfill(
      json({
        items: ["AAPL", "BRK-B", "RELIANCE.NS"].map((t, i) => ({
          ticker: t, price: 1234567.891 * (i + 1), change_pct: i % 2 ? -12.345 : 1.2, currency: i === 2 ? "INR" : "USD", rating: "Buy",
          added_at: 1, next_earnings_date: "2026-11-02", next_ex_dividend_date: "2026-11-09", last_dividend_amount: null, last_split: null,
        })),
      }),
    ),
  );
  await page.route("**/api/market/indices", (route) =>
    route.fulfill(json({ indices: [{ ticker: "^GSPC", name: LONG_NAME, price: 5000.5, change_pct: 0.5, region: "global" }] })),
  );
  await page.route("**/api/orders**", (route) =>
    route.fulfill(
      json({ orders: [{ order_id: "o1", ticker: "AAPL", side: "BUY", quantity: 1234.567891, execution_price: 233.1, currency: "USD", executed_at: 1, rationale: LONG_NAME }] }),
    ),
  );
  await page.route("**/api/portfolio", (route) => route.fulfill(json({ holdings: [], summary: null })));
  await page.route("**/api/companies/suggest**", (route) => route.fulfill(json({ suggestions: [] })));
}

const PAGES = ["/", "/watchlist", "/indices", "/screener", "/scoreboard", "/reports", "/profile", "/calculators", "/corporate-actions", "/chat"];

async function overflowOffenders(page: Page) {
  return page.evaluate(() => {
    const width = document.documentElement.clientWidth;
    const out: string[] = [];
    for (const el of Array.from(document.body.querySelectorAll<HTMLElement>("*"))) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      if (r.right > width + 1) {
        // Elements inside a horizontally scrolling strip are allowed to extend past the edge.
        let p: HTMLElement | null = el.parentElement;
        let scrollable = false;
        while (p && p !== document.body) {
          if (/(auto|scroll)/.test(getComputedStyle(p).overflowX)) { scrollable = true; break; }
          p = p.parentElement;
        }
        if (!scrollable) out.push(`${el.tagName.toLowerCase()}.${String(el.className).slice(0, 60)} right=${Math.round(r.right)} (viewport ${width})`);
      }
    }
    return out.slice(0, 8);
  });
}

test.describe("375px layout", () => {
  test.skip(({ viewport }) => (viewport?.width ?? 0) > 400, "phone project only");

  for (const path of PAGES) {
    test(`${path} does not scroll sideways`, async ({ page }) => {
      await mock(page);
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      const doc = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
      const offenders = await overflowOffenders(page);
      expect(offenders, `elements past the right edge on ${path}`).toEqual([]);
      expect(doc.scroll, `document is wider than the screen on ${path}`).toBeLessThanOrEqual(doc.client + 1);
    });
  }

  // WCAG 2.2 AA minimum target size is 24x24 CSS px.
  for (const path of ["/", "/watchlist", "/screener"]) {
    test(`controls on ${path} are at least 24px tall and wide`, async ({ page }) => {
      await mock(page);
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      const tooSmall = await page.evaluate(() =>
        Array.from(document.querySelectorAll<HTMLElement>("button, a[href], input:not([type=hidden]), select"))
          .map((el) => ({ el, r: el.getBoundingClientRect() }))
          .filter(({ r }) => r.width > 0 && r.height > 0 && (r.height < 24 || r.width < 24))
          .map(({ el, r }) => `${el.tagName.toLowerCase()} "${(el.textContent || el.getAttribute("title") || el.getAttribute("aria-label") || "").trim().slice(0, 24)}" ${Math.round(r.width)}x${Math.round(r.height)}`),
      );
      expect(tooSmall).toEqual([]);
    });
  }
});
