import { expect, test } from "@playwright/test";

// Tailwind silently drops a class it cannot resolve, so check in a real browser that the token utilities used by the
// component library produce the values documented in src/components/ui/tokens.ts.

const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

test("token utilities resolve to their documented values", async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem("finsight:onboarding-seen-v1", "1"));
  await page.route("**/api/**", (route) => route.fulfill(json({ message: "down" }, 503)));
  await page.route("**/api/auth/me", (route) =>
    route.fulfill(json({ user_id: "u1", email: "t@example.com", onboarding_completed: true })),
  );

  await page.goto("/watchlist");
  // The failed watchlist load shows LoadError, whose RETRY is a size="sm" Button.
  const retry = page.getByRole("button", { name: "RETRY" });
  await expect(retry).toBeVisible();

  const styles = await retry.evaluate((el) => {
    const s = getComputedStyle(el);
    return { fontSize: s.fontSize, minHeight: s.minHeight, borderRadius: s.borderRadius, paddingLeft: s.paddingLeft };
  });
  expect(styles.fontSize).toBe("10px"); // text-micro
  expect(styles.minHeight).toBe("32px"); // min-h-8
  expect(styles.borderRadius).toBe("8px"); // rounded-control
  expect(styles.paddingLeft).toBe("10px"); // px-2.5

  // Remaining token utilities, checked on throwaway elements.
  const resolved = await page.evaluate(() => {
    const probe = (cls: string, prop: string) => {
      const el = document.createElement("div");
      el.className = cls;
      document.body.appendChild(el);
      const v = getComputedStyle(el).getPropertyValue(prop);
      el.remove();
      return v;
    };
    return {
      caption: probe("text-caption", "font-size"),
      small: probe("text-small", "font-size"),
      body: probe("text-body", "font-size"),
      title: probe("text-title", "font-size"),
      display: probe("text-display", "font-size"),
      gutter: probe("px-gutter", "padding-left"),
      cardX: probe("px-card-x", "padding-left"),
      cardY: probe("py-card-y", "padding-top"),
    };
  });
  expect(resolved).toEqual({ caption: "11px", small: "12px", body: "14px", title: "18px", display: "20px", gutter: "20px", cardX: "14px", cardY: "10px" });
});
