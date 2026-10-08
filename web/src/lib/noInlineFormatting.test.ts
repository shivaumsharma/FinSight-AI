import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

// Numbers become text in numberFormat.ts only. This fails when an inline toFixed/toLocaleString/Intl.NumberFormat
// creeps back into a component, so edge cases (missing, NaN, -0, grouping) stay handled in one place.
const SRC = join(__dirname, "..");
const ALLOWED = new Set(["lib/numberFormat.ts", "app/profile/page.tsx"]); // profile formats a Date, not a number

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) return files(full);
    return /\.(ts|tsx)$/.test(name) && !/\.test\.(ts|tsx)$/.test(name) ? [full] : [];
  });
}

describe("number formatting", () => {
  it("has no inline toFixed, toLocaleString or Intl.NumberFormat outside the formatter", () => {
    const offenders = files(SRC)
      .map((f) => relative(SRC, f).replace(/\\/g, "/"))
      .filter((rel) => !ALLOWED.has(rel))
      .filter((rel) => /\.toFixed\(|\.toLocaleString\(|Intl\.NumberFormat/.test(readFileSync(join(SRC, rel), "utf8")));
    expect(offenders).toEqual([]);
  });
});
