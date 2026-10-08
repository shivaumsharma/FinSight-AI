import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { COLOR_TOKENS, RADIUS_TOKENS, SPACING_TOKENS, TYPE_TOKENS } from "./tokens";

const css = readFileSync(path.resolve(__dirname, "../../app/globals.css"), "utf-8");

function declared(block: string, name: string): string | undefined {
  return new RegExp(`${name.replace(/[-]/g, "\-")}:\s*([^;]+);`).exec(block)?.[1].trim();
}

describe("design tokens match globals.css", () => {
  it.each(Object.entries({ ...TYPE_TOKENS, ...SPACING_TOKENS, ...RADIUS_TOKENS }))("%s", (_name, token) => {
    expect(declared(css, token.cssVar)).toBe(token.value);
    expect(Number.parseFloat(token.value) * 16).toBe(token.px);
  });

  it("dark and light colours match", () => {
    const dark = /:root\s*{([^}]*)}/.exec(css)![1];
    const light = /\[data-theme="light"\]\s*{([^}]*)}/.exec(css)![1];
    for (const [name, v] of Object.entries(COLOR_TOKENS)) {
      expect(declared(dark, `--${name}`), `dark ${name}`).toBe(v.dark);
      expect(declared(light, `--${name}`), `light ${name}`).toBe(v.light);
    }
  });

  it("the type scale only gets larger", () => {
    const sizes = Object.values(TYPE_TOKENS).map((t) => t.px);
    expect([...sizes].sort((a, b) => a - b)).toEqual(sizes);
  });
});
