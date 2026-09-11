import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const dirname = path.dirname(fileURLToPath(import.meta.url));

// Minimal test setup -- covers the pure-logic hooks/utilities under
// src/lib, not a full component-level test suite. Started specifically
// to regression-test useRealtimeVoiceInput.ts/useWakeWordListener.ts's
// unmount cleanup (see their own comments), not as a general-purpose
// rollout; scope it up from here as more of the frontend gets covered.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
  resolve: {
    alias: {
      // Mirrors tsconfig.json's own "@/*" -> "./src/*" path mapping --
      // vitest doesn't read tsconfig paths on its own.
      "@": path.resolve(dirname, "./src"),
    },
  },
});
