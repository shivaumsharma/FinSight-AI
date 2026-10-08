import { defineConfig } from "@playwright/test";

const PORT = 3218;

// Render benchmark: a production build with React's profiling build enabled, served on its own port.
// Run with: npm run bench
export default defineConfig({
  testDir: "./bench",
  timeout: 180_000,
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://localhost:${PORT}` },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: {
    command: `npm run build && npm run start -- -p ${PORT}`,
    env: { NEXT_PUBLIC_BENCH: "1" },
    url: `http://localhost:${PORT}/bench`,
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
