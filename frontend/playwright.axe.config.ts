import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config dedicated to the axe accessibility gate. Spins up
 * the static Ladle catalog (`pnpm ladle:preview`) and walks every story
 * URL with `@axe-core/playwright`. Kept separate from the (yet-to-exist)
 * end-to-end suite so the a11y job can short-circuit on its own.
 */
export default defineConfig({
  testDir: "./tests/axe",
  timeout: 30_000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:61000",
    trace: "off",
    actionTimeout: 5_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "pnpm ladle:preview --port=61000",
    port: 61000,
    timeout: 60_000,
    reuseExistingServer: !process.env.CI,
  },
});
