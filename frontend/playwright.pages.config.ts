import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for the page-level accessibility gate. Boots the
 * Next.js production build with `NEXT_PUBLIC_AGENT_MOCK=true` and
 * walks the 5 Phase-5.3 routes through `@axe-core/playwright`. Kept
 * separate from the Ladle gate (`playwright.axe.config.ts`) because:
 *   - The two webServers run on different ports and could not share a
 *     single config without an awkward project-scoped webServer block.
 *   - It lets the Ladle gate short-circuit on its own in CI.
 */
export default defineConfig({
  testDir: "./tests/axe-pages",
  timeout: 60_000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:61001",
    trace: "off",
    actionTimeout: 5_000,
  },
  projects: [
    {
      name: "chromium-light",
      use: { ...devices["Desktop Chrome"], colorScheme: "light" },
    },
    {
      name: "chromium-dark",
      use: { ...devices["Desktop Chrome"], colorScheme: "dark" },
    },
  ],
  webServer: {
    // NEXT_PUBLIC_* values are inlined at build time, so the build step has
    // to run with the mock flag set even though `pnpm start` re-exports it.
    // Always rebuild here so a stale `.next/` from a non-mock build cannot
    // bleed through and cause the agent client to hit the live API.
    command:
      "NEXT_PUBLIC_AGENT_MOCK=true pnpm build && NEXT_PUBLIC_AGENT_MOCK=true pnpm start --port=61001",
    port: 61001,
    timeout: 240_000,
    reuseExistingServer: !process.env.CI,
  },
});
