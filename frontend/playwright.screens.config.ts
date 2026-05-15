import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for capturing marketing/README screenshots of the
 * 5 Phase-5.3 workflow screens in both themes. Reuses the same mock-mode
 * production build approach as the page-axe gate so the captures are
 * deterministic and reproducible from a clean clone.
 *
 * Run with `pnpm screenshots`. Outputs to `../docs/design/screenshots/`
 * via the spec at `tests/screenshots/workflow.spec.ts`.
 */
export default defineConfig({
  testDir: "./tests/screenshots",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:61002",
    trace: "off",
    actionTimeout: 5_000,
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  },
  projects: [
    {
      name: "light",
      use: { ...devices["Desktop Chrome"], colorScheme: "light" },
    },
    {
      name: "dark",
      use: { ...devices["Desktop Chrome"], colorScheme: "dark" },
    },
  ],
  webServer: {
    command:
      "NEXT_PUBLIC_AGENT_MOCK=true pnpm build && NEXT_PUBLIC_AGENT_MOCK=true pnpm start --port=61002",
    port: 61002,
    timeout: 240_000,
    reuseExistingServer: !process.env.CI,
  },
});
