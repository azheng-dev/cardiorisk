import AxeBuilder from "@axe-core/playwright";
import { type Page, expect, test } from "@playwright/test";

/**
 * Page-level axe gate. Walks the 5 Phase-5.3 routes against the
 * Next.js production build with `NEXT_PUBLIC_AGENT_MOCK=true` so a
 * single deterministic case populates the screens. Fails on any
 * serious or critical WCAG 2.x A/AA violation.
 *
 * The page-level checks complement the Phase 5.2 Ladle gate
 * (`tests/axe/catalog.spec.ts`) — that one covers individual
 * primitives, this one covers their composition into screens.
 */

const SEVERITIES = ["serious", "critical"] as const;
type BlockingImpact = (typeof SEVERITIES)[number];

const EXEMPTED_RULE_IDS = new Set<string>([
  // Same upstream cmdk portal quirks as the Ladle gate; tracked in ADR-021.
  "aria-required-children",
  "aria-required-parent",
]);

async function startMockCase(page: Page): Promise<string> {
  await page.goto("/cases/new");
  // The "Triage & score risk" submit button leads to /cases/[id]/risk.
  await page.getByRole("button", { name: /Triage & score risk/i }).click();
  await page.waitForURL(/\/cases\/.+\/risk$/);
  const m = page.url().match(/\/cases\/([^/]+)\//);
  return m?.[1] ?? "mock-001";
}

async function runAxe(page: Page, label: string) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const blocking = results.violations.filter(
    (v) =>
      SEVERITIES.includes((v.impact ?? "minor") as BlockingImpact) && !EXEMPTED_RULE_IDS.has(v.id),
  );
  expect(blocking, `[${label}] ${JSON.stringify(blocking, null, 2)}`).toEqual([]);
}

test.describe("workflow screens", () => {
  test("/ home", async ({ page }) => {
    await page.goto("/");
    await runAxe(page, "home");
  });

  test("/cases/new", async ({ page }) => {
    await page.goto("/cases/new");
    await runAxe(page, "new-case");
  });

  test("/cases/[id]/risk", async ({ page }) => {
    const id = await startMockCase(page);
    await page.goto(`/cases/${id}/risk`);
    await runAxe(page, "risk");
  });

  test("/cases/[id]/guideline", async ({ page }) => {
    const id = await startMockCase(page);
    await page.goto(`/cases/${id}/guideline`);
    await runAxe(page, "guideline");
  });

  test("/cases/[id]/letter", async ({ page }) => {
    const id = await startMockCase(page);
    await page.goto(`/cases/${id}/letter`);
    await runAxe(page, "letter");
  });

  test("/cases/[id]/audit", async ({ page }) => {
    const id = await startMockCase(page);
    await page.goto(`/cases/${id}/audit`);
    await runAxe(page, "audit");
  });
});
