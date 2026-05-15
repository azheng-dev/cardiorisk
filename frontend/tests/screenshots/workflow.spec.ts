import path from "node:path";
import { fileURLToPath } from "node:url";
import { type Page, test } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Captures README/marketing screenshots of the 5 Phase-5.3 workflow
 * screens in both light and dark themes. Outputs land in
 * `docs/design/screenshots/` with predictable filenames so the README
 * walkthrough can reference them directly.
 *
 * Not part of the CI test matrix — invoked locally via `pnpm screenshots`
 * when the visual story needs refreshing.
 */

const OUT_DIR = path.resolve(__dirname, "../../../docs/design/screenshots");

async function startMockCase(page: Page): Promise<string> {
  await page.goto("/cases/new");
  await page.getByRole("button", { name: /Triage & score risk/i }).click();
  await page.waitForURL(/\/cases\/.+\/risk$/);
  const m = page.url().match(/\/cases\/([^/]+)\//);
  return m?.[1] ?? "mock-001";
}

async function snap(page: Page, name: string, theme: string) {
  await page.waitForTimeout(350);
  await page.screenshot({
    path: path.join(OUT_DIR, `${name}-${theme}.png`),
    fullPage: true,
    animations: "disabled",
  });
}

/**
 * The mock store is in-memory per page session, so we must walk via
 * client-side navigation instead of `page.goto` once the case exists.
 * Otherwise the second `goto` reloads, the store empties, and every
 * downstream screen renders the "case not found" empty state.
 */
async function clickWorkflowLink(page: Page, label: RegExp) {
  await page
    .getByRole("complementary", { name: /Workflow navigation/i })
    .getByRole("link", {
      name: label,
    })
    .click();
}

test.describe("workflow screenshots", () => {
  test("home", async ({ page }, testInfo) => {
    await page.goto("/");
    await snap(page, "home", testInfo.project.name);
  });

  test("new-case", async ({ page }, testInfo) => {
    await page.goto("/cases/new");
    await snap(page, "new-case", testInfo.project.name);
  });

  test("workflow", async ({ page }, testInfo) => {
    await startMockCase(page);
    await snap(page, "risk", testInfo.project.name);
    await clickWorkflowLink(page, /Guideline/i);
    await page.waitForURL(/\/guideline$/);
    await snap(page, "guideline", testInfo.project.name);
    await clickWorkflowLink(page, /Letter/i);
    await page.waitForURL(/\/letter$/);
    await snap(page, "letter", testInfo.project.name);
    await clickWorkflowLink(page, /Audit log/i);
    await page.waitForURL(/\/audit$/);
    await snap(page, "audit", testInfo.project.name);
  });
});
