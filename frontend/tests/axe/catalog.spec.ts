import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Walks every Ladle story (read from the static `meta.json` produced by
 * `ladle build`) and runs axe-core. Fails the suite on any serious or
 * critical violation. Anything below that bar (e.g. minor WCAG advisories
 * we've consciously deferred) is logged but does not block CI.
 *
 * Documented exemptions live in `EXEMPTED_RULE_IDS`. The bar is high:
 * each exemption has to be (a) a known upstream library quirk that is
 * audited as fine by manual screen-reader testing, and (b) tracked in
 * ADR-021 §"Accessibility exemptions" so it's discoverable.
 */

type LadleMeta = {
  stories: Record<string, unknown>;
};

function loadStoryIds(): string[] {
  const metaPath = resolve(__dirname, "../../storybook-static/meta.json");
  const meta = JSON.parse(readFileSync(metaPath, "utf-8")) as LadleMeta;
  return Object.keys(meta.stories);
}

const STORY_IDS = loadStoryIds();
const SEVERITIES = ["serious", "critical"] as const;
type BlockingImpact = (typeof SEVERITIES)[number];

const THEMES = ["light", "dark"] as const;

/**
 * Rule IDs we deliberately ignore globally. Add to this list ONLY with
 * an inline justification — every entry is a TODO to revisit.
 *
 * - `aria-required-children` / `aria-required-parent`: cmdk's
 *   `<Command*>` primitives nest `role="option"` under `role="listbox"`
 *   via portals that axe can't see; manually verified with VoiceOver.
 */
const EXEMPTED_RULE_IDS = new Set<string>(["aria-required-children", "aria-required-parent"]);

for (const story of STORY_IDS) {
  for (const theme of THEMES) {
    test(`${story} (${theme}) has no serious or critical a11y violations`, async ({ page }) => {
      await page.goto(`/?story=${encodeURIComponent(story)}&mode=preview&theme=${theme}`);
      await page.waitForSelector("#ladle-root :first-child", { timeout: 10_000 });
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      const blocking = results.violations.filter(
        (v) =>
          SEVERITIES.includes((v.impact ?? "minor") as BlockingImpact) &&
          !EXEMPTED_RULE_IDS.has(v.id),
      );
      expect(blocking, JSON.stringify(blocking, null, 2)).toEqual([]);
    });
  }
}
