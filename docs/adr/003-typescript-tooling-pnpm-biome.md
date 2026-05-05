# ADR-003: TypeScript tooling — pnpm, Biome, strict tsc, Vitest

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Decision

For the TypeScript frontend:

- **Package manager:** [`pnpm`](https://pnpm.io/) (10+).
- **Lint + format:** [`biome`](https://biomejs.dev/) (single tool, replaces ESLint + Prettier).
- **Type check:** `tsc --noEmit` with strict mode + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`.
- **Test runner:** [`vitest`](https://vitest.dev/).
- **Node version:** 22 LTS, pinned in `frontend/package.json` engines.

## Context

In 2024–2026 the JS/TS tooling story narrowed:

- `pnpm` won on disk-efficient install, content-addressable store, and predictable monorepo behaviour.
- `biome` matured to where it can replace the `eslint + @typescript-eslint + prettier + import-sort` stack with one binary, in Rust, ~20× faster.
- `vitest` displaced `jest` as the de-facto test runner for ESM-native projects.

## Consequences

- **Positive:** `pnpm install` in seconds; `biome check .` near-instant. CI loop stays fast.
- **Positive:** one tool to configure (`biome.json`), one binary to upgrade.
- **Positive:** `noUncheckedIndexedAccess` catches a real class of bugs (`array[i]` is `T | undefined`).
- **Negative:** Biome doesn't yet have full parity with every ESLint plugin. If we later need rules from, e.g., `eslint-plugin-jsx-a11y`, we'd need to either accept the gap or run ESLint in parallel for that subset. Decision can be revisited in Phase 5 when accessibility lints become important.
- **Negative:** `exactOptionalPropertyTypes` requires more deliberate `undefined` handling — costly migration if turned on later, cheap to keep on from day one.

## Alternatives considered

- **`npm`.** Rejected: slower, larger node_modules, weaker monorepo story.
- **`yarn berry`.** Rejected: PnP mode is still rough with several deps; classic mode offers no advantage over pnpm.
- **`bun`.** Considered. Rejected (for now): the runtime story is tempting but we're going to be sharing tooling with the wider ecosystem (Vercel, Tailwind, shadcn, Next.js). Bun-as-runtime in production is a Phase 8 question if we even need it.
- **`eslint` + `prettier`.** Rejected: legacy choice. Biome solves the same problems with less config.
- **`jest`.** Rejected: ESM story still painful.

## Trigger to revisit

- Biome blocks an accessibility lint we need in Phase 5 (would supplement with ESLint, not replace).
- Bun reaches a stable v2 with first-class Next.js support and we want the runtime perf.
