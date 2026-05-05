# Frontend

Pre-alpha. Next.js 15 + Tailwind v4 + shadcn/ui land in Phase 5 (UI rebrand + redesign). Phase 0 is just enough scaffolding for `lint`, `type-check`, and `test` to run in CI.

## Layout

```
frontend/
  src/                  # source (only a smoke test in Phase 0)
  package.json
  biome.json            # lint + format config
  tsconfig.json         # strict TS config (noUncheckedIndexedAccess, exactOptionalPropertyTypes)
  vitest.config.ts
```

## Local commands

```bash
# from repo root
cd frontend

pnpm install              # install deps
pnpm test --run           # run unit tests once
pnpm test                 # run unit tests in watch mode
pnpm lint                 # biome check
pnpm lint:fix             # biome check --write
pnpm type-check           # tsc --noEmit
```
