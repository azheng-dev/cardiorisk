import { withSentryConfig } from "@sentry/nextjs";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The Next.js typed-routes plumbing is genuinely useful but it doesn't
  // play with our `verbatimModuleSyntax: true` tsconfig (which Phase 0
  // chose for explicit type imports). Re-enable in Phase 5.3 when the
  // route surface is large enough to justify a tsconfig split.
  typedRoutes: false,
  // Static-only export is a Phase 8 deploy decision; don't bake it in here.
  experimental: {
    // App Router is default in 15; nothing to flag.
  },
};

// Wrap with Sentry's Next config helper so the build process can:
// 1. Discover instrumentation files (instrumentation-client.ts,
//    sentry.server.config.ts, sentry.edge.config.ts).
// 2. Wire source-map upload IF a SENTRY_AUTH_TOKEN is set in CI/Vercel.
// 3. Inject the Sentry router transition handler for App Router.
//
// All Sentry-specific behaviour is env-var gated: without
// SENTRY_AUTH_TOKEN, the build still succeeds and emits maps — they
// just aren't uploaded. Without SENTRY_DSN at runtime, every init is
// a no-op.
export default withSentryConfig(nextConfig, {
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  silent: !process.env.CI,
  // Don't fail the build because we lack a token in non-prod
  // environments (every PR preview, every CI run without secrets).
  disableLogger: true,
  authToken: process.env.SENTRY_AUTH_TOKEN,
  telemetry: false,
});
