/**
 * Sentry edge-runtime initialisation (Phase 7, ADR-024).
 *
 * Loaded by `instrumentation.ts` on the Edge runtime. Same no-op-
 * without-DSN contract as the server config.
 */

import * as Sentry from "@sentry/nextjs";

const dsn = process.env.SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    release: process.env.VERCEL_GIT_COMMIT_SHA ?? process.env.APP_RELEASE,
    environment: process.env.APP_ENV ?? "production",
    tracesSampleRate: Number(process.env.SENTRY_TRACES_SAMPLE_RATE ?? 0.1),
    sendDefaultPii: false,
  });
}
