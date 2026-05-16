/**
 * Sentry server-side initialisation (Phase 7, ADR-024).
 *
 * Loaded by `instrumentation.ts` on the Node runtime. No-op when
 * SENTRY_DSN is unset.
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
    beforeSend(event) {
      return scrubPatient(event);
    },
  });
}

function scrubPatient<T>(value: T): T {
  if (value == null || typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return value.map((item) => scrubPatient(item)) as unknown as T;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (k.toLowerCase() === "patient") {
      out[k] = "<scrubbed>";
    } else {
      out[k] = scrubPatient(v);
    }
  }
  return out as unknown as T;
}
