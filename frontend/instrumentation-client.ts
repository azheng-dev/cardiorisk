/**
 * Sentry client-side initialisation (Phase 7, ADR-024).
 *
 * Loaded by Next.js 15 automatically — the file name is conventional.
 * Init is a no-op when NEXT_PUBLIC_SENTRY_DSN is unset, which is the
 * default in CI and any fork that hasn't opted in. Production wires
 * the DSN via the Vercel project env.
 */

import * as Sentry from "@sentry/nextjs";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    release: process.env.NEXT_PUBLIC_APP_RELEASE,
    environment: process.env.NEXT_PUBLIC_APP_ENV ?? "production",
    // Production sample rate — every event ships an inferred trace.
    // Preview / non-prod can lower this via the env if needed.
    tracesSampleRate: Number(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE ?? 0.1),
    // PII scrub: synthetic-data-only policy (AGENTS §6) means the
    // patient payload must never reach Sentry. Defense in depth on
    // top of the backend's `before_send` hook.
    beforeSend(event) {
      return scrubPatient(event);
    },
    // Defaults that match Sentry's recommended Next.js 15 setup.
    sendDefaultPii: false,
  });
}

/**
 * Recursively strip every value under a `patient` key. Same contract
 * as the backend scrubber in `cardiorisk.observability.sentry`.
 */
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
