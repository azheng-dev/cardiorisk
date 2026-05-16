/**
 * Next.js 15 instrumentation hook.
 *
 * Dispatches Sentry SDK init to the correct file per runtime
 * (Node vs Edge). Required by the `@sentry/nextjs` v8 setup.
 */

import * as Sentry from "@sentry/nextjs";

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    await import("./sentry.server.config");
  }
  if (process.env.NEXT_RUNTIME === "edge") {
    await import("./sentry.edge.config");
  }
}

export const onRequestError = Sentry.captureRequestError;
