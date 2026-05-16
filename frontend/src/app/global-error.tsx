"use client";

import * as Sentry from "@sentry/nextjs";
import NextError from "next/error";
import { useEffect } from "react";

/**
 * App-Router global error boundary. Required by `@sentry/nextjs` v8 to
 * capture React render errors that escape the route-level error
 * boundary. The body must be a full `<html>` document because Next.js
 * unmounts the root layout when this boundary fires (per
 * https://nextjs.org/docs/app/api-reference/file-conventions/error#global-errorjs).
 *
 * Sentry's `captureException` is a no-op when `SENTRY_DSN` /
 * `NEXT_PUBLIC_SENTRY_DSN` is unset (Phase 7 env-var-gating
 * contract; see ADR-024 §4), so CI / forks without keys still
 * render correctly.
 */
export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <body>
        <NextError statusCode={0} />
      </body>
    </html>
  );
}
