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

export default nextConfig;
