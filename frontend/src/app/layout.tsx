import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { ThemeProvider } from "@/components/theme-provider";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "CardioRisk Co-Pilot",
    template: "%s — CardioRisk Co-Pilot",
  },
  description:
    "An open-source agentic clinical co-pilot for cardiovascular disease risk assessment. " +
    "Research artefact only — synthetic data, not for clinical use.",
  applicationName: "CardioRisk Co-Pilot",
  authors: [{ name: "Andrew Zheng" }],
  keywords: ["cardiovascular risk", "clinical AI", "RAG", "explainable AI", "research artefact"],
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0c1418" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body>
        <ThemeProvider>{children}</ThemeProvider>
        {/* Vercel Web Analytics + Speed Insights — both auto-no-op
            outside Vercel (no DSN/token required). Phase 7, ADR-024. */}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
