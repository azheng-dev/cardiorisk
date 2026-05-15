/**
 * Static reference data for the brand preview page.
 *
 * The colour values in PALETTE_GROUPS reference CSS custom properties
 * declared in `app/globals.css` (single source of truth). This file
 * exists purely to label and document those tokens for the preview UI.
 */

export type ColourToken = {
  /** CSS custom property name, e.g. "--color-accent". */
  token: string;
  /** Human-readable label rendered under the swatch. */
  label: string;
  /**
   * Optional foreground token used to render an "on" pip on the swatch
   * — illustrates the legible-pair contract.
   */
  onToken?: string;
};

export type PaletteGroup = {
  title: string;
  description: string;
  tokens: ColourToken[];
};

export const PALETTE_GROUPS: PaletteGroup[] = [
  {
    title: "Surface",
    description: "Layered backgrounds: bg < surface < surface-muted < surface-raised.",
    tokens: [
      { token: "--color-bg", label: "bg", onToken: "--color-fg" },
      { token: "--color-surface", label: "surface", onToken: "--color-fg" },
      { token: "--color-surface-muted", label: "surface-muted", onToken: "--color-fg" },
      { token: "--color-surface-raised", label: "surface-raised", onToken: "--color-fg" },
    ],
  },
  {
    title: "Foreground",
    description: "Text colours, highest to lowest emphasis, plus the on-accent pair.",
    tokens: [
      { token: "--color-fg", label: "fg" },
      { token: "--color-fg-muted", label: "fg-muted" },
      { token: "--color-fg-subtle", label: "fg-subtle" },
      {
        token: "--color-fg-on-accent",
        label: "fg-on-accent",
        onToken: "--color-accent",
      },
    ],
  },
  {
    title: "Brand accent",
    description: "Clinical teal — used sparingly to anchor the hierarchy.",
    tokens: [
      { token: "--color-accent", label: "accent", onToken: "--color-accent-fg" },
      { token: "--color-accent-hover", label: "accent-hover", onToken: "--color-accent-fg" },
      { token: "--color-accent-soft", label: "accent-soft", onToken: "--color-accent" },
    ],
  },
  {
    title: "Risk bands",
    description:
      "Maps to the v1 risk model's three bands. WCAG-AA on both surface and bg in both themes.",
    tokens: [
      { token: "--color-risk-low", label: "risk-low" },
      { token: "--color-risk-low-soft", label: "risk-low-soft", onToken: "--color-risk-low" },
      { token: "--color-risk-intermediate", label: "risk-intermediate" },
      {
        token: "--color-risk-intermediate-soft",
        label: "risk-intermediate-soft",
        onToken: "--color-risk-intermediate",
      },
      { token: "--color-risk-high", label: "risk-high" },
      { token: "--color-risk-high-soft", label: "risk-high-soft", onToken: "--color-risk-high" },
    ],
  },
  {
    title: "Status",
    description: "Used for toasts, badges, and inline banners.",
    tokens: [
      { token: "--color-info", label: "info" },
      { token: "--color-success", label: "success" },
      { token: "--color-warning", label: "warning" },
      { token: "--color-danger", label: "danger" },
    ],
  },
  {
    title: "Borders + focus",
    description: "Border tones + the accessible focus ring shared by every interactive element.",
    tokens: [
      { token: "--color-border", label: "border" },
      { token: "--color-border-strong", label: "border-strong" },
      { token: "--color-focus", label: "focus", onToken: "--color-bg" },
    ],
  },
];

/**
 * Manual neutral scale so the user can verify the L-step rhythm under
 * both themes. Not consumed by any component — semantic tokens only.
 */
export const NEUTRAL_SCALE: { label: string; colour: string }[] = [
  { label: "neutral-50", colour: "oklch(99% 0.005 220)" },
  { label: "neutral-100", colour: "oklch(97% 0.006 220)" },
  { label: "neutral-200", colour: "oklch(91% 0.008 230)" },
  { label: "neutral-300", colour: "oklch(82% 0.012 230)" },
  { label: "neutral-400", colour: "oklch(68% 0.012 230)" },
  { label: "neutral-500", colour: "oklch(58% 0.012 230)" },
  { label: "neutral-600", colour: "oklch(43% 0.014 230)" },
  { label: "neutral-700", colour: "oklch(28% 0.014 230)" },
  { label: "neutral-800", colour: "oklch(20% 0.020 230)" },
  { label: "neutral-900", colour: "oklch(14% 0.015 230)" },
];

export type TypeRow = {
  token: string;
  className: string;
  fontFamily: string;
  sample: string;
};

export const TYPE_SCALE: TypeRow[] = [
  {
    token: "text-6xl / display",
    className: "font-display font-semibold text-6xl tracking-tighter",
    fontFamily: "var(--font-display)",
    sample: "Cardiovascular risk",
  },
  {
    token: "text-4xl / h1",
    className: "font-display font-semibold text-4xl tracking-tighter",
    fontFamily: "var(--font-display)",
    sample: "An open-source clinical co-pilot",
  },
  {
    token: "text-2xl / h2",
    className: "font-display font-semibold text-2xl tracking-tight",
    fontFamily: "var(--font-display)",
    sample: "Recommendation summary",
  },
  {
    token: "text-lg / h3",
    className: "font-display font-semibold text-lg tracking-tight",
    fontFamily: "var(--font-display)",
    sample: "Calibrated 5-year ACVR",
  },
  {
    token: "text-base / body",
    className: "font-sans text-base",
    fontFamily: "var(--font-sans)",
    sample:
      "Patient is a 62-year-old female with treated hypertension and ST-segment depression at peak exercise.",
  },
  {
    token: "text-sm / body-sm",
    className: "font-sans text-sm text-[var(--color-fg-muted)]",
    fontFamily: "var(--font-sans)",
    sample: "All claims must cite a guideline span. Suppressed claims are dropped before render.",
  },
  {
    token: "text-xs / caption",
    className: "font-sans text-xs text-[var(--color-fg-subtle)]",
    fontFamily: "var(--font-sans)",
    sample: "NVDPA 2023 §3.4 — page 28, lines 14–22",
  },
  {
    token: "mono / data",
    className: "font-mono text-sm",
    fontFamily: "var(--font-mono)",
    sample: "AUROC=0.872 [0.853, 0.890]   Brier=0.122",
  },
];

export const RADIUS_SCALE = [
  { token: "--radius-xs" },
  { token: "--radius-sm" },
  { token: "--radius-md" },
  { token: "--radius-lg" },
  { token: "--radius-xl" },
  { token: "--radius-2xl" },
];

export const SHADOW_SCALE = [
  { token: "--shadow-soft" },
  { token: "--shadow-card" },
  { token: "--shadow-pop" },
];
