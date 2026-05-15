import { cn } from "@/lib/cn";

export type RiskBand = "low" | "intermediate" | "high";

export type RiskScoreGaugeProps = {
  /** 0..1 calibrated probability of an event in the model horizon. */
  probability: number;
  /** Risk band, matched to NVDPA cut-points. */
  band: RiskBand;
  /** Optional human-readable horizon label, e.g. "5-year". */
  horizon?: string;
  className?: string;
};

const BAND_TOKEN: Record<RiskBand, { fg: string; bg: string; ring: string }> = {
  low: {
    fg: "text-[var(--color-risk-low)]",
    bg: "bg-[color-mix(in_oklch,_var(--color-risk-low)_15%,_transparent)]",
    ring: "stroke-[var(--color-risk-low)]",
  },
  intermediate: {
    fg: "text-[var(--color-risk-intermediate)]",
    bg: "bg-[color-mix(in_oklch,_var(--color-risk-intermediate)_15%,_transparent)]",
    ring: "stroke-[var(--color-risk-intermediate)]",
  },
  high: {
    fg: "text-[var(--color-risk-high)]",
    bg: "bg-[color-mix(in_oklch,_var(--color-risk-high)_15%,_transparent)]",
    ring: "stroke-[var(--color-risk-high)]",
  },
};

const BAND_LABEL: Record<RiskBand, string> = {
  low: "Low",
  intermediate: "Intermediate",
  high: "High",
};

/**
 * Circular dial showing the calibrated risk percentage with a band-coloured
 * arc. Used as the headline number on the patient dashboard. Reads as
 * `<RiskScoreGauge probability={0.13} band="intermediate" horizon="5-year" />`.
 */
export function RiskScoreGauge({ probability, band, horizon, className }: RiskScoreGaugeProps) {
  const clamped = Math.max(0, Math.min(1, probability));
  const pct = Math.round(clamped * 100);
  const radius = 58;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped);
  const tokens = BAND_TOKEN[band];
  const a11yLabel = horizon
    ? `${BAND_LABEL[band]} risk: ${pct} percent ${horizon}`
    : `${BAND_LABEL[band]} risk: ${pct} percent`;
  return (
    <div
      className={cn("flex flex-col items-center gap-2", className)}
      role="img"
      aria-label={a11yLabel}
    >
      <div
        className={cn("relative flex size-36 items-center justify-center rounded-full", tokens.bg)}
      >
        <svg
          className="-rotate-90 absolute inset-0"
          viewBox="0 0 144 144"
          aria-hidden="true"
          focusable="false"
        >
          <title>{a11yLabel}</title>
          <circle
            cx={72}
            cy={72}
            r={radius}
            strokeWidth={10}
            className="stroke-[var(--color-border)]"
            fill="transparent"
          />
          <circle
            cx={72}
            cy={72}
            r={radius}
            strokeWidth={10}
            strokeLinecap="round"
            className={cn("transition-[stroke-dashoffset] duration-700", tokens.ring)}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="flex flex-col items-center">
          <span className={cn("font-display font-semibold text-3xl", tokens.fg)}>{pct}%</span>
          <span className="text-[var(--color-fg-muted)] text-xs">{BAND_LABEL[band]}</span>
        </div>
      </div>
      {horizon ? (
        <span className="text-[var(--color-fg-subtle)] text-xs">{horizon} risk</span>
      ) : null}
    </div>
  );
}
