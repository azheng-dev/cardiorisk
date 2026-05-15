import { cn } from "@/lib/cn";

export type LogoProps = {
  /**
   * `mark` -> SVG monogram only.
   * `wordmark` -> wordmark text only.
   * `lockup` -> mark + wordmark (default).
   */
  variant?: "mark" | "wordmark" | "lockup";
  size?: "sm" | "md" | "lg";
  className?: string;
};

const sizeMap: Record<NonNullable<LogoProps["size"]>, { mark: number; text: string }> = {
  sm: { mark: 20, text: "text-base" },
  md: { mark: 28, text: "text-xl" },
  lg: { mark: 40, text: "text-3xl" },
};

/**
 * The CardioRisk Co-Pilot brand mark.
 *
 * Geometric monogram: an inscribed `C` ring, broken at 11-o'clock to
 * suggest both a heartbeat / QRS waveform and an "open" co-pilot loop.
 * Sized in `currentColor` so the mark recolours wherever it lands
 * (header, footer, button, favicon).
 *
 * Why custom: SVG (not a font icon) so the lockup is crisp at favicon
 * and OG-card sizes; `currentColor` so it survives every theme switch
 * without a token-binding step.
 */
export function Logo({ variant = "lockup", size = "md", className }: LogoProps) {
  const { mark: markSize, text: textCls } = sizeMap[size];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 font-display font-semibold tracking-tight",
        textCls,
        className,
      )}
    >
      {variant !== "wordmark" && <LogoMark size={markSize} />}
      {variant !== "mark" && (
        <span className="select-none">
          CardioRisk
          <span className="text-fg-muted"> Co-Pilot</span>
        </span>
      )}
    </span>
  );
}

function LogoMark({ size }: { size: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="CardioRisk Co-Pilot logo mark"
      style={{ color: "var(--color-accent)" }}
    >
      <title>CardioRisk Co-Pilot</title>
      {/* Outer broken ring — stylised C + open loop. */}
      <path
        d="M27 16a11 11 0 1 1-7-10.25"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {/* QRS-style waveform across the chord. */}
      <path
        d="M9 16h4l2-5 3 10 2-5h3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
