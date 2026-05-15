import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Badge } from "./badge";

describe("Badge", () => {
  it("renders text content", () => {
    render(<Badge>Active</Badge>);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("renders every risk-band variant with the right token reference", () => {
    const cases = [
      { variant: "risk-low", token: "var(--color-risk-low-soft)" },
      { variant: "risk-intermediate", token: "var(--color-risk-intermediate-soft)" },
      { variant: "risk-high", token: "var(--color-risk-high-soft)" },
    ] as const;
    for (const { variant, token } of cases) {
      const { unmount } = render(<Badge variant={variant}>{variant}</Badge>);
      const el = screen.getByText(variant);
      expect(el.className).toContain(token);
      unmount();
    }
  });

  it("falls back to neutral when no variant given", () => {
    render(<Badge>Default</Badge>);
    expect(screen.getByText("Default").className).toContain("var(--color-surface-muted)");
  });
});
