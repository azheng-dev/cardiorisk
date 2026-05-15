import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Logo } from "./logo";

describe("Logo", () => {
  it("renders mark + wordmark by default (lockup)", () => {
    render(<Logo />);
    expect(screen.getByLabelText("CardioRisk Co-Pilot logo mark")).toBeInTheDocument();
    // Wordmark splits "CardioRisk" + " Co-Pilot" across two spans, so
    // assert via the closest containing span rather than text matching
    // (which would collide with the SVG <title>).
    expect(screen.getByText(/Co-Pilot/, { selector: "span > span" })).toBeInTheDocument();
  });

  it("renders only the mark when variant=mark", () => {
    render(<Logo variant="mark" />);
    expect(screen.getByLabelText("CardioRisk Co-Pilot logo mark")).toBeInTheDocument();
    expect(screen.queryByText(/Co-Pilot/, { selector: "span > span" })).not.toBeInTheDocument();
  });

  it("renders only the wordmark when variant=wordmark", () => {
    render(<Logo variant="wordmark" />);
    expect(screen.queryByLabelText("CardioRisk Co-Pilot logo mark")).not.toBeInTheDocument();
    expect(screen.getByText(/Co-Pilot/, { selector: "span > span" })).toBeInTheDocument();
  });

  it.each(["sm", "md", "lg"] as const)("supports the %s size", (size) => {
    const { container } = render(<Logo size={size} />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
  });
});
