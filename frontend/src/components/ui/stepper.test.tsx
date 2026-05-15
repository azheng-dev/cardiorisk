import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Stepper } from "./stepper";

const STEPS = [
  { id: "triage", label: "Triage" },
  { id: "risk", label: "Risk" },
  { id: "guideline", label: "Guideline" },
];

describe("Stepper", () => {
  it("marks the current step with aria-current=step", () => {
    render(<Stepper steps={STEPS} current={1} />);
    const items = screen.getAllByRole("listitem");
    expect(items[1]).toHaveAttribute("aria-current", "step");
    expect(items[0]).not.toHaveAttribute("aria-current");
  });

  it("renders one item per step", () => {
    render(<Stepper steps={STEPS} current={0} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(STEPS.length);
  });
});
