import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RiskScoreGauge } from "./risk-score-gauge";

describe("RiskScoreGauge", () => {
  it("rounds the probability to a percentage", () => {
    render(<RiskScoreGauge probability={0.137} band="intermediate" />);
    expect(screen.getByText("14%")).toBeInTheDocument();
    expect(screen.getByText("Intermediate")).toBeInTheDocument();
  });

  it("clamps probabilities into [0,1]", () => {
    render(<RiskScoreGauge probability={1.5} band="high" />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("includes the horizon in the accessible label", () => {
    render(<RiskScoreGauge probability={0.04} band="low" horizon="5-year" />);
    expect(screen.getByRole("img", { name: /low risk: 4 percent 5-year/i })).toBeInTheDocument();
  });
});
