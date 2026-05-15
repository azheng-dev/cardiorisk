import type { Story } from "@ladle/react";

import { RiskScoreGauge } from "./risk-score-gauge";

export default { title: "Domain / RiskScoreGauge" };

export const Bands: Story = () => (
  <div className="flex flex-wrap gap-8">
    <RiskScoreGauge probability={0.04} band="low" horizon="5-year" />
    <RiskScoreGauge probability={0.13} band="intermediate" horizon="5-year" />
    <RiskScoreGauge probability={0.31} band="high" horizon="5-year" />
  </div>
);
