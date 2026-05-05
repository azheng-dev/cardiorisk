# Research notes

Phase 1 deliverables for CardioRisk Co-Pilot — a critical examination of a prior Honours study on CVD prediction with deep learning, and the proposed v1 design that supersedes it.

| # | Document | Status | Purpose |
|---|---|---|---|
| 01 | [Honours-work recap](./01-honours-recap.md) | Done | Sanitised, descriptive summary of the prior study (architectures, datasets, FS methods, headline numbers). |
| 02 | [Current state of the art (2025–2026)](./02-current-soa.md) | Done | Distilled summary of the current literature on tabular CVD risk prediction, calibrated against the user-supplied Deep Research report. |
| 03 | [Critical review](./03-critical-review.md) | Done | Opinionated, head-to-head verdict for each Honours design choice: defensible / outdated / what to upgrade / evidence. |
| 04 | [Revised v1 design](./04-revised-design.md) | Done | The proposed v1 ML system, justified line-by-line against §03. |
| 05 | [EDA findings (Phase 2.1)](./05-eda-findings.md) | Done | Concrete numbers from the four UCI subsets in HFP schema; what's there, what's broken, implications for Phase 2.2. |
| 06 | [Preprocessing decisions (Phase 2.2)](./06-preprocessing-decisions.md) | Done | Opinionated walkthrough of the cleaning + per-model pipeline + LODO splitter; cross-refs design and EDA. |
| ADR-006 | [Risk-model architecture](../adr/006-risk-model-architecture.md) | Proposed (Accepted on Phase 1 checkpoint) | Binding decision: chosen architecture, rejected alternatives, trigger to revisit. |
| ADR-008 | [Preprocessing pipeline](../adr/008-preprocessing-pipeline.md) | Accepted | Binding decision: cleaning prefix + per-model sklearn factories + LODO + indicators + RCS. |

## What to read first

If you're a recruiter or contributor reading the repo cold and you want the *opinionated* answer in one sitting: read [`03-critical-review.md`](./03-critical-review.md), then [`04-revised-design.md`](./04-revised-design.md), then [ADR-006](../adr/006-risk-model-architecture.md). Read the recap and SoA only if you want to verify the underlying claims.

If you're an ML researcher wanting to reproduce or extend this work: read in numerical order (01 → 02 → 03 → 04 → ADR-006).

## Honesty contract

These docs are written under [AGENTS.md §3](../../AGENTS.md) "honesty over impressiveness." Where the prior study is strong, the critical review will say so verbatim. Where the prior study is hard to defend in 2026, the critical review will say so plainly and cite the evidence. The verdict is not a marketing document.
