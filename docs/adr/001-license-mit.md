# ADR-001: License under MIT

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Decision

License this repository under the MIT License.

## Context

The repo is a public engineering portfolio piece. It is **not** a product, **not** a clinical tool, and is unlikely ever to be commercialised. The audience is recruiters, hiring managers, and engineers in the regulated-domain AI space who may want to read or borrow ideas. Friction-free reuse is more valuable than enforced share-alike.

## Consequences

- **Positive:** anyone can read, fork, copy, and reuse with attribution. Maximal signal to a reader that "this is meant to be useful, not defended."
- **Positive:** widely understood, well-tested in court, no ideological baggage.
- **Negative:** does not require derivative works to remain open. We accept this; the artefact value is the visible craftsmanship, not the bytes.
- **Note:** does not waive the disclaimer at the top of the README. Use of this software for any clinical purpose is explicitly disclaimed by both the README and the warranty section of the MIT license itself.

## Alternatives considered

- **Apache-2.0.** Rejected: marginal benefit (explicit patent grant) at the cost of slightly more friction. No patentable inventions are anticipated.
- **AGPL-3.0.** Rejected: would force any service running this code to publish their changes. Hostile to the friction-free reuse goal.
- **No license / All rights reserved.** Rejected: makes the repo useless to readers (they legally cannot copy from it), which defeats the portfolio purpose.
- **Polyform / Source-available.** Rejected: same reason as no-license.

## Trigger to revisit

- Commercialisation intent (would consider dual-license).
- Inclusion of third-party code with incompatible license obligations.
