import type { Story } from "@ladle/react";

import { CitationChip } from "./citation-chip";

const SUPPORTED_SPAN =
  "Patients with intermediate (10–15%) absolute CVD risk should be offered statin therapy where lifestyle measures alone are insufficient.";

export default { title: "Domain / CitationChip" };

export const Inline: Story = () => (
  <p className="max-w-prose text-sm leading-relaxed">
    Statin therapy is recommended in this case{" "}
    <CitationChip
      label="[1]"
      verdict="supported"
      span={SUPPORTED_SPAN}
      source={{ docId: "RACGP-Red-Book", pages: "112" }}
      entailment={0.93}
    />{" "}
    based on RACGP §3.4. The reviewer can verify the underlying span without leaving the screen{" "}
    <CitationChip
      label="[2]"
      verdict="unsupported"
      span="ARBs are first-line for hypertension in CKD stage 3."
      source={{ docId: "NVDPA-2023", pages: "44" }}
      entailment={0.18}
    />
    .
  </p>
);
