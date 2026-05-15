import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { Switch } from "./switch";

describe("Switch", () => {
  it("flips on click", async () => {
    const user = userEvent.setup();
    render(<Switch aria-label="TabICL" />);
    const sw = screen.getByRole("switch", { name: /tabicl/i });
    expect(sw).toHaveAttribute("aria-checked", "false");
    await user.click(sw);
    expect(sw).toHaveAttribute("aria-checked", "true");
  });
});
