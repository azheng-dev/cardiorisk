import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { Checkbox } from "./checkbox";

describe("Checkbox", () => {
  it("toggles between unchecked and checked on click", async () => {
    const user = userEvent.setup();
    render(<Checkbox aria-label="Patient consent" />);
    const cb = screen.getByRole("checkbox", { name: /patient consent/i });
    expect(cb).toHaveAttribute("data-state", "unchecked");
    await user.click(cb);
    expect(cb).toHaveAttribute("data-state", "checked");
  });

  it("supports keyboard activation via Space", async () => {
    const user = userEvent.setup();
    render(<Checkbox aria-label="Tick me" />);
    const cb = screen.getByRole("checkbox");
    cb.focus();
    expect(cb).toHaveFocus();
    await user.keyboard(" ");
    expect(cb).toHaveAttribute("data-state", "checked");
  });
});
