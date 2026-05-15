import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Input } from "./input";

describe("Input", () => {
  it("renders with the supplied placeholder", () => {
    render(<Input placeholder="Patient initials" />);
    expect(screen.getByPlaceholderText("Patient initials")).toBeInTheDocument();
  });

  it("forwards aria-invalid for downstream styling", () => {
    render(<Input aria-invalid />);
    expect(screen.getByRole("textbox")).toHaveAttribute("aria-invalid", "true");
  });

  it("respects the type prop", () => {
    render(<Input type="number" defaultValue={5} />);
    expect(screen.getByRole("spinbutton")).toHaveValue(5);
  });
});
