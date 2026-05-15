import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { HitlActionBar } from "./hitl-action-bar";

describe("HitlActionBar", () => {
  it("emits an approve decision without a note", async () => {
    const onDecide = vi.fn();
    const user = userEvent.setup();
    render(<HitlActionBar step="risk" onDecide={onDecide} />);
    await user.click(screen.getByRole("button", { name: /approve/i }));
    expect(onDecide).toHaveBeenCalledWith({ kind: "approve" });
  });

  it("requires a note before submitting an edit", async () => {
    const onDecide = vi.fn();
    const user = userEvent.setup();
    render(<HitlActionBar step="risk" onDecide={onDecide} />);
    await user.click(screen.getByRole("button", { name: /^edit$/i }));
    const submit = screen.getByRole("button", { name: /save edits/i });
    expect(submit).toBeDisabled();
    await user.type(screen.getByRole("textbox"), "Switch ST_Slope to Flat");
    expect(submit).toBeEnabled();
    await user.click(submit);
    expect(onDecide).toHaveBeenCalledWith({
      kind: "edit",
      note: "Switch ST_Slope to Flat",
    });
  });

  it("requires a note before submitting a reject", async () => {
    const onDecide = vi.fn();
    const user = userEvent.setup();
    render(<HitlActionBar step="letter" onDecide={onDecide} />);
    await user.click(screen.getByRole("button", { name: /reject/i }));
    expect(screen.getByRole("button", { name: /^reject$/i })).toBeDisabled();
    await user.type(screen.getByRole("textbox"), "Patient prefers SDM first.");
    await user.click(screen.getByRole("button", { name: /^reject$/i }));
    expect(onDecide).toHaveBeenCalledWith({
      kind: "reject",
      note: "Patient prefers SDM first.",
    });
  });
});
