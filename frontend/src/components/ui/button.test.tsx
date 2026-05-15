import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "./button";

describe("Button", () => {
  it("renders with default primary variant", () => {
    render(<Button>Hello</Button>);
    const btn = screen.getByRole("button", { name: "Hello" });
    expect(btn).toBeInTheDocument();
    expect(btn.className).toContain("var(--color-accent)");
  });

  it("defaults to type=button to avoid accidental form submits", () => {
    render(<Button>Submit</Button>);
    expect(screen.getByRole("button", { name: "Submit" })).toHaveAttribute("type", "button");
  });

  it("forwards onClick", async () => {
    const handle = vi.fn();
    render(<Button onClick={handle}>Click</Button>);
    screen.getByRole("button", { name: "Click" }).click();
    expect(handle).toHaveBeenCalledOnce();
  });

  it("renders all six variants without crashing", () => {
    const variants = ["primary", "secondary", "outline", "ghost", "danger", "link"] as const;
    for (const v of variants) {
      const { unmount } = render(<Button variant={v}>{v}</Button>);
      expect(screen.getByRole("button", { name: v })).toBeInTheDocument();
      unmount();
    }
  });

  it("applies disabled attribute and removes pointer events", () => {
    render(<Button disabled>Off</Button>);
    const btn = screen.getByRole("button", { name: "Off" });
    expect(btn).toBeDisabled();
    expect(btn.className).toContain("disabled:pointer-events-none");
  });

  it("renders the lg size with the larger height class", () => {
    render(
      <Button size="lg" data-testid="lg-btn">
        Large
      </Button>,
    );
    const btn = screen.getByTestId("lg-btn");
    expect(btn.className).toContain("h-11");
  });
});
