import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";

function Harness() {
  return (
    <Tabs defaultValue="risk">
      <TabsList>
        <TabsTrigger value="risk">Risk</TabsTrigger>
        <TabsTrigger value="explain">Explain</TabsTrigger>
      </TabsList>
      <TabsContent value="risk">Risk panel</TabsContent>
      <TabsContent value="explain">Explain panel</TabsContent>
    </Tabs>
  );
}

describe("Tabs", () => {
  it("renders the default tab and switches with arrow keys", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    expect(screen.getByText("Risk panel")).toBeInTheDocument();
    expect(screen.queryByText("Explain panel")).not.toBeInTheDocument();
    const risk = screen.getByRole("tab", { name: /risk/i });
    risk.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByText("Explain panel")).toBeInTheDocument();
  });
});
