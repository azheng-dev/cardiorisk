import { zodResolver } from "@hookform/resolvers/zod";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { Button } from "./button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "./form";
import { Input } from "./input";

const Schema = z.object({
  age: z.number({ message: "Age is required" }).int().min(20, "Patient must be 20 or older"),
});

function Harness() {
  const form = useForm<z.infer<typeof Schema>>({
    resolver: zodResolver(Schema),
    defaultValues: { age: undefined as unknown as number },
  });
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(() => undefined)} aria-label="Risk-form" noValidate>
        <FormField
          control={form.control}
          name="age"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Age</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  inputMode="numeric"
                  {...field}
                  onChange={(e) => field.onChange(e.target.valueAsNumber)}
                  value={Number.isFinite(field.value) ? field.value : ""}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <Button type="submit">Submit</Button>
      </form>
    </Form>
  );
}

describe("Form", () => {
  it("surfaces zod validation messages on submit", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(await screen.findByText("Age is required")).toBeInTheDocument();
  });

  it("clears the error when a valid value is supplied", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(await screen.findByText("Age is required")).toBeInTheDocument();
    await user.type(screen.getByLabelText("Age"), "55");
    await user.click(screen.getByRole("button", { name: /submit/i }));
    expect(screen.queryByText("Age is required")).not.toBeInTheDocument();
  });
});
