import { zodResolver } from "@hookform/resolvers/zod";
import type { Story } from "@ladle/react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "./button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./form";
import { Input } from "./input";

const Schema = z.object({
  age: z
    .number({ message: "Age is required" })
    .int("Whole years only")
    .min(20, "Patient must be 20 or older")
    .max(120, "Plausibility ceiling: 120"),
  systolic: z
    .number({ message: "Systolic BP is required" })
    .int()
    .min(60, "Below 60 mmHg is implausible")
    .max(260, "Above 260 mmHg is implausible"),
});
type Schema = z.infer<typeof Schema>;

export default { title: "Primitives / Form" };

export const RHFWithZod: Story = () => {
  const form = useForm<Schema>({
    resolver: zodResolver(Schema),
    defaultValues: {
      age: undefined as unknown as number,
      systolic: undefined as unknown as number,
    },
  });
  return (
    <Form {...form}>
      <form
        className="flex max-w-sm flex-col gap-4"
        onSubmit={form.handleSubmit((values) => {
          alert(`Submitted: ${JSON.stringify(values)}`);
        })}
      >
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
              <FormDescription>Patient age in whole years.</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="systolic"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Systolic BP (mmHg)</FormLabel>
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
        <Button type="submit">Score risk</Button>
      </form>
    </Form>
  );
};
