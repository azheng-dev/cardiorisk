import type { Story } from "@ladle/react";

import { Button } from "@/components/ui/button";

import { EmptyState, ErrorState, LoadingState } from "./states";

export default { title: "Domain / States" };

export const Empty: Story = () => (
  <EmptyState
    title="No cases yet"
    description="Create a synthetic patient to start the agent flow."
    action={<Button>Create case</Button>}
  />
);

export const ErrorStory: Story = () => (
  <ErrorState
    title="Citation verifier unreachable"
    description="The NLI service didn’t respond after three retries. The agent stopped to keep claims auditable."
    action={
      <Button variant="outline" size="sm">
        Retry
      </Button>
    }
  />
);

export const Loading: Story = () => <LoadingState label="Retrieving guideline chunks…" rows={4} />;
