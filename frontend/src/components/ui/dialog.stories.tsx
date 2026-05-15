import type { Story } from "@ladle/react";

import { Button } from "./button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./dialog";

export default { title: "Primitives / Dialog" };

export const Confirm: Story = () => (
  <Dialog>
    <DialogTrigger asChild>
      <Button variant="danger">Reject draft</Button>
    </DialogTrigger>
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Reject letter draft?</DialogTitle>
        <DialogDescription>
          The draft will be archived and the letter agent will retry with the reviewer’s reasoning
          attached.
        </DialogDescription>
      </DialogHeader>
      <DialogFooter>
        <Button variant="ghost">Cancel</Button>
        <Button variant="danger">Reject</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
);
