import type { Story } from "@ladle/react";

import { Badge } from "./badge";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./table";

export default { title: "Primitives / Table" };

const ROWS = [
  { name: "TabICL ensemble", auroc: 0.87, brier: 0.122, band: "Top" },
  { name: "XGBoost (calibrated)", auroc: 0.84, brier: 0.131, band: "Reference" },
  { name: "Logistic regression (RCS+L1)", auroc: 0.82, brier: 0.135, band: "Baseline" },
];

export const ModelCard: Story = () => (
  <Table>
    <TableCaption>Aggregate metrics across the four LODO folds.</TableCaption>
    <TableHeader>
      <TableRow>
        <TableHead>Model</TableHead>
        <TableHead>AUROC</TableHead>
        <TableHead>Brier</TableHead>
        <TableHead>Band</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {ROWS.map((row) => (
        <TableRow key={row.name}>
          <TableCell className="font-medium">{row.name}</TableCell>
          <TableCell className="font-mono">{row.auroc.toFixed(3)}</TableCell>
          <TableCell className="font-mono">{row.brier.toFixed(3)}</TableCell>
          <TableCell>
            <Badge variant={row.band === "Top" ? "success" : "neutral"}>{row.band}</Badge>
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);
