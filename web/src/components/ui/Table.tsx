import type { ReactNode } from "react";

export type CellTone = "default" | "gain" | "loss" | "muted";

export interface TableCellValue {
  value: ReactNode;
  tone?: CellTone;
}

export type TableCell = ReactNode | TableCellValue;

export interface TableRow {
  key: string;
  label: ReactNode;
  cells: TableCell[];
}

const TONES: Record<CellTone, string> = {
  default: "text-text",
  gain: "text-accent",
  loss: "text-danger",
  muted: "text-dim",
};

function isCellValue(cell: TableCell): cell is TableCellValue {
  return typeof cell === "object" && cell !== null && !Array.isArray(cell) && "value" in cell && !("$$typeof" in cell);
}

// A labelled-row data table: first column is the row label, the rest are right-aligned values. Scrolls sideways inside
// its own box on a phone instead of widening the page.
export default function Table({ columns, rows, caption }: { columns: ReactNode[]; rows: TableRow[]; caption?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[320px] font-mono text-small">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="text-micro uppercase tracking-wide text-dim">
            <th scope="col" className="pb-2 text-left font-normal">
              {columns[0]}
            </th>
            {columns.slice(1).map((c, i) => (
              <th key={i} scope="col" className="pb-2 text-right font-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-t border-border-subtle">
              <th scope="row" className="py-1.5 text-left font-normal text-dim">
                {row.label}
              </th>
              {row.cells.map((cell, i) => {
                const { value, tone } = isCellValue(cell) ? cell : { value: cell, tone: "default" as CellTone };
                return (
                  <td key={i} className={`py-1.5 text-right ${TONES[tone ?? "default"]}`}>
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
