// Reference numbers quoted from published work, shown in the UI strictly as
// "Paper Result" and never mixed with what the playground computes live.
//
// ─────────────────────────────────────────────────────────────────────────────
// THESE ARE INTENTIONALLY EMPTY.
//
// Nobody has supplied the XNet paper's reported figures, and inventing numbers
// that render under a "Paper Result" label would be fabricating a citation.
// The panel therefore renders each row as "not supplied" until you fill it in.
//
// To populate: set `citation` below, then give a row a `value` (and `unit`).
// Anything left as `null` keeps showing the empty state, so a half-filled table
// is still honest.
// ─────────────────────────────────────────────────────────────────────────────

export type BenchmarkRow = {
  task: string;
  metric: string;
  /** null until a real reported figure is supplied. */
  value: number | null;
  unit?: string;
  /** Comparison baseline the paper reported alongside it, if any. */
  baseline?: { label: string; value: number | null };
  note?: string;
};

export type PaperBenchmarks = {
  /** Full citation. Shown verbatim under the table. */
  citation: string | null;
  rows: BenchmarkRow[];
};

export const PAPER_BENCHMARKS: PaperBenchmarks = {
  citation: null,
  rows: [
    {
      task: "Heat equation",
      metric: "relative L² error",
      value: null,
      baseline: { label: "Tanh PINN", value: null },
    },
    {
      task: "Poisson equation",
      metric: "relative L² error",
      value: null,
      baseline: { label: "Tanh PINN", value: null },
    },
    {
      task: "Burgers equation",
      metric: "relative L² error",
      value: null,
      baseline: { label: "Tanh PINN", value: null },
    },
    {
      task: "Heaviside approximation",
      metric: "max error",
      value: null,
      baseline: { label: "Fixed activation", value: null },
    },
  ],
};

export const hasPaperData = PAPER_BENCHMARKS.rows.some((r) => r.value !== null);
