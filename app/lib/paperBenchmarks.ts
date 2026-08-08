// Figures reported in published work, shown in the UI strictly as
// "Paper Result" and never mixed with anything the playground computes.
//
// Every entry records its source, task, architecture, metric and reported
// value exactly as supplied. Nothing here is derived, rescaled or inferred:
// where a paper did not report a quantity (a runtime, a parameter count) the
// field is simply absent rather than estimated.
//
// These numbers are NOT comparable with the live playground results. The papers
// train far larger networks, for far longer, with different optimisers. The UI
// states that wherever the two appear near each other.

export type PaperEntry = {
  /** Method as named in the paper. */
  method: string;
  /** Layer sizes as reported, e.g. "[1,64,1]". */
  architecture: string;
  metric: string;
  value: number;
  /** Reported wall-clock time in seconds, where the paper gives one. */
  timeSeconds?: number;
  /** True for the Cauchy/XNet rows, so the table can mark them. */
  isXNet?: boolean;
  note?: string;
};

export type PaperTask = {
  task: string;
  source: string;
  entries: PaperEntry[];
};

const SHALLOW_XNET = "From Kolmogorov to Cauchy: Shallow XNet Surpasses KANs";
const CAUCHY_XNET = "Cauchy activation function and XNet";

export const PAPER_TASKS: PaperTask[] = [
  {
    task: "Heaviside step approximation",
    source: SHALLOW_XNET,
    entries: [
      { method: "XNet", architecture: "[1,64,1]", metric: "MSE", value: 8.99e-8, isXNet: true },
      { method: "ReLU (shallow)", architecture: "[1,64,1]", metric: "MSE", value: 2.05e-3 },
      { method: "ReLU (deep)", architecture: "[1,64,64,1]", metric: "MSE", value: 6.81e-5 },
      { method: "KAN", architecture: "[1,1], 200 grids", metric: "MSE", value: 5.98e-4 },
    ],
  },
  {
    task: "Heat equation",
    source: SHALLOW_XNET,
    entries: [
      { method: "MLP", architecture: "[2,20,20,1]", metric: "MSE", value: 2.4536e-5, timeSeconds: 43.8 },
      { method: "XNet", architecture: "[2,20,1]", metric: "MSE", value: 3.8936e-8, timeSeconds: 43.5, isXNet: true },
      { method: "KAN", architecture: "[2,10,1]", metric: "MSE", value: 1.5106e-7, timeSeconds: 254.6 },
      { method: "XNet", architecture: "[2,200,1]", metric: "MSE", value: 3.6867e-9, timeSeconds: 108.3, isXNet: true },
    ],
  },
  {
    task: "Poisson equation",
    source: SHALLOW_XNET,
    entries: [
      { method: "PINN", architecture: "[2,20,20,1]", metric: "MSE", value: 1.7998e-5, timeSeconds: 48.9 },
      { method: "XNet", architecture: "[2,20,1]", metric: "MSE", value: 1.8651e-8, timeSeconds: 57.2, isXNet: true },
      { method: "KAN", architecture: "[2,10,1]", metric: "MSE", value: 5.743e-8, timeSeconds: 286.3 },
      { method: "XNet", architecture: "[2,200,1]", metric: "MSE", value: 1.0937e-9, timeSeconds: 154.8, isXNet: true },
    ],
  },
  {
    task: "Poisson PINN",
    source: CAUCHY_XNET,
    entries: [
      { method: "Tanh PINN", architecture: "not reported", metric: "training loss", value: 0.0349 },
      { method: "Cauchy PINN", architecture: "not reported", metric: "training loss", value: 0.00354, isXNet: true },
    ],
  },
  {
    task: "Heat example",
    source: CAUCHY_XNET,
    entries: [
      { method: "Sigmoid PINN", architecture: "not reported", metric: "mean error", value: 2e-3 },
      { method: "Cauchy", architecture: "not reported", metric: "mean error", value: 6e-5, isXNet: true },
    ],
  },
];

export const PAPER_SOURCES = [SHALLOW_XNET, CAUCHY_XNET];
