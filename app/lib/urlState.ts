// Shareable configuration in the URL hash, the way TensorFlow Playground does
// it: every control writes to the hash, and loading that hash restores the
// exact experiment.

import { ACTIVATION_IDS, type ActivationId } from "./activations";
import {
  BATCH_SIZES,
  ERROR_TARGETS,
  LEARNING_RATES,
  MAX_LAYERS,
  MAX_NEURONS,
  NOISE_LEVELS,
  REG_RATES,
  TARGETS,
  type Regularization,
  type TargetId,
} from "./model";
import { PDES, type PdeId } from "./pde";

export type Mode = "fit" | "pde";

export type PlaygroundConfig = {
  mode: Mode;
  compare: boolean;
  activation: ActivationId;
  target: TargetId;
  shape: number[];
  learningRate: number;
  regularization: Regularization;
  regRate: number;
  batchSize: number;
  noise: number;
  percentTrain: number;
  dataSeed: number;
  netSeed: number;
  showTest: boolean;
  errorTarget: number;
  /** Initial Cauchy shape parameters, φ(x) = (λ₁x + λ₂)/(x² + d²). */
  l1: number;
  l2: number;
  d: number;
  showDerivative: boolean;
  pde: PdeId;
  pdeLearningRate: number;
};

export const DEFAULT_CONFIG: PlaygroundConfig = {
  mode: "fit",
  compare: false,
  activation: "cauchy",
  target: "sin",
  shape: [6, 4],
  learningRate: 0.03,
  regularization: "none",
  regRate: 0,
  batchSize: 10,
  noise: 0,
  percentTrain: 50,
  dataSeed: 20250807,
  netSeed: 1337,
  showTest: false,
  errorTarget: 1e-3,
  l1: 1,
  l2: 0,
  d: 0.5,
  showDerivative: false,
  pde: "poisson",
  pdeLearningRate: 0.02,
};

const REGULARIZATIONS: Regularization[] = ["none", "l1", "l2"];
const PDE_LRS = [0.005, 0.01, 0.02, 0.05];

function pickFrom<T extends string>(raw: string | null, allowed: readonly T[], fallback: T): T {
  return allowed.includes(raw as T) ? (raw as T) : fallback;
}

function pickNumber(raw: string | null, allowed: readonly number[], fallback: number): number {
  const value = Number(raw);
  return raw !== null && allowed.includes(value) ? value : fallback;
}

function clampNumber(raw: string | null, min: number, max: number, fallback: number): number {
  const value = Number(raw);
  if (raw === null || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function parseShape(raw: string | null, fallback: number[]): number[] {
  if (raw === null) return fallback;
  if (raw === "") return [];
  const parts = raw.split(",").map((part) => Number(part));
  if (parts.some((part) => !Number.isInteger(part) || part < 1 || part > MAX_NEURONS)) {
    return fallback;
  }
  return parts.slice(0, MAX_LAYERS);
}

export function readConfig(hash: string): PlaygroundConfig {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const d = DEFAULT_CONFIG;
  return {
    mode: pickFrom(params.get("mode"), ["fit", "pde"] as const, d.mode),
    compare: params.get("compare") === "true",
    activation: pickFrom(params.get("activation"), ACTIVATION_IDS as ActivationId[], d.activation),
    target: pickFrom(params.get("target"), TARGETS.map((t) => t.id), d.target),
    shape: parseShape(params.get("shape"), d.shape),
    learningRate: pickNumber(params.get("lr"), LEARNING_RATES, d.learningRate),
    regularization: pickFrom(params.get("reg"), REGULARIZATIONS, d.regularization),
    regRate: pickNumber(params.get("regRate"), REG_RATES, d.regRate),
    batchSize: pickNumber(params.get("batch"), BATCH_SIZES, d.batchSize),
    noise: pickNumber(params.get("noise"), NOISE_LEVELS, d.noise),
    percentTrain: clampNumber(params.get("train"), 10, 90, d.percentTrain),
    dataSeed: clampNumber(params.get("dataSeed"), 0, 2 ** 31, d.dataSeed),
    netSeed: clampNumber(params.get("netSeed"), 0, 2 ** 31, d.netSeed),
    showTest: params.get("showTest") === "true",
    errorTarget: pickNumber(params.get("errTarget"), ERROR_TARGETS, d.errorTarget),
    l1: clampNumber(params.get("l1"), -3, 3, d.l1),
    l2: clampNumber(params.get("l2"), -3, 3, d.l2),
    d: clampNumber(params.get("d"), 0.1, 2, d.d),
    showDerivative: params.get("deriv") === "true",
    pde: pickFrom(params.get("pde"), PDES.map((p) => p.id) as PdeId[], d.pde),
    pdeLearningRate: pickNumber(params.get("pdeLr"), PDE_LRS, d.pdeLearningRate),
  };
}

export function writeConfig(config: PlaygroundConfig): string {
  const params = new URLSearchParams({
    mode: config.mode,
    compare: String(config.compare),
    activation: config.activation,
    target: config.target,
    shape: config.shape.join(","),
    lr: String(config.learningRate),
    reg: config.regularization,
    regRate: String(config.regRate),
    batch: String(config.batchSize),
    noise: String(config.noise),
    train: String(config.percentTrain),
    dataSeed: String(config.dataSeed),
    netSeed: String(config.netSeed),
    showTest: String(config.showTest),
    errTarget: String(config.errorTarget),
    l1: String(config.l1),
    l2: String(config.l2),
    d: String(config.d),
    deriv: String(config.showDerivative),
    pde: config.pde,
    pdeLr: String(config.pdeLearningRate),
  });
  return `#${params.toString()}`;
}

export { PDE_LRS };
