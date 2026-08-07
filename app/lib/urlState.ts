// Shareable configuration in the URL hash, the way TensorFlow Playground does
// it: every control writes to the hash, and loading that hash restores the
// exact experiment.

import {
  ACTIVATIONS,
  BATCH_SIZES,
  LEARNING_RATES,
  MAX_LAYERS,
  MAX_NEURONS,
  REG_RATES,
  TARGETS,
  type Activation,
  type Regularization,
  type TargetId,
} from "./model";

export type PlaygroundConfig = {
  activation: Activation;
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
};

export const DEFAULT_CONFIG: PlaygroundConfig = {
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
};

const REGULARIZATIONS: Regularization[] = ["none", "l1", "l2"];

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
    activation: pickFrom(params.get("activation"), ACTIVATIONS.map((a) => a.id), d.activation),
    target: pickFrom(params.get("target"), TARGETS.map((t) => t.id), d.target),
    shape: parseShape(params.get("shape"), d.shape),
    learningRate: pickNumber(params.get("lr"), LEARNING_RATES, d.learningRate),
    regularization: pickFrom(params.get("reg"), REGULARIZATIONS, d.regularization),
    regRate: pickNumber(params.get("regRate"), REG_RATES, d.regRate),
    batchSize: pickNumber(params.get("batch"), BATCH_SIZES, d.batchSize),
    noise: clampNumber(params.get("noise"), 0, 0.5, d.noise),
    percentTrain: clampNumber(params.get("train"), 10, 90, d.percentTrain),
    dataSeed: clampNumber(params.get("dataSeed"), 0, 2 ** 31, d.dataSeed),
    netSeed: clampNumber(params.get("netSeed"), 0, 2 ** 31, d.netSeed),
    showTest: params.get("showTest") === "true",
  };
}

export function writeConfig(config: PlaygroundConfig): string {
  const params = new URLSearchParams({
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
  });
  return `#${params.toString()}`;
}
