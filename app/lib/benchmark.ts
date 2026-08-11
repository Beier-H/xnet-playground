// Width / parameter-efficiency sweep.
//
// Lives in its own module because it spans two model families: the node-
// activation MLPs from `model.ts` and the edge-function KAN from `kan.ts`.
// Importing both from `model.ts` would create a cycle.

import { ACTIVATION_COLORS, activationMeta, type ActivationId } from "./activations";
import {
  buildKan,
  kanFunctionMse,
  kanLoss,
  kanParameterCount,
  kanPredict,
  kanTrainEpoch,
  type KanNet,
} from "./kan";
import {
  buildNetwork,
  functionMse,
  loss,
  mulberry32,
  parameterCount,
  predict,
  shuffledOrder,
  trainEpoch,
  type Dataset,
  type Network,
  type ShapeParams,
  type TargetId,
  type TrainOptions,
} from "./model";

/** Everything that can be trained and compared, across both model families. */
export type ModelId = ActivationId | "kan";

/** The sweep's line-up. KAN is the paper's baseline, so it belongs here. */
export const BENCHMARK_SET: ModelId[] = ["cauchy", "relu", "tanh", "kan"];

export const MODEL_COLORS: Record<string, string> = {
  ...ACTIVATION_COLORS,
  kan: "#9333ea",
};

export function modelColor(id: ModelId): string {
  return MODEL_COLORS[id] ?? "#5d6d82";
}

export function modelLabel(id: ModelId): string {
  return id === "kan" ? "KAN" : activationMeta(id).label;
}

export function modelParams(id: ModelId, width: number): number {
  return id === "kan" ? kanParameterCount(width) : parameterCount([width], id);
}

/** How the sweep's x axis is measured. */
export type SweepAxis = "width" | "params";

export type BenchmarkResult = {
  width: number;
  model: ModelId;
  trainMse: number;
  testMse: number;
  functionMse: number;
  epochsToTarget: number | null;
  runtimeMs: number;
  params: number;
};

type JobState = { kind: "mlp"; net: Network } | { kind: "kan"; net: KanNet };

export type BenchmarkJob = BenchmarkResult & {
  state: JobState;
  epochsDone: number;
  done: boolean;
};

/**
 * One shuffled order per epoch, generated once and shared by every job.
 *
 * This is what makes the sweep a fair test: each model at each width sees not
 * merely the same data and seed but the identical sequence of mini-batches, so
 * the only thing varying is the model.
 */
export function makeEpochOrders(epochs: number, trainSize: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: epochs }, () => shuffledOrder(trainSize, rng));
}

export function createBenchmarkJobs(
  widths: number[],
  models: ModelId[],
  netSeed: number,
  init: ShapeParams,
): BenchmarkJob[] {
  const jobs: BenchmarkJob[] = [];
  for (const width of widths) {
    for (const model of models) {
      jobs.push({
        width,
        model,
        // Same width and same seed for every model.
        state:
          model === "kan"
            ? { kind: "kan", net: buildKan(width, netSeed) }
            : { kind: "mlp", net: buildNetwork([width], netSeed, init) },
        epochsDone: 0,
        done: false,
        trainMse: NaN,
        testMse: NaN,
        functionMse: NaN,
        epochsToTarget: null,
        runtimeMs: 0,
        params: modelParams(model, width),
      });
    }
  }
  return jobs;
}

/** How often the true approximation error is sampled while sweeping. */
const PROBE_EVERY = 5;

function jobFunctionMse(state: JobState, model: ModelId, target: TargetId): number {
  return state.kind === "kan"
    ? kanFunctionMse(state.net, target)
    : functionMse(state.net, model as ActivationId, target);
}

function jobLoss(state: JobState, points: Dataset["train"], model: ModelId): number {
  return state.kind === "kan" ? kanLoss(state.net, points) : loss(state.net, points, model as ActivationId);
}

export function jobPredict(job: BenchmarkJob, x: number): number {
  return job.state.kind === "kan"
    ? kanPredict(job.state.net, x)
    : predict(job.state.net, x, job.model as ActivationId);
}

/**
 * Runs more epochs of one job and returns its updated state.
 *
 * Deliberately incremental: the caller yields to the browser between slices so
 * a 256-neuron sweep cannot freeze the page.
 */
export function advanceBenchmarkJob(
  job: BenchmarkJob,
  dataset: Dataset,
  target: TargetId,
  opts: Omit<TrainOptions, "activation">,
  orders: number[][],
  maxEpochs: number,
  errorTarget: number,
  /**
   * Wall-clock budget for this slice. A fixed epoch count is the wrong unit:
   * one epoch at 256 neurons costs far more than one at 8, so a flat slice that
   * feels instant at the small end drops frames at the large end.
   */
  budgetMs = 16,
): BenchmarkJob {
  if (job.done) return job;

  const budget = orders.length;
  const limit = Math.min(budget, job.epochsDone + maxEpochs);
  let state = job.state;
  let epochsToTarget = job.epochsToTarget;
  let epoch = job.epochsDone;
  const started = performance.now();

  for (; epoch < limit; epoch += 1) {
    state =
      state.kind === "kan"
        ? {
            kind: "kan",
            net: kanTrainEpoch(
              state.net,
              dataset.train,
              { learningRate: opts.learningRate, batchSize: opts.batchSize },
              orders[epoch],
            ),
          }
        : {
            kind: "mlp",
            net: trainEpoch(
              state.net,
              dataset.train,
              { ...opts, activation: job.model as ActivationId },
              orders[epoch],
            ),
          };

    if (epochsToTarget === null && (epoch + 1) % PROBE_EVERY === 0) {
      if (jobFunctionMse(state, job.model, target) <= errorTarget) epochsToTarget = epoch + 1;
    }
    // Always complete at least one epoch, then yield as soon as the budget is up.
    if (performance.now() - started >= budgetMs) {
      epoch += 1;
      break;
    }
  }

  const elapsed = performance.now() - started;

  return {
    ...job,
    state,
    epochsDone: epoch,
    done: epoch >= budget,
    epochsToTarget,
    runtimeMs: job.runtimeMs + elapsed,
    // Final metrics are only meaningful once a job finishes, but computing them
    // per slice keeps the table alive while the sweep runs.
    trainMse: jobLoss(state, dataset.train, job.model),
    testMse: jobLoss(state, dataset.test, job.model),
    functionMse: jobFunctionMse(state, job.model, target),
  };
}
