// Core model for the Cauchy activation playground.
//
// A small fully-connected network on a 1-D input, trained with mini-batch SGD.
// Under the "cauchy" activation every hidden unit also carries its own shape
// parameters (λ₁, λ₂, d), so the network learns the shape of its nonlinearity
// as well as its weights. Every other activation has a fixed shape, which is
// the comparison the playground exists to make visible.

import {
  ACTIVATION_IDS,
  D_FLOOR,
  dPhi,
  phi,
  shapeGradient,
  sigmoid,
  softplus,
  activationMeta,
  type ActivationId,
  type ShapeParams,
} from "./activations";

export type { ActivationId, ShapeParams } from "./activations";
export type TargetId = "step" | "sin" | "highsin" | "runge" | "abs" | "exp";
export type Regularization = "none" | "l1" | "l2";

export const X_MIN = -1;
export const X_MAX = 1;

/** Shared x grid for the approximation plot and the hovered-neuron overlay. */
export const PLOT_XS = Array.from({ length: 97 }, (_, i) => X_MIN + (i * (X_MAX - X_MIN)) / 96);

/** Denser grid used only for measuring true approximation error. */
const ERROR_XS = Array.from({ length: 401 }, (_, i) => X_MIN + (i * (X_MAX - X_MIN)) / 400);

/** Per-parameter gradient clip. Rational activations can spike; this keeps SGD sane. */
const GRAD_CLIP = 8;
const PARAM_CLIP = 60;

export const LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
export const REG_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1];
export const BATCH_SIZES = [1, 2, 5, 10, 20, 30];
/** Noise levels are discrete so a comparison can be repeated exactly. */
export const NOISE_LEVELS = [0, 0.05, 0.1, 0.2];
export const ERROR_TARGETS = [1e-2, 1e-3, 1e-4];

export const MAX_LAYERS = 4;
export const MAX_NEURONS = 8;

/** Widths swept by the Neuron Efficiency benchmark. */
export const BENCHMARK_WIDTHS = [8, 16, 32, 64, 128, 256];
export const BENCHMARK_EPOCHS = [100, 200, 400];
/** Half-width of the window used for the discontinuity metric. */
export const LOCAL_RADIUS = 0.15;

/** The three activations pitted against each other in Compare mode. */
export const COMPARE_SET: ActivationId[] = ["cauchy", "relu", "tanh"];

export const TARGETS: {
  id: TargetId;
  label: string;
  note: string;
  /** Flagged in the UI as the headline XNet benchmark. */
  featured?: boolean;
}[] = [
  { id: "step", label: "step(x)", note: "Heaviside — discontinuous", featured: true },
  { id: "sin", label: "sin(πx)", note: "smooth, analytic" },
  { id: "highsin", label: "sin(10πx)", note: "high frequency" },
  { id: "runge", label: "Runge", note: "1/(1+25x²) — sharp peak" },
  { id: "abs", label: "|x|", note: "kink at x = 0" },
  { id: "exp", label: "eˣ⁄e", note: "smooth, monotone" },
];

export function targetValue(target: TargetId, x: number): number {
  switch (target) {
    case "step":
      // Heaviside, the discontinuous benchmark from the XNet work.
      return x < 0 ? 0 : 1;
    case "sin":
      return Math.sin(Math.PI * x);
    case "highsin":
      return Math.sin(10 * Math.PI * x);
    case "runge":
      return 1 / (1 + 25 * x * x);
    case "abs":
      return Math.abs(x);
    case "exp":
      return Math.exp(x) / Math.E;
  }
}

// ---------------------------------------------------------------- randomness

/** Deterministic PRNG, so a given seed renders identically on server and client. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ------------------------------------------------------------------- dataset

export type Point = { x: number; y: number };
export type Dataset = { train: Point[]; test: Point[] };

const POINT_COUNT = 160;

export function makeDataset(
  target: TargetId,
  noise: number,
  percentTrain: number,
  seed: number,
): Dataset {
  const rng = mulberry32(seed);
  const points: Point[] = Array.from({ length: POINT_COUNT }, () => {
    const x = X_MIN + rng() * (X_MAX - X_MIN);
    return { x, y: targetValue(target, x) + gaussian(rng) * noise };
  });

  // Shuffle before splitting so the two halves cover the same x range.
  for (let i = points.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [points[i], points[j]] = [points[j], points[i]];
  }
  const cut = Math.round((percentTrain / 100) * points.length);
  return { train: points.slice(0, cut), test: points.slice(cut) };
}

// ------------------------------------------------------------------- network

export type Neuron = {
  /** Incoming weights, one per unit in the previous layer. */
  w: number[];
  b: number;
  /** Cauchy shape parameters. Ignored by every fixed-shape activation. */
  l1: number;
  l2: number;
  /** Pre-softplus half-width; the effective d is `shapeOf(neuron).d`. */
  dRaw: number;
};

/** Hidden layers followed by a single linear output neuron. */
export type Network = Neuron[][];

export function shapeOf(n: Neuron): ShapeParams {
  return { l1: n.l1, l2: n.l2, d: softplus(n.dRaw) + D_FLOOR };
}

/** Inverse of `softplus(dRaw) + D_FLOOR`, for seeding d from a UI control. */
function dToRaw(d: number): number {
  const above = Math.max(1e-3, d - D_FLOOR);
  return Math.log(Math.expm1(above) + 1e-12);
}

/** `shape` lists hidden layer widths; the output neuron is appended here. */
export function buildNetwork(
  shape: number[],
  seed: number,
  init: ShapeParams = { l1: 1, l2: 0, d: 0.5 },
): Network {
  const rng = mulberry32(seed);
  const widths = [...shape, 1];
  const net: Network = [];
  let fanIn = 1;
  const baseDRaw = dToRaw(init.d);

  for (const width of widths) {
    const layer: Neuron[] = [];
    // Scale weights by fan-in so deeper stacks start in a sane range.
    const spread = 1.6 / Math.sqrt(fanIn);
    for (let j = 0; j < width; j += 1) {
      layer.push({
        w: Array.from({ length: fanIn }, () => (rng() * 2 - 1) * spread),
        b: (rng() * 2 - 1) * 0.6,
        // Jitter around the configured shape so units can specialise.
        l1: init.l1 + (rng() * 2 - 1) * 0.25,
        l2: init.l2 + (rng() * 2 - 1) * 0.25,
        dRaw: baseDRaw + (rng() * 2 - 1) * 0.25,
      });
    }
    net.push(layer);
    fanIn = width;
  }
  return net;
}

export function networkShape(net: Network): number[] {
  return net.slice(0, -1).map((layer) => layer.length);
}

/** Trainable parameter count, including shape parameters where they exist. */
export function parameterCount(shape: number[], activation: ActivationId): number {
  const extra = activationMeta(activation).shapeParams;
  let fanIn = 1;
  let total = 0;
  for (const width of shape) {
    total += width * (fanIn + 1 + extra);
    fanIn = width;
  }
  total += fanIn + 1; // linear output neuron
  return total;
}

function cloneNetwork(net: Network): Network {
  return net.map((layer) => layer.map((n) => ({ ...n, w: [...n.w] })));
}

// ------------------------------------------------------------ forward / loss

type Override = { layer: number; index: number; value: number };

/**
 * Runs the network and keeps every intermediate value.
 * `acts[0]` is the input; `acts[l + 1]` is the output of layer `l`.
 * An `override` pins one hidden unit's output, which is how contribution is
 * measured by ablation.
 */
function forward(net: Network, x: number, act: ActivationId, override?: Override) {
  const acts: number[][] = [[x]];
  const zs: number[][] = [[]];

  net.forEach((layer, l) => {
    const prev = acts[l];
    const isOutput = l === net.length - 1;
    const z: number[] = [];
    const a: number[] = [];
    layer.forEach((n, j) => {
      let sum = n.b;
      for (let i = 0; i < prev.length; i += 1) sum += n.w[i] * prev[i];
      z.push(sum);
      // The output neuron stays linear so the network can span any range.
      let value = isOutput ? sum : phi(act, sum, shapeOf(n));
      if (override && override.layer === l && override.index === j) value = override.value;
      a.push(value);
    });
    zs.push(z);
    acts.push(a);
  });

  return { acts, zs };
}

export function predict(net: Network, x: number, act: ActivationId): number {
  const { acts } = forward(net, x, act);
  return acts[acts.length - 1][0];
}

export function loss(net: Network, points: Point[], act: ActivationId): number {
  if (points.length === 0) return 0;
  let total = 0;
  for (const p of points) {
    const err = predict(net, p.x, act) - p.y;
    total += err * err;
  }
  return total / points.length;
}

/**
 * Squared error against the *noiseless* target on a dense grid.
 *
 * Distinct from training loss, which is measured on finitely many noisy
 * samples: this is the quantity that actually says how well the function was
 * approximated, and it is the one that exposes overfitting to noise.
 */
export function functionMse(net: Network, act: ActivationId, target: TargetId): number {
  let total = 0;
  for (const x of ERROR_XS) {
    const err = predict(net, x, act) - targetValue(target, x);
    total += err * err;
  }
  return total / ERROR_XS.length;
}

/**
 * Squared error against the target restricted to |x| ≤ radius.
 *
 * The headline number for the Heaviside benchmark: a global MSE is dominated by
 * the two flat halves, where every activation does fine, and hides what happens
 * at the jump — which is the only interesting part.
 */
export function localMse(
  net: Network,
  act: ActivationId,
  target: TargetId,
  radius = LOCAL_RADIUS,
): number {
  const N = 121;
  let total = 0;
  for (let i = 0; i < N; i += 1) {
    const x = -radius + (2 * radius * i) / (N - 1);
    const err = predict(net, x, act) - targetValue(target, x);
    total += err * err;
  }
  return total / N;
}

export type Localization = { mu: number; width: number };

/**
 * Where a Cauchy neuron sits and how wide it is, in input space.
 *
 * With z = w·x + b the activation can be rewritten
 *   φ(z) ∝ [λ₁w(x − μ) + λ₂] / [(x − μ)² + (d/|w|)²],  μ = −b/w
 * so the centre and width are already implied by the existing parameters — no
 * extra ones are introduced. The crossing is located numerically so that this
 * also works for neurons in deeper layers, where z is no longer linear in x;
 * there it is a local linearisation around the point where z vanishes.
 */
export function effectiveLocalization(
  net: Network,
  xs: number[],
  act: ActivationId,
  layer: number,
  index: number,
): Localization | null {
  if (act !== "cauchy") return null;
  const neuron = net[layer]?.[index];
  if (!neuron) return null;

  const zs = xs.map((x) => forward(net, x, act).zs[layer + 1][index]);
  let best = 0;
  for (let i = 1; i < zs.length; i += 1) {
    if (Math.abs(zs[i]) < Math.abs(zs[best])) best = i;
  }

  const lo = Math.max(0, best - 1);
  const hi = Math.min(zs.length - 1, best + 1);
  const slope = (zs[hi] - zs[lo]) / (xs[hi] - xs[lo]);
  if (!Number.isFinite(slope) || Math.abs(slope) < 1e-9) return null;

  // Linear estimate of the crossing; exact for a first-layer neuron.
  const mu = xs[best] - zs[best] / slope;
  const width = shapeOf(neuron).d / Math.abs(slope);
  if (!Number.isFinite(mu) || !Number.isFinite(width)) return null;
  return { mu, width };
}

/**
 * Output of every hidden neuron across `xs`, indexed `[layer][neuron][sample]`.
 * Drives the per-neuron thumbnails in the network diagram.
 */
export function neuronCurves(
  net: Network,
  xs: number[],
  act: ActivationId,
): number[][][] {
  const hiddenCount = net.length - 1;
  const curves: number[][][] = Array.from({ length: hiddenCount }, (_, l) =>
    Array.from({ length: net[l].length }, () => [] as number[]),
  );
  for (const x of xs) {
    const { acts } = forward(net, x, act);
    for (let l = 0; l < hiddenCount; l += 1) {
      for (let j = 0; j < net[l].length; j += 1) curves[l][j].push(acts[l + 1][j]);
    }
  }
  return curves;
}

export type Contribution = {
  /** Own output of the neuron across `xs`. */
  own: number[];
  /** Change in network output when this neuron is pinned to its mean. */
  delta: number[];
  /** x-range where |delta| is at least half its peak. */
  band: { lo: number; hi: number } | null;
  peak: number;
};

/**
 * How much one neuron actually matters, measured by ablation: pin its output to
 * its own mean and see how far the network's output moves. This works for a
 * neuron in any layer, unlike simply reading its outgoing weight.
 */
export function neuronContribution(
  net: Network,
  xs: number[],
  act: ActivationId,
  layer: number,
  index: number,
): Contribution {
  const own = xs.map((x) => forward(net, x, act).acts[layer + 1][index]);
  const mean = own.reduce((s, v) => s + v, 0) / (own.length || 1);

  const delta = xs.map((x, i) => {
    const full = forward(net, x, act).acts[net.length][0];
    const ablated = forward(net, x, act, { layer, index, value: mean }).acts[net.length][0];
    void i;
    return full - ablated;
  });

  const abs = delta.map(Math.abs);
  const peak = Math.max(...abs);
  let band: { lo: number; hi: number } | null = null;
  if (peak > 1e-9) {
    // Contiguous run around the peak where influence stays above half.
    const centre = abs.indexOf(peak);
    let lo = centre;
    let hi = centre;
    while (lo > 0 && abs[lo - 1] >= peak * 0.5) lo -= 1;
    while (hi < abs.length - 1 && abs[hi + 1] >= peak * 0.5) hi += 1;
    band = { lo: xs[lo], hi: xs[hi] };
  }

  return { own, delta, band, peak };
}

// ------------------------------------------------------------------ training

export type TrainOptions = {
  activation: ActivationId;
  learningRate: number;
  batchSize: number;
  regularization: Regularization;
  regRate: number;
};

type Grad = { w: number[]; b: number; l1: number; l2: number; dRaw: number };

function zeroGrads(net: Network): Grad[][] {
  return net.map((layer) =>
    layer.map((n) => ({ w: new Array(n.w.length).fill(0), b: 0, l1: 0, l2: 0, dRaw: 0 })),
  );
}

function backprop(
  net: Network,
  grads: Grad[][],
  point: Point,
  act: ActivationId,
  scale: number,
) {
  const { acts, zs } = forward(net, point.x, act);
  const output = acts[acts.length - 1][0];

  // d(squared error)/d(output), pre-scaled by 1 / batchSize.
  let dA = [2 * (output - point.y) * scale];

  for (let l = net.length - 1; l >= 0; l -= 1) {
    const prev = acts[l];
    const dPrev = new Array(prev.length).fill(0);
    const isOutput = l === net.length - 1;

    net[l].forEach((n, j) => {
      const z = zs[l + 1][j];
      const g = grads[l][j];
      let dZ: number;

      if (isOutput) {
        dZ = dA[j]; // linear output neuron
      } else {
        const params = shapeOf(n);
        dZ = dA[j] * dPhi(act, z, params);
        if (act === "cauchy") {
          const sg = shapeGradient(act, z, params);
          g.l1 += dA[j] * sg.l1;
          g.l2 += dA[j] * sg.l2;
          // Chain through d = softplus(dRaw) + floor.
          g.dRaw += dA[j] * sg.d * sigmoid(n.dRaw);
        }
      }

      g.b += dZ;
      for (let i = 0; i < prev.length; i += 1) {
        g.w[i] += dZ * prev[i];
        dPrev[i] += dZ * n.w[i];
      }
    });

    dA = dPrev;
  }
}

function clip(value: number, limit: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-limit, Math.min(limit, value));
}

/** Keeps a parameter finite and bounded; falls back to its previous value. */
function settle(next: number, previous: number, limit = PARAM_CLIP): number {
  if (!Number.isFinite(next)) return previous;
  return Math.max(-limit, Math.min(limit, next));
}

function regGradient(w: number, reg: Regularization, rate: number): number {
  if (reg === "l1") return rate * Math.sign(w);
  if (reg === "l2") return rate * w;
  return 0;
}

/** A shuffled visiting order. Shared across runs so a comparison is exact. */
export function shuffledOrder(count: number, rng: () => number): number[] {
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

/**
 * One pass over the training set in mini-batches. Returns a new network.
 *
 * `order` is supplied by the caller rather than drawn here so that Compare mode
 * can hand every activation the identical sequence of mini-batches.
 */
export function trainEpoch(
  net: Network,
  train: Point[],
  opts: TrainOptions,
  order: number[],
): Network {
  if (train.length === 0) return net;

  const next = cloneNetwork(net);
  const isCauchy = opts.activation === "cauchy";

  for (let start = 0; start < order.length; start += opts.batchSize) {
    const batch = order.slice(start, start + opts.batchSize);
    const grads = zeroGrads(next);
    const scale = 1 / batch.length;
    for (const index of batch) backprop(next, grads, train[index], opts.activation, scale);

    const lr = opts.learningRate;
    next.forEach((layer, l) => {
      layer.forEach((n, j) => {
        const g = grads[l][j];
        for (let i = 0; i < n.w.length; i += 1) {
          const step =
            clip(g.w[i], GRAD_CLIP) + regGradient(n.w[i], opts.regularization, opts.regRate);
          n.w[i] = settle(n.w[i] - lr * step, n.w[i]);
        }
        n.b = settle(n.b - lr * clip(g.b, GRAD_CLIP), n.b);
        if (isCauchy) {
          n.l1 = settle(n.l1 - lr * clip(g.l1, GRAD_CLIP), n.l1);
          n.l2 = settle(n.l2 - lr * clip(g.l2, GRAD_CLIP), n.l2);
          n.dRaw = settle(n.dRaw - lr * clip(g.dRaw, GRAD_CLIP), n.dRaw, 6);
        }
      });
    });
  }

  return next;
}

// ------------------------------------------------- neuron efficiency benchmark

export type BenchmarkResult = {
  width: number;
  activation: ActivationId;
  trainMse: number;
  testMse: number;
  functionMse: number;
  epochsToTarget: number | null;
  runtimeMs: number;
  params: number;
};

export type BenchmarkJob = BenchmarkResult & {
  net: Network;
  epochsDone: number;
  done: boolean;
};

/**
 * One shuffled order per epoch, generated once and shared by every job.
 *
 * This is what makes the sweep a fair test: each activation at each width sees
 * not merely the same data and seed but the identical sequence of mini-batches,
 * so the only thing varying is the activation.
 */
export function makeEpochOrders(epochs: number, trainSize: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: epochs }, () => shuffledOrder(trainSize, rng));
}

export function createBenchmarkJobs(
  widths: number[],
  activations: ActivationId[],
  netSeed: number,
  init: ShapeParams,
): BenchmarkJob[] {
  const jobs: BenchmarkJob[] = [];
  for (const width of widths) {
    for (const activation of activations) {
      jobs.push({
        width,
        activation,
        // Same width, same seed, same initial shape for every activation.
        net: buildNetwork([width], netSeed, init),
        epochsDone: 0,
        done: false,
        trainMse: NaN,
        testMse: NaN,
        functionMse: NaN,
        epochsToTarget: null,
        runtimeMs: 0,
        params: parameterCount([width], activation),
      });
    }
  }
  return jobs;
}

/** How often the true approximation error is sampled while sweeping. */
const PROBE_EVERY = 5;

/**
 * Runs at most `slice` more epochs of one job and returns its updated state.
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
   * one epoch at 256 neurons costs roughly 30× one at 8, so a flat slice that
   * feels instant at the small end drops frames at the large end.
   */
  budgetMs = 16,
): BenchmarkJob {
  if (job.done) return job;

  const budget = orders.length;
  const limit = Math.min(budget, job.epochsDone + maxEpochs);
  let net = job.net;
  let epochsToTarget = job.epochsToTarget;
  let epoch = job.epochsDone;
  const started = performance.now();

  for (; epoch < limit; epoch += 1) {
    net = trainEpoch(net, dataset.train, { ...opts, activation: job.activation }, orders[epoch]);
    if (epochsToTarget === null && (epoch + 1) % PROBE_EVERY === 0) {
      if (functionMse(net, job.activation, target) <= errorTarget) epochsToTarget = epoch + 1;
    }
    // Always complete at least one epoch, then yield as soon as the budget is up.
    if (performance.now() - started >= budgetMs) {
      epoch += 1;
      break;
    }
  }

  const until = epoch;
  const elapsed = performance.now() - started;
  const done = until >= budget;

  return {
    ...job,
    net,
    epochsDone: until,
    done,
    epochsToTarget,
    runtimeMs: job.runtimeMs + elapsed,
    // Final metrics are only meaningful once a job finishes, but computing them
    // per slice keeps the table alive while the sweep runs.
    trainMse: computeMse(net, dataset.train, job.activation),
    testMse: computeMse(net, dataset.test, job.activation),
    functionMse: functionMse(net, job.activation, target),
  };
}

const computeMse = loss;

export { ACTIVATION_IDS };
