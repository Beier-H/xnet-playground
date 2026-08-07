// Core model for the Cauchy activation playground.
//
// A small fully-connected network on a 1-D input, trained with mini-batch SGD.
// Every hidden unit carries its own Cauchy shape parameters (l1, l2, sigma), so
// under the "cauchy" activation the network learns the *shape* of its
// nonlinearity as well as its weights. ReLU and tanh ignore those parameters,
// which is what makes the three settings comparable.

export type Activation = "cauchy" | "relu" | "tanh";
export type TargetId = "sin" | "exp" | "abs" | "step";
export type Regularization = "none" | "l1" | "l2";

export const X_MIN = -1;
export const X_MAX = 1;

/** Shared x grid for the approximation plot and the hovered-neuron overlay. */
export const PLOT_XS = Array.from({ length: 97 }, (_, i) => X_MIN + (i * (X_MAX - X_MIN)) / 96);

/** Smallest half-width a Cauchy unit may take, so the denominator never vanishes. */
const SIGMA_FLOOR = 0.08;
/** Per-parameter gradient clip. Rational activations can spike; this keeps SGD sane. */
const GRAD_CLIP = 8;
const PARAM_CLIP = 60;

export const ACTIVATIONS: { id: Activation; label: string }[] = [
  { id: "cauchy", label: "Cauchy" },
  { id: "relu", label: "ReLU" },
  { id: "tanh", label: "Tanh" },
];

export const TARGETS: { id: TargetId; label: string; note: string }[] = [
  { id: "sin", label: "sin(πx)", note: "smooth, analytic" },
  { id: "exp", label: "eˣ⁄e", note: "smooth, monotone" },
  { id: "abs", label: "|x|", note: "kink at x = 0" },
  { id: "step", label: "step(x)", note: "discontinuous" },
];

export const LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
export const REG_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1];
export const BATCH_SIZES = [1, 2, 5, 10, 20, 30];

export const MAX_LAYERS = 4;
export const MAX_NEURONS = 8;

export function targetValue(target: TargetId, x: number): number {
  switch (target) {
    case "sin":
      return Math.sin(Math.PI * x);
    case "exp":
      return Math.exp(x) / Math.E;
    case "abs":
      return Math.abs(x);
    case "step":
      return x < 0 ? -0.6 : 0.6;
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

const POINT_COUNT = 120;

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
  /** Cauchy shape parameters. Unused by ReLU and tanh. */
  l1: number;
  l2: number;
  /** Pre-softplus half-width; the effective sigma is `sigmaOf(neuron)`. */
  sRaw: number;
};

/** Hidden layers followed by a single linear output neuron. */
export type Network = Neuron[][];

export function sigmaOf(n: Neuron): number {
  return softplus(n.sRaw) + SIGMA_FLOOR;
}

function softplus(x: number): number {
  return x > 18 ? x : Math.log1p(Math.exp(x));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** `shape` lists hidden layer widths; the output neuron is appended here. */
export function buildNetwork(shape: number[], seed: number): Network {
  const rng = mulberry32(seed);
  const widths = [...shape, 1];
  const net: Network = [];
  let fanIn = 1;

  for (const width of widths) {
    const layer: Neuron[] = [];
    // Scale weights by fan-in so deeper stacks start in a sane range.
    const spread = 1.6 / Math.sqrt(fanIn);
    for (let j = 0; j < width; j += 1) {
      layer.push({
        w: Array.from({ length: fanIn }, () => (rng() * 2 - 1) * spread),
        b: (rng() * 2 - 1) * 0.6,
        l1: (rng() * 2 - 1) * 0.5,
        l2: (rng() * 2 - 1) * 0.5,
        sRaw: -0.2 + (rng() * 2 - 1) * 0.3,
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

function cloneNetwork(net: Network): Network {
  return net.map((layer) => layer.map((n) => ({ ...n, w: [...n.w] })));
}

// ------------------------------------------------------------ forward / loss

function activate(n: Neuron, z: number, act: Activation): number {
  if (act === "relu") return Math.max(0, z);
  if (act === "tanh") return Math.tanh(z);
  const s = sigmaOf(n);
  return (n.l1 * z + n.l2) / (z * z + s * s);
}

/**
 * Runs the network and keeps every intermediate value.
 * `acts[0]` is the input; `acts[l + 1]` is the output of layer `l`.
 */
function forward(net: Network, x: number, act: Activation) {
  const acts: number[][] = [[x]];
  const zs: number[][] = [[]];

  net.forEach((layer, l) => {
    const prev = acts[l];
    const isOutput = l === net.length - 1;
    const z: number[] = [];
    const a: number[] = [];
    for (const n of layer) {
      let sum = n.b;
      for (let i = 0; i < prev.length; i += 1) sum += n.w[i] * prev[i];
      z.push(sum);
      // The output neuron stays linear so the network can span any range.
      a.push(isOutput ? sum : activate(n, sum, act));
    }
    zs.push(z);
    acts.push(a);
  });

  return { acts, zs };
}

export function predict(net: Network, x: number, act: Activation): number {
  const { acts } = forward(net, x, act);
  return acts[acts.length - 1][0];
}

export function loss(net: Network, points: Point[], act: Activation): number {
  if (points.length === 0) return 0;
  let total = 0;
  for (const p of points) {
    const err = predict(net, p.x, act) - p.y;
    total += err * err;
  }
  return total / points.length;
}

/**
 * Output of every hidden neuron across `xs`, indexed `[layer][neuron][sample]`.
 * Drives the per-neuron thumbnails in the network diagram.
 */
export function neuronCurves(
  net: Network,
  xs: number[],
  act: Activation,
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

// ------------------------------------------------------------------ training

export type TrainOptions = {
  activation: Activation;
  learningRate: number;
  batchSize: number;
  regularization: Regularization;
  regRate: number;
};

type Grad = { w: number[]; b: number; l1: number; l2: number; sRaw: number };

function zeroGrads(net: Network): Grad[][] {
  return net.map((layer) =>
    layer.map((n) => ({ w: new Array(n.w.length).fill(0), b: 0, l1: 0, l2: 0, sRaw: 0 })),
  );
}

function backprop(
  net: Network,
  grads: Grad[][],
  point: Point,
  act: Activation,
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
      const a = acts[l + 1][j];
      const g = grads[l][j];
      let dZ: number;

      if (isOutput) {
        dZ = dA[j]; // linear output neuron
      } else {
        let dPhi: number;
        if (act === "relu") {
          dPhi = z > 0 ? 1 : 0;
        } else if (act === "tanh") {
          dPhi = 1 - a * a;
        } else {
          const s = sigmaOf(n);
          const den = z * z + s * s;
          const num = n.l1 * z + n.l2;
          dPhi = (n.l1 * den - 2 * z * num) / (den * den);
          // The shape parameters only exist on the Cauchy path.
          g.l1 += (dA[j] * z) / den;
          g.l2 += dA[j] / den;
          g.sRaw += dA[j] * ((-2 * s * num) / (den * den)) * sigmoid(n.sRaw);
        }
        dZ = dA[j] * dPhi;
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

/** One pass over the training set in mini-batches. Returns a new network. */
export function trainEpoch(
  net: Network,
  train: Point[],
  opts: TrainOptions,
  rng: () => number,
): Network {
  if (train.length === 0) return net;

  const next = cloneNetwork(net);
  const order = train.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

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
          const step = clip(g.w[i], GRAD_CLIP) + regGradient(n.w[i], opts.regularization, opts.regRate);
          n.w[i] = settle(n.w[i] - lr * step, n.w[i]);
        }
        n.b = settle(n.b - lr * clip(g.b, GRAD_CLIP), n.b);
        if (opts.activation === "cauchy") {
          n.l1 = settle(n.l1 - lr * clip(g.l1, GRAD_CLIP), n.l1);
          n.l2 = settle(n.l2 - lr * clip(g.l2, GRAD_CLIP), n.l2);
          n.sRaw = settle(n.sRaw - lr * clip(g.sRaw, GRAD_CLIP), n.sRaw, 6);
        }
      });
    });
  }

  return next;
}
