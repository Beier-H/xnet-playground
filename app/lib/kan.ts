// A minimal Kolmogorov–Arnold Network, for live comparison against XNet.
//
// ─────────────────────────────────────────────────────────────────────────────
// THIS IS NOT THE REFERENCE KAN IMPLEMENTATION.
//
// The published KAN relies on grid updates, grid extension, LBFGS refinement,
// entropy regularization and a tuned initialization. None of that is here: the
// grid is fixed and it is trained with exactly the same plain mini-batch SGD,
// seed and batch order as every other model in this playground.
//
// That is deliberate — it is the only way the comparison isolates the
// *representation* rather than the training recipe — but it means this KAN will
// underperform the numbers reported in the KAN literature. Do not read a win
// over this implementation as a win over KAN. The UI labels it accordingly.
// ─────────────────────────────────────────────────────────────────────────────
//
// Structure, following the KAN formulation: learnable univariate functions live
// on the *edges* and are summed at the nodes, rather than fixed activations on
// nodes with learnable scalar weights on edges.
//
//   h_i = φ_i(x)              (input → hidden edge functions)
//   y   = Σ_i ψ_i(h_i)        (hidden → output edge functions)
//   φ(t) = w_b · silu(t) + w_s · Σ_j c_j B_j(t)

import { sigmoid } from "./activations";
import { mulberry32, targetValue, type Dataset, type Point, type TargetId } from "./model";

/** Cubic splines, matching the KAN default. */
export const SPLINE_ORDER = 3;
/** Grid intervals per edge. Basis count is GRID + SPLINE_ORDER. */
export const SPLINE_GRID = 5;

const GRAD_CLIP = 8;
const PARAM_CLIP = 60;

/** Input domain of the playground; layer 1 sees exactly this. */
const GRID_1: [number, number] = [-1, 1];
/**
 * Hidden activations are unbounded, so layer 2 gets a wider fixed grid. Outside
 * it the spline basis is zero and only the silu base term carries the edge —
 * the honest cost of not implementing grid updates.
 */
const GRID_2: [number, number] = [-3, 3];

export type KanEdge = {
  /** Weight on the silu base branch. */
  wb: number;
  /** Weight on the spline branch. */
  ws: number;
  /** B-spline coefficients. */
  c: number[];
};

export type KanNet = {
  width: number;
  knots1: number[];
  knots2: number[];
  /** input → hidden, one edge per hidden unit */
  e1: KanEdge[];
  /** hidden → output, one edge per hidden unit */
  e2: KanEdge[];
};

export const basisCount = () => SPLINE_GRID + SPLINE_ORDER;

/** Open-uniform knot vector extended by `SPLINE_ORDER` on each side. */
function makeKnots([a, b]: [number, number]): number[] {
  const h = (b - a) / SPLINE_GRID;
  const knots: number[] = [];
  for (let j = 0; j <= SPLINE_GRID + 2 * SPLINE_ORDER; j += 1) {
    knots.push(a + (j - SPLINE_ORDER) * h);
  }
  return knots;
}

/**
 * Cox–de Boor basis and its derivative, evaluated together.
 * Returns arrays of length `knots.length - SPLINE_ORDER - 1`.
 */
function basisAndDeriv(x: number, knots: number[]): { b: number[]; db: number[] } {
  const k = SPLINE_ORDER;
  const n = knots.length - 1;
  let prev = new Array<number>(n).fill(0);

  // Degree 0. The half-open convention leaves x exactly at the right edge with
  // no support, so close the final interval.
  for (let i = 0; i < n; i += 1) {
    const inside = x >= knots[i] && x < knots[i + 1];
    const atEnd = i === n - 1 && x === knots[n];
    prev[i] = inside || atEnd ? 1 : 0;
  }

  let lower: number[] = prev; // degree k-1, kept for the derivative
  for (let p = 1; p <= k; p += 1) {
    const size = n - p;
    const next = new Array<number>(size).fill(0);
    for (let i = 0; i < size; i += 1) {
      const d1 = knots[i + p] - knots[i];
      const d2 = knots[i + p + 1] - knots[i + 1];
      const left = d1 > 0 ? ((x - knots[i]) / d1) * prev[i] : 0;
      const right = d2 > 0 ? ((knots[i + p + 1] - x) / d2) * prev[i + 1] : 0;
      next[i] = left + right;
    }
    if (p === k - 1) lower = next;
    prev = next;
  }

  const count = knots.length - k - 1;
  const db = new Array<number>(count).fill(0);
  for (let i = 0; i < count; i += 1) {
    const d1 = knots[i + k] - knots[i];
    const d2 = knots[i + k + 1] - knots[i + 1];
    const a = d1 > 0 ? lower[i] / d1 : 0;
    const b2 = d2 > 0 ? lower[i + 1] / d2 : 0;
    db[i] = k * (a - b2);
  }

  return { b: prev.slice(0, count), db };
}

const silu = (x: number) => x * sigmoid(x);
const dSilu = (x: number) => {
  const s = sigmoid(x);
  return s * (1 + x * (1 - s));
};

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function buildKan(width: number, seed: number): KanNet {
  const rng = mulberry32(seed);
  const count = basisCount();
  const makeEdge = (scale: number): KanEdge => ({
    wb: (rng() * 2 - 1) * scale,
    ws: scale,
    c: Array.from({ length: count }, () => gaussian(rng) * 0.1 * scale),
  });

  return {
    width,
    knots1: makeKnots(GRID_1),
    knots2: makeKnots(GRID_2),
    e1: Array.from({ length: width }, () => makeEdge(1)),
    // The output sums `width` edges, so scale them down to keep it O(1).
    e2: Array.from({ length: width }, () => makeEdge(1 / Math.sqrt(width))),
  };
}

export function kanParameterCount(width: number): number {
  return 2 * width * (2 + basisCount());
}

function edgeValue(edge: KanEdge, t: number, b: number[]): number {
  let spline = 0;
  for (let j = 0; j < edge.c.length; j += 1) spline += edge.c[j] * b[j];
  return edge.wb * silu(t) + edge.ws * spline;
}

export function kanPredict(net: KanNet, x: number): number {
  const { b: b1 } = basisAndDeriv(x, net.knots1);
  let y = 0;
  for (let i = 0; i < net.width; i += 1) {
    const h = edgeValue(net.e1[i], x, b1);
    const { b: b2 } = basisAndDeriv(h, net.knots2);
    y += edgeValue(net.e2[i], h, b2);
  }
  return Number.isFinite(y) ? y : 0;
}

export function kanLoss(net: KanNet, points: Point[]): number {
  if (points.length === 0) return 0;
  let total = 0;
  for (const p of points) {
    const e = kanPredict(net, p.x) - p.y;
    total += e * e;
  }
  return total / points.length;
}

/** The learned function on each edge, for the diagram. */
export function kanEdgeCurves(net: KanNet, xs: number[]): { inner: number[][]; outer: number[][] } {
  const inner: number[][] = Array.from({ length: net.width }, () => []);
  const outer: number[][] = Array.from({ length: net.width }, () => []);
  for (const x of xs) {
    const { b: b1 } = basisAndDeriv(x, net.knots1);
    for (let i = 0; i < net.width; i += 1) {
      const h = edgeValue(net.e1[i], x, b1);
      inner[i].push(h);
      const { b: b2 } = basisAndDeriv(h, net.knots2);
      outer[i].push(edgeValue(net.e2[i], h, b2));
    }
  }
  return { inner, outer };
}

// ------------------------------------------------------------------ training

type EdgeGrad = { wb: number; ws: number; c: number[] };

function zeroGrads(net: KanNet): { g1: EdgeGrad[]; g2: EdgeGrad[] } {
  const make = () => ({ wb: 0, ws: 0, c: new Array<number>(basisCount()).fill(0) });
  return {
    g1: Array.from({ length: net.width }, make),
    g2: Array.from({ length: net.width }, make),
  };
}

function cloneKan(net: KanNet): KanNet {
  return {
    ...net,
    e1: net.e1.map((e) => ({ ...e, c: [...e.c] })),
    e2: net.e2.map((e) => ({ ...e, c: [...e.c] })),
  };
}

function clip(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(-GRAD_CLIP, Math.min(GRAD_CLIP, v));
}
function settle(next: number, prev: number): number {
  if (!Number.isFinite(next)) return prev;
  return Math.max(-PARAM_CLIP, Math.min(PARAM_CLIP, next));
}

function accumulate(
  net: KanNet,
  g: { g1: EdgeGrad[]; g2: EdgeGrad[] },
  point: Point,
  scale: number,
) {
  const x = point.x;
  const { b: b1 } = basisAndDeriv(x, net.knots1);
  const s1 = silu(x);

  const hs: number[] = [];
  const spl1: number[] = [];
  let y = 0;
  const b2s: number[][] = [];
  const db2s: number[][] = [];
  const spl2: number[] = [];

  for (let i = 0; i < net.width; i += 1) {
    const e = net.e1[i];
    let sp = 0;
    for (let j = 0; j < e.c.length; j += 1) sp += e.c[j] * b1[j];
    spl1.push(sp);
    const h = e.wb * s1 + e.ws * sp;
    hs.push(h);

    const { b, db } = basisAndDeriv(h, net.knots2);
    b2s.push(b);
    db2s.push(db);
    const o = net.e2[i];
    let sp2 = 0;
    for (let j = 0; j < o.c.length; j += 1) sp2 += o.c[j] * b[j];
    spl2.push(sp2);
    y += o.wb * silu(h) + o.ws * sp2;
  }

  const e0 = 2 * (y - point.y) * scale;

  for (let i = 0; i < net.width; i += 1) {
    const h = hs[i];
    const o = net.e2[i];
    g.g2[i].wb += e0 * silu(h);
    g.g2[i].ws += e0 * spl2[i];
    for (let j = 0; j < o.c.length; j += 1) g.g2[i].c[j] += e0 * o.ws * b2s[i][j];

    // dψ/dh, needed to reach the inner edge.
    let dSpline2 = 0;
    for (let j = 0; j < o.c.length; j += 1) dSpline2 += o.c[j] * db2s[i][j];
    const dh = e0 * (o.wb * dSilu(h) + o.ws * dSpline2);

    const inner = net.e1[i];
    g.g1[i].wb += dh * s1;
    g.g1[i].ws += dh * spl1[i];
    for (let j = 0; j < inner.c.length; j += 1) g.g1[i].c[j] += dh * inner.ws * b1[j];
  }
}

export type KanTrainOptions = { learningRate: number; batchSize: number };

/**
 * One pass over the training set. `order` is supplied by the caller so a KAN
 * sees the identical mini-batch sequence as every other model in a comparison.
 */
export function kanTrainEpoch(
  net: KanNet,
  train: Point[],
  opts: KanTrainOptions,
  order: number[],
): KanNet {
  if (train.length === 0) return net;
  const next = cloneKan(net);

  for (let start = 0; start < order.length; start += opts.batchSize) {
    const batch = order.slice(start, start + opts.batchSize);
    const g = zeroGrads(next);
    const scale = 1 / batch.length;
    for (const idx of batch) accumulate(next, g, train[idx], scale);

    const lr = opts.learningRate;
    const apply = (edges: KanEdge[], grads: EdgeGrad[]) => {
      edges.forEach((edge, i) => {
        edge.wb = settle(edge.wb - lr * clip(grads[i].wb), edge.wb);
        edge.ws = settle(edge.ws - lr * clip(grads[i].ws), edge.ws);
        for (let j = 0; j < edge.c.length; j += 1) {
          edge.c[j] = settle(edge.c[j] - lr * clip(grads[i].c[j]), edge.c[j]);
        }
      });
    };
    apply(next.e1, g.g1);
    apply(next.e2, g.g2);
  }

  return next;
}

const ERROR_XS = Array.from({ length: 401 }, (_, i) => -1 + i / 200);

export function kanFunctionMse(net: KanNet, target: TargetId): number {
  let total = 0;
  for (const x of ERROR_XS) {
    const e = kanPredict(net, x) - targetValue(target, x);
    total += e * e;
  }
  return total / ERROR_XS.length;
}

export function kanLocalMse(net: KanNet, target: TargetId, radius: number): number {
  let total = 0;
  let count = 0;
  for (const x of ERROR_XS) {
    if (Math.abs(x) > radius) continue;
    const e = kanPredict(net, x) - targetValue(target, x);
    total += e * e;
    count += 1;
  }
  return count === 0 ? NaN : total / count;
}

export function kanTestLoss(net: KanNet, data: Dataset): number {
  return kanLoss(net, data.test);
}

/**
 * What one hidden path contributes to the output.
 *
 * The output is a plain sum of the outer edge functions, so pinning edge `i` to
 * its own mean shifts F(x) by exactly ψ_i(h_i(x)) − mean(ψ_i). No ablation
 * re-run is needed — this is the contribution, exactly.
 */
export function kanEdgeContribution(
  net: KanNet,
  xs: number[],
  index: number,
): { own: number[]; delta: number[]; band: { lo: number; hi: number } | null; peak: number } {
  const own: number[] = [];
  const psi: number[] = [];
  for (const x of xs) {
    const { b: b1 } = basisAndDeriv(x, net.knots1);
    const h = edgeValue(net.e1[index], x, b1);
    own.push(h);
    const { b: b2 } = basisAndDeriv(h, net.knots2);
    psi.push(edgeValue(net.e2[index], h, b2));
  }
  const mean = psi.reduce((s, v) => s + v, 0) / (psi.length || 1);
  const delta = psi.map((v) => v - mean);

  const abs = delta.map(Math.abs);
  const peak = Math.max(...abs);
  let band: { lo: number; hi: number } | null = null;
  if (peak > 1e-9) {
    const centre = abs.indexOf(peak);
    let lo = centre;
    let hi = centre;
    while (lo > 0 && abs[lo - 1] >= peak * 0.5) lo -= 1;
    while (hi < abs.length - 1 && abs[hi + 1] >= peak * 0.5) hi += 1;
    band = { lo: xs[lo], hi: xs[hi] };
  }
  return { own, delta, band, peak };
}
