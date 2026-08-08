// A deliberately small PINN prototype for the PDE demo.
//
// One hidden layer, a linear output, and the PDE residual written directly in
// terms of the analytic φ' and φ'' from `activations.ts`. Every problem here
// has a closed-form solution, so "absolute error" is a real number rather than
// a comparison against another approximation.
//
// Parameter gradients are taken by central finite differences. For a network
// this small that costs little, and it keeps the residual — which already
// involves second derivatives — from needing hand-derived third derivatives.
// It is a prototype optimiser, not a production one.

import {
  D_FLOOR,
  d2Phi,
  dPhi,
  phi,
  softplus,
  activationMeta,
  type ActivationId,
  type ShapeParams,
} from "./activations";

export type PdeId = "poisson" | "heat" | "burgers";

export type PdeMeta = {
  id: PdeId;
  label: string;
  /** Rendered above the charts. */
  equation: string;
  exactLabel: string;
  xMin: number;
  xMax: number;
  timeDependent: boolean;
  nu: number;
  note: string;
};

const NU_HEAT = 0.1;
const NU_BURGERS = 0.1;
const BURGERS_A = 0.5;
const BURGERS_C = 0.5;

export const PDES: PdeMeta[] = [
  {
    id: "poisson",
    label: "Poisson",
    equation: "u″(x) = f(x),  u(0) = u(1) = 0",
    exactLabel: "u = sin(πx)",
    xMin: 0,
    xMax: 1,
    timeDependent: false,
    nu: 0,
    note: "Second-order, steady state",
  },
  {
    id: "heat",
    label: "Heat",
    equation: "uₜ = ν uₓₓ,  ν = 0.1",
    exactLabel: "u = e^(−νπ²t)·sin(πx)",
    xMin: 0,
    xMax: 1,
    timeDependent: true,
    nu: NU_HEAT,
    note: "Diffusion, smooth decay",
  },
  {
    id: "burgers",
    label: "Burgers",
    equation: "uₜ + u·uₓ = ν uₓₓ,  ν = 0.1",
    exactLabel: "travelling wave",
    xMin: -1,
    xMax: 1,
    timeDependent: true,
    nu: NU_BURGERS,
    note: "Nonlinear advection–diffusion",
  },
];

export function pdeMeta(id: PdeId): PdeMeta {
  return PDES.find((p) => p.id === id) ?? PDES[0];
}

/** Closed-form solution for each problem. */
export function exactSolution(id: PdeId, x: number, t: number): number {
  switch (id) {
    case "poisson":
      return Math.sin(Math.PI * x);
    case "heat":
      return Math.exp(-NU_HEAT * Math.PI * Math.PI * t) * Math.sin(Math.PI * x);
    case "burgers": {
      const k = BURGERS_A / (2 * NU_BURGERS);
      return BURGERS_C - BURGERS_A * Math.tanh(k * (x - BURGERS_C * t));
    }
  }
}

/** Right-hand side of the Poisson problem, chosen to match `exactSolution`. */
function poissonSource(x: number): number {
  return -Math.PI * Math.PI * Math.sin(Math.PI * x);
}

// ------------------------------------------------------------------ network

/**
 * A flat parameter vector, which is what makes finite-difference gradients
 * straightforward. Layout per hidden neuron: weights, bias, output weight, and
 * for Cauchy the three shape parameters. The output bias is the final slot.
 */
export type PdeNet = {
  activation: ActivationId;
  nIn: number;
  nHidden: number;
  stride: number;
  params: Float64Array;
  /** Adam moments and step count. */
  m: Float64Array;
  v: Float64Array;
  steps: number;
};

function strideFor(activation: ActivationId, nIn: number): number {
  return nIn + 2 + activationMeta(activation).shapeParams; // w..., b, a, [l1, l2, dRaw]
}

export function pdeParameterCount(activation: ActivationId, nIn: number, nHidden: number): number {
  return strideFor(activation, nIn) * nHidden + 1;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makePdeNet(
  activation: ActivationId,
  nIn: number,
  nHidden: number,
  seed: number,
): PdeNet {
  const stride = strideFor(activation, nIn);
  const size = stride * nHidden + 1;
  const params = new Float64Array(size);
  const rng = mulberry32(seed);

  for (let j = 0; j < nHidden; j += 1) {
    const o = j * stride;
    for (let i = 0; i < nIn; i += 1) params[o + i] = (rng() * 2 - 1) * 2.2;
    params[o + nIn] = (rng() * 2 - 1) * 1.2; // bias
    params[o + nIn + 1] = (rng() * 2 - 1) * 0.8; // output weight
    if (activation === "cauchy") {
      params[o + nIn + 2] = 1 + (rng() * 2 - 1) * 0.3; // l1
      params[o + nIn + 3] = (rng() * 2 - 1) * 0.3; // l2
      params[o + nIn + 4] = -0.2 + (rng() * 2 - 1) * 0.3; // dRaw
    }
  }
  params[size - 1] = 0;

  return {
    activation,
    nIn,
    nHidden,
    stride,
    params,
    m: new Float64Array(size),
    v: new Float64Array(size),
    steps: 0,
  };
}

// This is the hot loop: a finite-difference gradient evaluates the whole
// collocation set twice per parameter. Everything below is written to allocate
// nothing per call — one shared scratch object for the shape parameters, and
// scalars instead of arrays for the derivatives (there are at most two inputs).
const SCRATCH: ShapeParams = { l1: 1, l2: 0, d: 1 };
const OUT = { u: 0, ux: 0, ut: 0, uxx: 0 };

/** Fills `OUT` with u and the derivatives the residuals need. Allocation-free. */
function evaluate(net: PdeNet, p: Float64Array, x: number, t: number): typeof OUT {
  const { nIn, nHidden, stride, activation } = net;
  const cauchy = activation === "cauchy";
  let u = p[p.length - 1];
  let ux = 0;
  let ut = 0;
  let uxx = 0;

  for (let j = 0; j < nHidden; j += 1) {
    const o = j * stride;
    const wx = p[o];
    const wt = nIn > 1 ? p[o + 1] : 0;
    const z = wx * x + wt * t + p[o + nIn];
    const a = p[o + nIn + 1];

    if (cauchy) {
      SCRATCH.l1 = p[o + nIn + 2];
      SCRATCH.l2 = p[o + nIn + 3];
      SCRATCH.d = softplus(p[o + nIn + 4]) + D_FLOOR;
    }

    u += a * phi(activation, z, SCRATCH);
    const d1 = dPhi(activation, z, SCRATCH);
    const d2 = d2Phi(activation, z, SCRATCH);
    ux += a * wx * d1;
    uxx += a * wx * wx * d2;
    if (nIn > 1) ut += a * wt * d1;
  }

  OUT.u = u;
  OUT.ux = ux;
  OUT.ut = ut;
  OUT.uxx = uxx;
  return OUT;
}

export function pdePredict(net: PdeNet, x: number, t: number): number {
  return evaluate(net, net.params, x, t).u;
}

/** Signed PDE residual at a point — zero for an exact solution. */
export function pdeResidual(net: PdeNet, id: PdeId, x: number, t: number): number {
  return residualWith(net, net.params, id, x, t);
}

function residualWith(
  net: PdeNet,
  p: Float64Array,
  id: PdeId,
  x: number,
  t: number,
): number {
  const r = evaluate(net, p, x, t);
  if (id === "poisson") return r.uxx - poissonSource(x);
  if (id === "heat") return r.ut - NU_HEAT * r.uxx;
  return r.ut + r.u * r.ux - NU_BURGERS * r.uxx;
}

// ---------------------------------------------------------------- collocation

export type PdeSamples = {
  /** Interior points where the residual is enforced. */
  interior: { x: number; t: number }[];
  /** Boundary and initial points where the value is pinned. */
  anchors: { x: number; t: number; u: number }[];
};

export function makeSamples(id: PdeId): PdeSamples {
  const meta = pdeMeta(id);
  const interior: { x: number; t: number }[] = [];
  const anchors: { x: number; t: number; u: number }[] = [];
  const span = meta.xMax - meta.xMin;

  if (!meta.timeDependent) {
    const N = 64;
    for (let i = 1; i < N; i += 1) interior.push({ x: meta.xMin + (i / N) * span, t: 0 });
    anchors.push({ x: meta.xMin, t: 0, u: exactSolution(id, meta.xMin, 0) });
    anchors.push({ x: meta.xMax, t: 0, u: exactSolution(id, meta.xMax, 0) });
    return { interior, anchors };
  }

  // Kept small on purpose: a finite-difference gradient touches every point
  // twice per parameter, and three networks train side by side each frame.
  const NX = 16;
  const NT = 8;
  for (let i = 1; i < NX; i += 1) {
    for (let k = 1; k <= NT; k += 1) {
      interior.push({ x: meta.xMin + (i / NX) * span, t: k / NT });
    }
  }
  // Initial condition plus both spatial boundaries.
  for (let i = 0; i <= NX; i += 1) {
    const x = meta.xMin + (i / NX) * span;
    anchors.push({ x, t: 0, u: exactSolution(id, x, 0) });
  }
  for (let k = 1; k <= NT; k += 1) {
    const t = k / NT;
    anchors.push({ x: meta.xMin, t, u: exactSolution(id, meta.xMin, t) });
    anchors.push({ x: meta.xMax, t, u: exactSolution(id, meta.xMax, t) });
  }
  return { interior, anchors };
}

const ANCHOR_WEIGHT = 10;

function totalLoss(net: PdeNet, p: Float64Array, id: PdeId, s: PdeSamples): number {
  let res = 0;
  for (const q of s.interior) {
    const r = residualWith(net, p, id, q.x, q.t);
    res += r * r;
  }
  res /= Math.max(1, s.interior.length);

  let bc = 0;
  for (const q of s.anchors) {
    const e = evaluate(net, p, q.x, q.t).u - q.u;
    bc += e * e;
  }
  bc /= Math.max(1, s.anchors.length);

  const value = res + ANCHOR_WEIGHT * bc;
  return Number.isFinite(value) ? value : 1e6;
}

export function pdeLoss(net: PdeNet, id: PdeId, s: PdeSamples): number {
  return totalLoss(net, net.params, id, s);
}

/** Mean |u_pred − u_exact| over a grid, the headline accuracy number. */
export function pdeAbsError(net: PdeNet, id: PdeId): number {
  const meta = pdeMeta(id);
  let total = 0;
  let count = 0;
  const times = meta.timeDependent ? [0, 0.25, 0.5, 0.75, 1] : [0];
  for (const t of times) {
    for (let i = 0; i <= 40; i += 1) {
      const x = meta.xMin + (i / 40) * (meta.xMax - meta.xMin);
      total += Math.abs(pdePredict(net, x, t) - exactSolution(id, x, t));
      count += 1;
    }
  }
  return total / count;
}

// ------------------------------------------------------------------ training

const FD_H = 1e-4;
const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-8;

/**
 * One Adam step on finite-difference gradients. Returns a new net; the caller
 * keeps the previous one so a diverged run can be discarded.
 */
export function pdeStep(net: PdeNet, id: PdeId, s: PdeSamples, lr: number): PdeNet {
  const p = Float64Array.from(net.params);
  const m = Float64Array.from(net.m);
  const v = Float64Array.from(net.v);
  const steps = net.steps + 1;

  const grad = new Float64Array(p.length);
  const probe = Float64Array.from(p);
  for (let i = 0; i < p.length; i += 1) {
    const original = probe[i];
    probe[i] = original + FD_H;
    const up = totalLoss(net, probe, id, s);
    probe[i] = original - FD_H;
    const down = totalLoss(net, probe, id, s);
    probe[i] = original;
    const g = (up - down) / (2 * FD_H);
    grad[i] = Number.isFinite(g) ? Math.max(-1e4, Math.min(1e4, g)) : 0;
  }

  const bc1 = 1 - Math.pow(ADAM_B1, steps);
  const bc2 = 1 - Math.pow(ADAM_B2, steps);
  for (let i = 0; i < p.length; i += 1) {
    m[i] = ADAM_B1 * m[i] + (1 - ADAM_B1) * grad[i];
    v[i] = ADAM_B2 * v[i] + (1 - ADAM_B2) * grad[i] * grad[i];
    const step = (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + ADAM_EPS);
    const next = p[i] - step;
    p[i] = Number.isFinite(next) ? Math.max(-60, Math.min(60, next)) : p[i];
  }

  return { ...net, params: p, m, v, steps };
}
