// Activation functions, each with analytic first and second derivatives.
//
// φ'  drives the "Activation Shape" chart and ordinary backpropagation.
// φ'' is what the PDE demo needs: a PINN residual for a second-order equation
//     is written directly in terms of it.
//
// Only the Cauchy family carries trainable shape parameters. Everything else
// has a fixed shape, which is precisely the comparison this playground exists
// to make visible.

export type ActivationId =
  | "cauchy"
  | "relu"
  | "tanh"
  | "sigmoid"
  | "gelu"
  | "silu"
  | "sine";

/** Cauchy shape parameters: φ(z) = (λ₁z + λ₂) / (z² + d²). */
export type ShapeParams = { l1: number; l2: number; d: number };

/** Smallest half-width a Cauchy unit may take, so the denominator never vanishes. */
export const D_FLOOR = 0.08;

export const DEFAULT_SHAPE: ShapeParams = { l1: 1, l2: 0, d: 0.5 };

export type ActivationMeta = {
  id: ActivationId;
  label: string;
  /** Primary options get a segmented button; the rest live in the dropdown. */
  primary: boolean;
  /** Trainable shape parameters per neuron, on top of weights and bias. */
  shapeParams: number;
  note: string;
};

export const ACTIVATIONS: ActivationMeta[] = [
  { id: "cauchy", label: "Cauchy", primary: true, shapeParams: 3, note: "Trainable, localized, smooth" },
  { id: "relu", label: "ReLU", primary: true, shapeParams: 0, note: "Fixed kink, φ'' = 0" },
  { id: "tanh", label: "Tanh", primary: true, shapeParams: 0, note: "Fixed, saturating" },
  { id: "sigmoid", label: "Sigmoid", primary: false, shapeParams: 0, note: "Fixed, saturating" },
  { id: "gelu", label: "GELU", primary: false, shapeParams: 0, note: "Fixed, smooth" },
  { id: "silu", label: "SiLU / Swish", primary: false, shapeParams: 0, note: "Fixed, smooth" },
  { id: "sine", label: "Sine", primary: false, shapeParams: 0, note: "Fixed, periodic" },
];

export const ACTIVATION_IDS = ACTIVATIONS.map((a) => a.id);

export function activationMeta(id: ActivationId): ActivationMeta {
  return ACTIVATIONS.find((a) => a.id === id) ?? ACTIVATIONS[0];
}

/** Series colour: orange marks Cauchy everywhere, blue is the standard network. */
export const ACTIVATION_COLORS: Record<ActivationId, string> = {
  cauchy: "#f59322",
  relu: "#0877bd",
  tanh: "#12a594",
  sigmoid: "#8b5cf6",
  gelu: "#e11d48",
  silu: "#0891b2",
  sine: "#65a30d",
};

// ------------------------------------------------------------------ helpers

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function softplus(x: number): number {
  return x > 18 ? x : Math.log1p(Math.exp(x));
}

/** Abramowitz & Stegun 7.1.26 — plenty accurate for drawing and for GELU. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-a * a);
  return sign * y;
}

/** Standard normal CDF and PDF, used by GELU. */
const normalCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));
const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

// ------------------------------------------------------------- φ, φ' and φ''

export function phi(id: ActivationId, z: number, p: ShapeParams): number {
  switch (id) {
    case "relu":
      return Math.max(0, z);
    case "tanh":
      return Math.tanh(z);
    case "sigmoid":
      return sigmoid(z);
    case "gelu":
      return z * normalCdf(z);
    case "silu":
      return z * sigmoid(z);
    case "sine":
      return Math.sin(z);
    case "cauchy":
      return (p.l1 * z + p.l2) / (z * z + p.d * p.d);
  }
}

export function dPhi(id: ActivationId, z: number, p: ShapeParams): number {
  switch (id) {
    case "relu":
      return z > 0 ? 1 : 0;
    case "tanh": {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    case "sigmoid": {
      const s = sigmoid(z);
      return s * (1 - s);
    }
    case "gelu":
      return normalCdf(z) + z * normalPdf(z);
    case "silu": {
      const s = sigmoid(z);
      return s * (1 + z * (1 - s));
    }
    case "sine":
      return Math.cos(z);
    case "cauchy": {
      const den = z * z + p.d * p.d;
      const num = p.l1 * z + p.l2;
      return (p.l1 * den - 2 * z * num) / (den * den);
    }
  }
}

export function d2Phi(id: ActivationId, z: number, p: ShapeParams): number {
  switch (id) {
    case "relu":
      // Zero almost everywhere. This is exactly why a ReLU PINN cannot solve a
      // second-order PDE, and the demo is meant to show that.
      return 0;
    case "tanh": {
      const t = Math.tanh(z);
      return -2 * t * (1 - t * t);
    }
    case "sigmoid": {
      const s = sigmoid(z);
      return s * (1 - s) * (1 - 2 * s);
    }
    case "gelu":
      return normalPdf(z) * (2 - z * z);
    case "silu": {
      const s = sigmoid(z);
      const ds = s * (1 - s);
      return 2 * ds + z * ds * (1 - 2 * s);
    }
    case "sine":
      return -Math.sin(z);
    case "cauchy": {
      const den = z * z + p.d * p.d;
      // φ' = u / den²  with  u = -λ₁z² - 2λ₂z + λ₁d²
      const u = -p.l1 * z * z - 2 * p.l2 * z + p.l1 * p.d * p.d;
      const du = -2 * p.l1 * z - 2 * p.l2;
      return (du * den - 4 * z * u) / (den * den * den);
    }
  }
}

/**
 * ∂φ/∂(shape parameters). Zero for every fixed-shape activation, which is the
 * whole point: those networks can only move weights, not reshape neurons.
 */
export function shapeGradient(
  id: ActivationId,
  z: number,
  p: ShapeParams,
): ShapeParams {
  if (id !== "cauchy") return { l1: 0, l2: 0, d: 0 };
  const den = z * z + p.d * p.d;
  const num = p.l1 * z + p.l2;
  return {
    l1: z / den,
    l2: 1 / den,
    d: (-2 * p.d * num) / (den * den),
  };
}
