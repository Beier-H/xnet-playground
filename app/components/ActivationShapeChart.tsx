"use client";

import {
  ACTIVATION_COLORS,
  dPhi,
  phi,
  type ActivationId,
  type ShapeParams,
} from "../lib/activations";

const W = 240;
const H = 120;
const PAD = 10;
const Z_MAX = 4;
/** Fixed y range, clipped. Auto-ranging would make the curve jump as sliders move. */
const Y_MAX = 2.5;
const ZS = Array.from({ length: 121 }, (_, i) => -Z_MAX + (i * 2 * Z_MAX) / 120);

type Props = {
  activation: ActivationId;
  params: ShapeParams;
  showDerivative: boolean;
  /** Set when a specific neuron is selected, so the chart shows its real shape. */
  caption?: string;
};

export default function ActivationShapeChart({
  activation,
  params,
  showDerivative,
  caption,
}: Props) {
  const toX = (z: number) => PAD + ((z + Z_MAX) / (2 * Z_MAX)) * (W - PAD * 2);
  const toY = (y: number) => {
    const c = Math.max(-Y_MAX, Math.min(Y_MAX, y));
    return PAD + ((Y_MAX - c) / (2 * Y_MAX)) * (H - PAD * 2);
  };

  const path = (fn: (z: number) => number) =>
    ZS.map((z, i) => {
      const v = fn(z);
      const y = Number.isFinite(v) ? v : 0;
      return `${i === 0 ? "M" : "L"}${toX(z).toFixed(1)},${toY(y).toFixed(1)}`;
    }).join(" ");

  const color = ACTIVATION_COLORS[activation];

  return (
    <div className="shape-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Activation shape">
        <rect x="0" y="0" width={W} height={H} rx="8" className="plot-background" />
        <line x1={PAD} x2={W - PAD} y1={toY(0)} y2={toY(0)} className="zero-line" />
        <line x1={toX(0)} x2={toX(0)} y1={PAD} y2={H - PAD} className="grid" />
        {showDerivative && <path d={path((z) => dPhi(activation, z, params))} className="shape-deriv" />}
        <path d={path((z) => phi(activation, z, params))} style={{ stroke: color }} className="shape-line" />
        <text x={W - PAD} y={H - 3} textAnchor="end" className="axis-label">
          z
        </text>
      </svg>
      <div className="shape-legend">
        <span style={{ color }}>— φ(z)</span>
        {showDerivative && <span className="deriv-key">-- φ′(z)</span>}
        {caption && <span className="shape-caption">{caption}</span>}
      </div>
    </div>
  );
}
