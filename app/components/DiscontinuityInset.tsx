"use client";

import type { PlotRun } from "./ApproximationPlot";
import { LOCAL_RADIUS, predict, targetValue, type TargetId } from "../lib/model";

const W = 340;
const H = 150;
const PAD = { left: 34, right: 12, top: 12, bottom: 24 };
const N = 161;
const XS = Array.from({ length: N }, (_, i) => -LOCAL_RADIUS + (2 * LOCAL_RADIUS * i) / (N - 1));

/**
 * Zoomed view of the jump. A global plot flattens the only region where the
 * activations actually differ on this target, so the interesting comparison is
 * invisible without it.
 */
export default function DiscontinuityInset({
  runs,
  target,
}: {
  runs: PlotRun[];
  target: TargetId;
}) {
  // Fixed range: the Heaviside spans 0 to 1, and a fitted axis would jump
  // around as the models move.
  const low = -0.35;
  const high = 1.35;

  const toX = (x: number) =>
    PAD.left + ((x + LOCAL_RADIUS) / (2 * LOCAL_RADIUS)) * (W - PAD.left - PAD.right);
  const toY = (y: number) => {
    const c = Math.max(low, Math.min(high, y));
    return PAD.top + ((high - c) / (high - low)) * (H - PAD.top - PAD.bottom);
  };

  const path = (fn: (x: number) => number) =>
    XS.map((x, i) => `${i === 0 ? "M" : "L"}${toX(x).toFixed(1)},${toY(fn(x)).toFixed(1)}`).join(" ");

  return (
    <div className="pde-chart">
      <span className="pde-chart-title">
        Discontinuity Benchmark — x ∈ [−{LOCAL_RADIUS}, {LOCAL_RADIUS}]
      </span>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Zoomed view of the discontinuity">
        <rect x="0" y="0" width={W} height={H} rx="8" className="plot-background" />
        <line x1={toX(0)} x2={toX(0)} y1={PAD.top} y2={H - PAD.bottom} className="grid" />
        {[0, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={toY(v)} y2={toY(v)} className="grid" />
            <text x={PAD.left - 6} y={toY(v) + 4} textAnchor="end" className="axis-label">
              {v}
            </text>
          </g>
        ))}
        <path d={path((x) => targetValue(target, x))} className="target-line" />
        {runs.map((run) => (
          <path
            key={run.activation}
            d={path((x) => predict(run.net, x, run.activation))}
            style={{ stroke: run.color }}
            className="model-line"
          />
        ))}
        <text x={toX(-LOCAL_RADIUS) + 2} y={H - 8} className="axis-label">
          −{LOCAL_RADIUS}
        </text>
        <text x={toX(LOCAL_RADIUS) - 2} y={H - 8} textAnchor="end" className="axis-label">
          {LOCAL_RADIUS}
        </text>
      </svg>
    </div>
  );
}
