"use client";

import { useMemo } from "react";

import {
  PLOT_XS,
  predict,
  targetValue,
  type Activation,
  type Network,
  type Point,
  type TargetId,
} from "../lib/model";

/** Rounds a coordinate to a stable number of decimals for SSR/client agreement. */
const round = (value: number) => Math.round(value * 10) / 10;

const WIDTH = 720;
const HEIGHT = 340;
const PAD = { left: 46, right: 18, top: 16, bottom: 34 };
const SAMPLES = PLOT_XS;
const X_TICKS = [-1, -0.5, 0, 0.5, 1];

type Props = {
  net: Network;
  activation: Activation;
  target: TargetId;
  train: Point[];
  test: Point[];
  showTest: boolean;
  hoveredCurve: number[] | null;
};

export default function ApproximationPlot({
  net,
  activation,
  target,
  train,
  test,
  showTest,
  hoveredCurve,
}: Props) {
  // The vertical range deliberately ignores the model: a diverging network would
  // otherwise rescale the axes every frame and make the plot unreadable.
  const { low, high } = useMemo(() => {
    const values = [
      ...SAMPLES.map((x) => targetValue(target, x)),
      ...train.map((p) => p.y),
      ...test.map((p) => p.y),
    ].filter(Number.isFinite);
    return {
      low: Math.min(-1.1, ...values) - 0.15,
      high: Math.max(1.1, ...values) + 0.15,
    };
  }, [target, train, test]);

  const toX = (x: number) => PAD.left + ((x + 1) / 2) * (WIDTH - PAD.left - PAD.right);
  const toY = (y: number) => {
    const clamped = Math.max(low, Math.min(high, y));
    return PAD.top + ((high - clamped) / (high - low)) * (HEIGHT - PAD.top - PAD.bottom);
  };

  const path = (values: number[]) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${toX(SAMPLES[i]).toFixed(1)},${toY(v).toFixed(1)}`)
      .join(" ");

  // Everything below is independent of the network, so it is built once per
  // dataset instead of on every training frame. Stable element identity lets
  // React bail out of diffing these subtrees entirely.
  const axes = useMemo(() => {
    const yTicks = [low, (low + high) / 2, high];
    return (
      <g>
        {X_TICKS.map((tick) => (
          <g key={tick}>
            <line x1={toX(tick)} x2={toX(tick)} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="grid" />
            <text x={toX(tick)} y={HEIGHT - 12} textAnchor="middle" className="axis-label">
              {tick}
            </text>
          </g>
        ))}
        {yTicks.map((tick, i) => (
          <g key={i}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={round(toY(tick))} y2={round(toY(tick))} className="grid" />
            <text x={PAD.left - 8} y={round(toY(tick)) + 4} textAnchor="end" className="axis-label">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={round(toY(0))} y2={round(toY(0))} className="zero-line" />
      </g>
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [low, high]);

  const scatter = useMemo(
    () => (
      <g>
        {train.map((p, i) => (
          <circle key={`tr-${i}`} cx={round(toX(p.x))} cy={round(toY(p.y))} r="3" className="point-train" />
        ))}
        {showTest &&
          test.map((p, i) => (
            <circle key={`te-${i}`} cx={round(toX(p.x))} cy={round(toY(p.y))} r="3.4" className="point-test" />
          ))}
      </g>
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [train, test, showTest, low, high],
  );

  const targetPath = useMemo(
    () => path(SAMPLES.map((x) => targetValue(target, x))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [target, low, high],
  );

  const modelPath = path(SAMPLES.map((x) => predict(net, x, activation)));

  return (
    <svg
      className="plot"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Target function, sampled data, and network prediction"
    >
      <rect x="0" y="0" width={WIDTH} height={HEIGHT} rx="12" className="plot-background" />
      {axes}
      {scatter}
      <path d={targetPath} className="target-line" />
      {hoveredCurve && <path d={path(hoveredCurve)} className="neuron-line" />}
      <path d={modelPath} className="model-line" />

      <g transform={`translate(${WIDTH - 170},26)`}>
        <line x1="0" x2="24" y1="0" y2="0" className="target-line" />
        <text x="32" y="4" className="legend-label">target</text>
        <line x1="0" x2="24" y1="19" y2="19" className="model-line" />
        <text x="32" y="23" className="legend-label">network</text>
        {hoveredCurve && (
          <>
            <line x1="0" x2="24" y1="38" y2="38" className="neuron-line" />
            <text x="32" y="42" className="legend-label">neuron</text>
          </>
        )}
      </g>
    </svg>
  );
}
