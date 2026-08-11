"use client";

import { useMemo } from "react";

import { PLOT_XS, targetValue, type Point, type TargetId } from "../lib/model";

/** Rounds a coordinate to a stable number of decimals for SSR/client agreement. */
const round = (value: number) => Math.round(value * 10) / 10;

const WIDTH = 720;
const HEIGHT = 320;
const PAD = { left: 46, right: 18, top: 16, bottom: 34 };
const SAMPLES = PLOT_XS;
const X_TICKS = [-1, -0.5, 0, 0.5, 1];

/** A curve to draw. Carrying `predict` keeps the plot independent of which
 * model family produced it — MLP or KAN. */
export type PlotRun = { id: string; color: string; predict: (x: number) => number };

type Props = {
  runs: PlotRun[];
  target: TargetId;
  train: Point[];
  test: Point[];
  showTest: boolean;
  /** Selected neuron's influence on the output, plus where it matters most. */
  overlay: {
    delta: number[];
    band: { lo: number; hi: number } | null;
    /** Cauchy only: analytic centre μ and half-width d in input space. */
    localization: { mu: number; width: number } | null;
  } | null;
};

export default function ApproximationPlot({
  runs,
  target,
  train,
  test,
  showTest,
  overlay,
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

  // Independent of the network, so built once per dataset rather than per frame.
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
          <circle key={`tr-${i}`} cx={round(toX(p.x))} cy={round(toY(p.y))} r="2.6" className="point-train" />
        ))}
        {showTest &&
          test.map((p, i) => (
            <circle key={`te-${i}`} cx={round(toX(p.x))} cy={round(toY(p.y))} r="3" className="point-test" />
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

  return (
    <svg
      className="plot"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Target function, samples, and network predictions"
    >
      <rect x="0" y="0" width={WIDTH} height={HEIGHT} rx="12" className="plot-background" />

      {/* Where the selected neuron does most of its work, measured by ablation. */}
      {overlay?.band && (
        <rect
          x={round(toX(overlay.band.lo))}
          y={PAD.top}
          width={Math.max(2, round(toX(overlay.band.hi) - toX(overlay.band.lo)))}
          height={HEIGHT - PAD.top - PAD.bottom}
          className="influence-band"
        />
      )}

      {/* Analytic Cauchy localisation: centre μ and half-width d in input space. */}
      {overlay?.localization && (
        <g>
          <rect
            x={round(toX(overlay.localization.mu - overlay.localization.width))}
            y={PAD.top}
            width={Math.max(
              2,
              round(
                toX(overlay.localization.mu + overlay.localization.width) -
                  toX(overlay.localization.mu - overlay.localization.width),
              ),
            )}
            height={HEIGHT - PAD.top - PAD.bottom}
            className="localization-band"
          />
          <line
            x1={round(toX(overlay.localization.mu))}
            x2={round(toX(overlay.localization.mu))}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            className="localization-center"
          />
          <text
            x={round(toX(overlay.localization.mu))}
            y={PAD.top + 12}
            textAnchor="middle"
            className="localization-label"
          >
            μ
          </text>
        </g>
      )}

      {axes}
      {scatter}
      <path d={targetPath} className="target-line" />
      {/* While a neuron is under inspection the fits recede so its own
          contribution curve is readable against them. */}
      {runs.map((run) => (
        <path
          key={run.id}
          d={path(SAMPLES.map((x) => run.predict(x)))}
          style={{ stroke: run.color, opacity: overlay ? 0.28 : 1 }}
          className="model-line"
        />
      ))}
      {overlay && <path d={path(overlay.delta)} className="neuron-line" />}
    </svg>
  );
}
