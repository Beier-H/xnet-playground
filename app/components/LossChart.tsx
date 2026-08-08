"use client";

export type LossPoint = { train: number; test: number };

export type LossSeries = {
  id: string;
  color: string;
  values: number[];
  dashed?: boolean;
};

/**
 * Epochs held in the chart. The x axis is pinned to this window rather than to
 * the series length: dividing by the current length would rescale the whole
 * axis on every epoch, so each new point would drag every existing point
 * sideways and the curve would visibly crawl.
 */
export const HISTORY_LIMIT = 220;

// A fixed logarithmic domain. Normalising against the window maximum instead
// would amplify ordinary mini-batch jitter to fill the whole chart as soon as
// the loss got small, and would make two runs impossible to compare.
const LOSS_MIN = 1e-5;
const LOSS_MAX = 10;
const LOG_MIN = Math.log10(LOSS_MIN);
const LOG_SPAN = Math.log10(LOSS_MAX) - LOG_MIN;

const DECADES = [1, 0.1, 0.01, 0.001, 0.0001];

function toY(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 4;
  const t = (Math.log10(value) - LOG_MIN) / LOG_SPAN;
  return 96 - Math.max(0, Math.min(1, t)) * 92;
}

export default function LossChart({ series }: { series: LossSeries[] }) {
  const points = (values: number[]) =>
    values
      .map((v, i) => `${((i / (HISTORY_LIMIT - 1)) * 100).toFixed(2)},${toY(v).toFixed(2)}`)
      .join(" ");

  return (
    <svg
      className="loss-chart"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      role="img"
      aria-label="Loss per epoch, log scale"
    >
      {DECADES.map((decade) => (
        <line
          key={decade}
          x1="0"
          x2="100"
          y1={toY(decade)}
          y2={toY(decade)}
          className="loss-grid"
          vectorEffect="non-scaling-stroke"
        />
      ))}
      {series.map((s) => (
        <polyline
          key={s.id}
          points={points(s.values)}
          fill="none"
          stroke={s.color}
          strokeWidth="1.6"
          strokeDasharray={s.dashed ? "3 2" : undefined}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}
