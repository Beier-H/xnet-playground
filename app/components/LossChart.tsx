"use client";

export type LossPoint = { train: number; test: number };

/**
 * Epochs held in the chart. The x axis is pinned to this window rather than to
 * `history.length`: dividing by the current length would rescale the whole axis
 * on every epoch, so each new point would drag every existing point sideways
 * and the curve would visibly crawl. With a fixed window the curve grows
 * rightwards and then scrolls, and drawn points stay put.
 */
export const HISTORY_LIMIT = 220;

// A fixed logarithmic domain. Normalising against the window maximum instead
// would amplify ordinary mini-batch jitter to fill the whole chart as soon as
// the loss got small, and would make two runs impossible to compare.
const LOSS_MIN = 1e-4;
const LOSS_MAX = 10;
const LOG_MIN = Math.log10(LOSS_MIN);
const LOG_SPAN = Math.log10(LOSS_MAX) - LOG_MIN;

/** Decade gridlines, so the reader can see which order of magnitude they are at. */
const DECADES = [1, 0.1, 0.01, 0.001];

function toY(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 4;
  const t = (Math.log10(value) - LOG_MIN) / LOG_SPAN;
  return 96 - Math.max(0, Math.min(1, t)) * 92;
}

export default function LossChart({ history }: { history: LossPoint[] }) {
  const toPoints = (pick: (h: LossPoint) => number) =>
    history
      .map((h, i) => {
        const x = (i / (HISTORY_LIMIT - 1)) * 100;
        return `${x.toFixed(2)},${toY(pick(h)).toFixed(2)}`;
      })
      .join(" ");

  return (
    <svg
      className="loss-chart"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      role="img"
      aria-label="Training and test loss per epoch, log scale"
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
      <polyline points={toPoints((h) => h.test)} className="loss-test" vectorEffect="non-scaling-stroke" />
      <polyline points={toPoints((h) => h.train)} className="loss-train" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
