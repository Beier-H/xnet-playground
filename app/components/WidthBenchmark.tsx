"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ACTIVATION_COLORS, activationMeta, type ActivationId } from "../lib/activations";
import {
  BENCHMARK_EPOCHS,
  BENCHMARK_WIDTHS,
  COMPARE_SET,
  ERROR_TARGETS,
  LEARNING_RATES,
  advanceBenchmarkJob,
  createBenchmarkJobs,
  makeEpochOrders,
  makeDataset,
  type BenchmarkJob,
  type Regularization,
  type TargetId,
} from "../lib/model";

const W = 620;
const H = 300;
const PAD = { left: 58, right: 18, top: 16, bottom: 40 };

// Fixed log domain, like every other chart here: the sweep fills in point by
// point, and an axis fitted to the data so far would move everything already
// drawn on each update.
const Y_MIN = 1e-7;
const Y_MAX = 10;
const LOG_MIN = Math.log10(Y_MIN);
const LOG_SPAN = Math.log10(Y_MAX) - LOG_MIN;
const DECADES = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];

/**
 * Upper bound on epochs per frame; the runner also stops on a wall-clock budget,
 * so wide networks yield sooner than narrow ones.
 */
const SLICE = 40;
const FRAME_BUDGET_MS = 16;

type Props = {
  target: TargetId;
  learningRate: number;
  batchSize: number;
  regularization: Regularization;
  regRate: number;
  noise: number;
  percentTrain: number;
  dataSeed: number;
  netSeed: number;
  errorTarget: number;
  init: { l1: number; l2: number; d: number };
  onLearningRate: (lr: number) => void;
  onErrorTarget: (v: number) => void;
};

export default function WidthBenchmark(props: Props) {
  const {
    target,
    learningRate,
    batchSize,
    regularization,
    regRate,
    noise,
    percentTrain,
    dataSeed,
    netSeed,
    errorTarget,
    init,
  } = props;

  const [budget, setBudget] = useState(BENCHMARK_EPOCHS[0]);
  const [jobs, setJobs] = useState<BenchmarkJob[]>([]);
  const [running, setRunning] = useState(false);
  const [cursor, setCursor] = useState(0);

  const jobsRef = useRef<BenchmarkJob[]>([]);
  const cursorRef = useRef(0);
  const ordersRef = useRef<number[][]>([]);
  const datasetRef = useRef(makeDataset(target, noise, percentTrain, dataSeed));

  const totalEpochs = jobs.length * budget;
  const doneEpochs = jobs.reduce((sum, j) => sum + j.epochsDone, 0);
  const progress = totalEpochs === 0 ? 0 : doneEpochs / totalEpochs;

  const start = useCallback(() => {
    const dataset = makeDataset(target, noise, percentTrain, dataSeed);
    datasetRef.current = dataset;
    // One order per epoch, shared by every width and activation.
    ordersRef.current = makeEpochOrders(budget, dataset.train.length, 20250808);
    const fresh = createBenchmarkJobs(BENCHMARK_WIDTHS, COMPARE_SET, netSeed, init);
    jobsRef.current = fresh;
    cursorRef.current = 0;
    setJobs(fresh);
    setCursor(0);
    setRunning(true);
  }, [target, noise, percentTrain, dataSeed, netSeed, init, budget]);

  const stop = useCallback(() => setRunning(false), []);

  // The sweep advances one time-budgeted slice per task rather than in a single
  // blocking loop, so the browser keeps painting and the progress bar moves.
  //
  // Scheduled with a timer rather than an animation frame on purpose: a sweep is
  // something you start and then look away from, and requestAnimationFrame stops
  // entirely in a background tab, which would silently freeze it at 0%.
  useEffect(() => {
    if (!running) return;
    let timer = 0;
    const tick = () => {
      const list = jobsRef.current;
      let index = cursorRef.current;
      if (index >= list.length) {
        setRunning(false);
        return;
      }

      const advanced = advanceBenchmarkJob(
        list[index],
        datasetRef.current,
        target,
        { learningRate, batchSize, regularization, regRate },
        ordersRef.current,
        SLICE,
        errorTarget,
        FRAME_BUDGET_MS,
      );
      const next = [...list];
      next[index] = advanced;
      if (advanced.done) index += 1;

      jobsRef.current = next;
      cursorRef.current = index;
      setJobs(next);
      setCursor(index);

      if (index >= next.length) {
        setRunning(false);
        return;
      }
      timer = window.setTimeout(tick, 0);
    };
    timer = window.setTimeout(tick, 0);
    return () => window.clearTimeout(timer);
  }, [running, target, learningRate, batchSize, regularization, regRate, errorTarget]);

  const widths = BENCHMARK_WIDTHS;
  const toX = (width: number) =>
    PAD.left +
    ((Math.log2(width) - Math.log2(widths[0])) /
      (Math.log2(widths[widths.length - 1]) - Math.log2(widths[0]))) *
      (W - PAD.left - PAD.right);
  const toY = (v: number) => {
    if (!Number.isFinite(v) || v <= 0) return PAD.top;
    const t = (Math.log10(v) - LOG_MIN) / LOG_SPAN;
    return PAD.top + (1 - Math.max(0, Math.min(1, t))) * (H - PAD.top - PAD.bottom);
  };

  const seriesFor = (activation: ActivationId) =>
    jobs
      .filter((j) => j.activation === activation && j.done)
      .sort((a, b) => a.width - b.width);

  const hasAny = jobs.some((j) => j.done);

  return (
    <div className="bench">
      <section className="control-bar" aria-label="Benchmark controls">
        <button type="button" className="play-button wide" onClick={running ? stop : start}>
          {running ? "❚❚" : "▶"}
        </button>
        <div className="readout">
          <span>Sweep</span>
          <strong>{Math.round(progress * 100)}%</strong>
        </div>
        <label className="field">
          Epoch budget
          <select
            value={budget}
            disabled={running}
            onChange={(e) => setBudget(Number(e.target.value))}
          >
            {BENCHMARK_EPOCHS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Learning rate
          <select
            value={learningRate}
            disabled={running}
            onChange={(e) => props.onLearningRate(Number(e.target.value))}
          >
            {LEARNING_RATES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Target error
          <select
            value={errorTarget}
            disabled={running}
            onChange={(e) => props.onErrorTarget(Number(e.target.value))}
          >
            {ERROR_TARGETS.map((t) => (
              <option key={t} value={t}>
                {t.toExponential(0)}
              </option>
            ))}
          </select>
        </label>
        <div className="bench-status">
          <span className="badge live">Live Result</span>
          {running && jobs[cursor] && (
            <span className="muted">
              training {activationMeta(jobs[cursor].activation).label} · width {jobs[cursor].width}
            </span>
          )}
        </div>
      </section>

      <div className="bench-body">
        <div className="bench-progress">
          <div className="bar" style={{ width: `${progress * 100}%` }} />
        </div>

        <div className="bench-chart">
          <span className="pde-chart-title">
            Test MSE vs hidden neurons — identical data, seed, optimiser and batch order
          </span>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Test MSE against network width">
            <rect x="0" y="0" width={W} height={H} rx="10" className="plot-background" />
            {DECADES.map((d) => (
              <g key={d}>
                <line x1={PAD.left} x2={W - PAD.right} y1={toY(d)} y2={toY(d)} className="grid" />
                <text x={PAD.left - 8} y={toY(d) + 4} textAnchor="end" className="axis-label">
                  {d.toExponential(0)}
                </text>
              </g>
            ))}
            {widths.map((w) => (
              <g key={w}>
                <line
                  x1={toX(w)}
                  x2={toX(w)}
                  y1={PAD.top}
                  y2={H - PAD.bottom}
                  className="grid"
                />
                <text x={toX(w)} y={H - PAD.bottom + 18} textAnchor="middle" className="axis-label">
                  {w}
                </text>
              </g>
            ))}
            <text x={W / 2} y={H - 6} textAnchor="middle" className="axis-label">
              hidden neurons (log₂)
            </text>

            {COMPARE_SET.map((activation) => {
              const pts = seriesFor(activation);
              if (pts.length === 0) return null;
              return (
                <g key={activation}>
                  <path
                    d={pts
                      .map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.width).toFixed(1)},${toY(p.testMse).toFixed(1)}`)
                      .join(" ")}
                    fill="none"
                    stroke={ACTIVATION_COLORS[activation]}
                    strokeWidth="2.4"
                  />
                  {pts.map((p) => (
                    <circle
                      key={p.width}
                      cx={toX(p.width)}
                      cy={toY(p.testMse)}
                      r="3.4"
                      fill={ACTIVATION_COLORS[activation]}
                    />
                  ))}
                </g>
              );
            })}

            <g transform={`translate(${W - 130},26)`}>
              {COMPARE_SET.map((a, i) => (
                <g key={a} transform={`translate(0,${i * 16})`}>
                  <line x1="0" x2="18" y1="0" y2="0" stroke={ACTIVATION_COLORS[a]} strokeWidth="2.4" />
                  <text x="24" y="4" className="legend-label">
                    {activationMeta(a).label}
                  </text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        {hasAny && (
          <table className="bench-table">
            <thead>
              <tr>
                <th>Width</th>
                <th>Activation</th>
                <th>Train MSE</th>
                <th>Test MSE</th>
                <th>Fn MSE</th>
                <th>Ep→{errorTarget.toExponential(0)}</th>
                <th>Time</th>
                <th>Params</th>
              </tr>
            </thead>
            <tbody>
              {jobs
                .filter((j) => j.done)
                .map((j) => (
                  <tr key={`${j.width}-${j.activation}`}>
                    <td>{j.width}</td>
                    <td>
                      <span className="dot" style={{ background: ACTIVATION_COLORS[j.activation] }} />
                      {activationMeta(j.activation).label}
                    </td>
                    <td>{j.trainMse.toExponential(2)}</td>
                    <td className="emph">{j.testMse.toExponential(2)}</td>
                    <td>{j.functionMse.toExponential(2)}</td>
                    <td>{j.epochsToTarget ?? "—"}</td>
                    <td>{(j.runtimeMs / 1000).toFixed(2)}s</td>
                    <td>{j.params}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}

        <p className="pde-note">
          Each width trains a single hidden layer for {budget} epochs. Cauchy carries three
          extra parameters per neuron, so compare the <em>Params</em> column, not just the
          width — reaching an error with fewer neurons is not the same as reaching it with
          fewer parameters.
        </p>
      </div>
    </div>
  );
}
