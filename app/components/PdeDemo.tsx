"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import LossChart, { HISTORY_LIMIT, type LossSeries } from "./LossChart";
import { ACTIVATION_COLORS, activationMeta, type ActivationId } from "../lib/activations";
import {
  PDES,
  exactSolution,
  makePdeNet,
  makeSamples,
  pdeAbsError,
  pdeLoss,
  pdeParameterCount,
  pdePredict,
  pdeResidual,
  pdeStep,
  pdeMeta,
  type PdeId,
  type PdeNet,
} from "../lib/pde";
import { PDE_LRS } from "../lib/urlState";

/** The three PINNs compared, under identical architecture, seed and optimiser. */
const PINNS: ActivationId[] = ["cauchy", "tanh", "relu"];
const N_HIDDEN = 6;
const SEED = 4242;
const GRID = Array.from({ length: 81 }, (_, i) => i / 80);

const W = 340;
const H = 150;
const PAD = { left: 40, right: 12, top: 12, bottom: 24 };

type Run = { activation: ActivationId; net: PdeNet; history: number[] };

function buildRuns(pde: PdeId): Run[] {
  const nIn = pdeMeta(pde).timeDependent ? 2 : 1;
  return PINNS.map((activation) => ({
    activation,
    net: makePdeNet(activation, nIn, N_HIDDEN, SEED),
    history: [],
  }));
}

type LineChartProps = {
  title: string;
  xs: number[];
  series: { id: string; color: string; values: number[]; dashed?: boolean }[];
  logScale?: boolean;
};

function LineChart({ title, xs, series, logScale }: LineChartProps) {
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const rawMax = Math.max(1e-6, ...all.map(Math.abs));
  const max = logScale ? rawMax : Math.max(rawMax, 0.1);
  const min = logScale ? 0 : Math.min(0, ...all);

  const toX = (i: number) => PAD.left + (i / (xs.length - 1)) * (W - PAD.left - PAD.right);
  const toY = (v: number) => {
    if (logScale) {
      const lo = Math.log10(Math.max(1e-8, max)) - 4;
      const t = (Math.log10(Math.max(1e-8, Math.abs(v))) - lo) / 4;
      return PAD.top + (1 - Math.max(0, Math.min(1, t))) * (H - PAD.top - PAD.bottom);
    }
    const t = (v - min) / (max - min || 1);
    return PAD.top + (1 - Math.max(0, Math.min(1, t))) * (H - PAD.top - PAD.bottom);
  };

  return (
    <div className="pde-chart">
      <span className="pde-chart-title">{title}</span>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        <rect x="0" y="0" width={W} height={H} rx="8" className="plot-background" />
        <line x1={PAD.left} x2={W - PAD.right} y1={toY(logScale ? 1e-8 : 0)} y2={toY(logScale ? 1e-8 : 0)} className="zero-line" />
        <text x={PAD.left - 6} y={PAD.top + 8} textAnchor="end" className="axis-label">
          {logScale ? max.toExponential(0) : max.toFixed(1)}
        </text>
        {series.map((s) => (
          <path
            key={s.id}
            d={s.values
              .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`)
              .join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth={s.dashed ? 1.8 : 2.2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
          />
        ))}
      </svg>
    </div>
  );
}

export default function PdeDemo({
  pde,
  learningRate,
  onChangePde,
  onChangeLearningRate,
}: {
  pde: PdeId;
  learningRate: number;
  onChangePde: (id: PdeId) => void;
  onChangeLearningRate: (lr: number) => void;
}) {
  const meta = pdeMeta(pde);
  const [runs, setRuns] = useState<Run[]>(() => buildRuns(pde));
  const [steps, setSteps] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);

  const samples = useMemo(() => makeSamples(pde), [pde]);
  const runsRef = useRef(runs);
  useEffect(() => {
    runsRef.current = runs;
  }, [runs]);

  const reset = useCallback(
    (nextPde: PdeId) => {
      setPlaying(false);
      setRuns(buildRuns(nextPde));
      setSteps(0);
      setTime(0);
    },
    [],
  );

  const step = useCallback(() => {
    const next = runsRef.current.map((run) => {
      const net = pdeStep(run.net, pde, samples, learningRate);
      return {
        ...run,
        net,
        history: [...run.history, pdeLoss(net, pde, samples)].slice(-HISTORY_LIMIT),
      };
    });
    runsRef.current = next;
    setRuns(next);
    setSteps((s) => s + 1);
  }, [pde, samples, learningRate]);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const loop = () => {
      step();
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [playing, step]);

  const xs = useMemo(
    () => GRID.map((g) => meta.xMin + g * (meta.xMax - meta.xMin)),
    [meta.xMin, meta.xMax],
  );
  const t = meta.timeDependent ? time : 0;

  const exact = useMemo(() => xs.map((x) => exactSolution(pde, x, t)), [xs, pde, t]);

  const solutionSeries = [
    { id: "exact", color: "#8494a6", values: exact, dashed: true },
    ...runs.map((r) => ({
      id: r.activation,
      color: ACTIVATION_COLORS[r.activation],
      values: xs.map((x) => pdePredict(r.net, x, t)),
    })),
  ];

  const errorSeries = runs.map((r) => ({
    id: r.activation,
    color: ACTIVATION_COLORS[r.activation],
    values: xs.map((x, i) => Math.abs(pdePredict(r.net, x, t) - exact[i])),
  }));

  const residualSeries = runs.map((r) => ({
    id: r.activation,
    color: ACTIVATION_COLORS[r.activation],
    values: xs.map((x) => Math.abs(pdeResidual(r.net, pde, x, t))),
  }));

  const lossSeries: LossSeries[] = runs.map((r) => ({
    id: r.activation,
    color: ACTIVATION_COLORS[r.activation],
    values: r.history,
  }));

  const nIn = meta.timeDependent ? 2 : 1;

  return (
    <div className="pde">
      <section className="control-bar" aria-label="PDE controls">
        <div className="transport">
          <button
            type="button"
            className="icon-button"
            aria-label="Reset the networks"
            onClick={() => reset(pde)}
          >
            ↺
          </button>
          <button
            type="button"
            className="play-button"
            aria-label={playing ? "Pause training" : "Start training"}
            onClick={() => setPlaying((v) => !v)}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="Train one step"
            onClick={() => {
              setPlaying(false);
              step();
            }}
          >
            ▶❘
          </button>
        </div>

        <div className="readout">
          <span>Step</span>
          <strong>{steps.toLocaleString("en-US", { minimumIntegerDigits: 5, useGrouping: true })}</strong>
        </div>

        <label className="field">
          Equation
          <select
            value={pde}
            onChange={(e) => {
              const next = e.target.value as PdeId;
              onChangePde(next);
              reset(next);
            }}
          >
            {PDES.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Learning rate
          <select value={learningRate} onChange={(e) => onChangeLearningRate(Number(e.target.value))}>
            {PDE_LRS.map((lr) => (
              <option key={lr} value={lr}>
                {lr}
              </option>
            ))}
          </select>
        </label>

        {meta.timeDependent && (
          <label className="field slider-field">
            t = <strong>{time.toFixed(2)}</strong>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={time}
              onChange={(e) => setTime(Number(e.target.value))}
            />
          </label>
        )}
      </section>

      <div className="pde-body">
        <div className="pde-equation">
          <code>{meta.equation}</code>
          <span className="muted">exact: {meta.exactLabel}</span>
        </div>

        <div className="pde-charts">
          <LineChart title={`Solution u(x, t=${t.toFixed(2)}) — dashed is exact`} xs={xs} series={solutionSeries} />
          <LineChart title="Absolute error |u − u_exact|" xs={xs} series={errorSeries} logScale />
          <LineChart title="PDE residual |r(x, t)|" xs={xs} series={residualSeries} logScale />
          <div className="pde-chart">
            <span className="pde-chart-title">Training loss (log)</span>
            <LossChart series={lossSeries} />
          </div>
        </div>

        <table className="pde-table">
          <thead>
            <tr>
              <th>Network</th>
              <th>Loss</th>
              <th>Mean |error|</th>
              <th>Params</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.activation}>
                <td>
                  <span className="dot" style={{ background: ACTIVATION_COLORS[r.activation] }} />
                  {activationMeta(r.activation).label} PINN
                </td>
                <td>{pdeLoss(r.net, pde, samples).toExponential(2)}</td>
                <td className="emph">{pdeAbsError(r.net, pde).toExponential(2)}</td>
                <td>{pdeParameterCount(r.activation, nIn, N_HIDDEN)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <p className="pde-note">
          Prototype: one hidden layer of {N_HIDDEN} units, identical seed and optimiser
          (Adam on finite-difference gradients). ReLU carries φ″ = 0, so it cannot
          represent a second-order operator at all — that is a property of the
          activation, not a bug.
        </p>
      </div>
    </div>
  );
}
