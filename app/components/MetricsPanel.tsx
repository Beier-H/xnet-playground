"use client";

import { ACTIVATION_COLORS, activationMeta, type ActivationId } from "../lib/activations";

export type RunMetrics = {
  activation: ActivationId;
  trainLoss: number;
  testLoss: number;
  functionMse: number;
  /** Error restricted to |x| ≤ 0.15; only meaningful for the step target. */
  localMse: number;
  epochsToTarget: number | null;
  runtimeMs: number;
  params: number;
};

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return value < 1e-3 ? value.toExponential(1) : value.toFixed(4);
}

/**
 * Deliberately shows accuracy, convergence and cost side by side. Cauchy buys
 * expressiveness with three extra parameters per neuron and more work per step,
 * and the table is meant to let that trade-off be seen rather than asserted.
 */
export default function MetricsPanel({
  rows,
  errorTarget,
  showLocal,
}: {
  rows: RunMetrics[];
  errorTarget: number;
  /** Adds the near-discontinuity column; only shown for the step target. */
  showLocal: boolean;
}) {
  const multi = rows.length > 1;
  // Lowest function MSE wins; only marked when there is something to compare.
  const best = multi
    ? rows.reduce((a, b) => (b.functionMse < a.functionMse ? b : a)).activation
    : null;

  return (
    <div className="metrics">
      <div className="metrics-head">
        <span className="badge live">Live Result</span>
        <span className="muted">computed in this browser</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>{multi ? "Activation" : "Metric"}</th>
            <th title="Mean squared error on the noisy training samples">Train</th>
            <th title="Mean squared error on the held-out samples">Test</th>
            <th title="Squared error against the true noiseless target on a dense grid">
              Fn MSE
            </th>
            {showLocal && (
              <th title="Squared error restricted to |x| ≤ 0.15, around the discontinuity">
                Local MSE
              </th>
            )}
            <th title={`First epoch where Fn MSE ≤ ${errorTarget}`}>
              Ep→{errorTarget.toExponential(0)}
            </th>
            <th title="Wall-clock time spent training">Time</th>
            <th title="Trainable parameters, including Cauchy shape parameters">Params</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.activation} className={r.activation === best ? "best" : undefined}>
              <td>
                <span className="dot" style={{ background: ACTIVATION_COLORS[r.activation] }} />
                {activationMeta(r.activation).label}
              </td>
              <td>{fmt(r.trainLoss)}</td>
              <td>{fmt(r.testLoss)}</td>
              <td className="emph">{fmt(r.functionMse)}</td>
              {showLocal && <td className="emph">{fmt(r.localMse)}</td>}
              <td>{r.epochsToTarget === null ? "—" : r.epochsToTarget.toLocaleString()}</td>
              <td>{(r.runtimeMs / 1000).toFixed(1)}s</td>
              <td>{r.params}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
