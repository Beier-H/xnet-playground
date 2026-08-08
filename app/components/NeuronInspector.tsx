"use client";

import ActivationShapeChart from "./ActivationShapeChart";
import type { NeuronRef } from "./NetworkDiagram";
import { activationMeta, type ActivationId } from "../lib/activations";
import { shapeOf, type Contribution, type Localization, type Network } from "../lib/model";

type Props = {
  net: Network;
  activation: ActivationId;
  ref_: NeuronRef;
  contribution: Contribution;
  localization: Localization | null;
  showDerivative: boolean;
  pinned: boolean;
};

/**
 * Detail for one neuron: the exact shape it has learned, its parameters, and
 * how much the network's output actually depends on it.
 */
export default function NeuronInspector({
  net,
  activation,
  ref_,
  contribution,
  localization,
  showDerivative,
  pinned,
}: Props) {
  const neuron = net[ref_.layer]?.[ref_.index];
  if (!neuron) return null;

  const params = shapeOf(neuron);
  const meta = activationMeta(activation);
  const isCauchy = activation === "cauchy";

  return (
    <div className="inspector">
      <div className="inspector-head">
        <strong>
          Layer {ref_.layer + 1} · neuron {ref_.index + 1}
        </strong>
        <span className={pinned ? "pin-badge on" : "pin-badge"}>{pinned ? "pinned" : "hover"}</span>
      </div>

      <ActivationShapeChart
        activation={activation}
        params={params}
        showDerivative={showDerivative}
        caption="this neuron"
      />

      {localization && (
        <div className="localization-row">
          <span className="badge localized">Localized response</span>
          <span className="chip cauchy">μ {localization.mu.toFixed(3)}</span>
          <span className="chip cauchy">width {localization.width.toFixed(3)}</span>
        </div>
      )}

      <div className="param-chips">
        {isCauchy ? (
          <>
            <span className="chip cauchy">λ₁ {neuron.l1.toFixed(2)}</span>
            <span className="chip cauchy">λ₂ {neuron.l2.toFixed(2)}</span>
            <span className="chip cauchy">d {params.d.toFixed(2)}</span>
          </>
        ) : (
          <span className="chip muted">{meta.label}: shape is fixed</span>
        )}
        <span className="chip">w {neuron.w.map((v) => v.toFixed(2)).join(", ")}</span>
        <span className="chip">b {neuron.b.toFixed(2)}</span>
      </div>

      <div className="influence">
        <span>Contribution to output</span>
        <strong>{contribution.peak.toFixed(3)}</strong>
        <small>
          {contribution.band
            ? `strongest on x ∈ [${contribution.band.lo.toFixed(2)}, ${contribution.band.hi.toFixed(2)}]`
            : "negligible — this neuron is not doing much"}
        </small>
      </div>
    </div>
  );
}
