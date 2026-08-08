"use client";

import type { ActivationId } from "../lib/activations";
import { MAX_LAYERS, MAX_NEURONS, neuronCurves, type Network } from "../lib/model";

export type NeuronRef = { layer: number; index: number };

const THUMB_XS = Array.from({ length: 25 }, (_, i) => -1 + (2 * i) / 24);

const NODE = 44;
const ROW = 62;
const COL_MIN = 104;

/**
 * Output magnitude that fills most of a thumbnail before saturating. Low enough
 * that a quiet neuron early in training is still legible, high enough that a
 * unit doing real work has not yet saturated.
 */
const THUMB_SCALE = 0.6;

type Props = {
  net: Network;
  activation: ActivationId;
  hovered: NeuronRef | null;
  selected: NeuronRef | null;
  onHover: (ref: NeuronRef | null) => void;
  onSelect: (ref: NeuronRef | null) => void;
  onAddLayer: () => void;
  onRemoveLayer: () => void;
  onAddNeuron: (layer: number) => void;
  onRemoveNeuron: (layer: number) => void;
};

/** Maps a weight to a stroke width; small weights stay visible but thin. */
function edgeWidth(w: number): number {
  return Math.min(5, 0.4 + Math.abs(w) * 1.6);
}

/**
 * Draws a neuron's response on a *fixed* vertical scale.
 *
 * Auto-ranging each box to its own min/max is unusable while training: every
 * box would be stretched to full height regardless of amplitude, so a neuron
 * that barely moves still renders full-height and the tiniest change makes the
 * curve leap across the box on the next frame. tanh rather than a hard clamp so
 * large Cauchy spikes saturate smoothly instead of clipping flat.
 */
function polyline(values: number[], width: number, height: number): string {
  const pad = 5;
  const usable = height - pad * 2;
  return values
    .map((value, i) => {
      const x = pad + (i / (values.length - 1)) * (width - pad * 2);
      const unit = Math.tanh((Number.isFinite(value) ? value : 0) / THUMB_SCALE);
      const y = pad + (1 - (unit + 1) / 2) * usable;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function NetworkDiagram({
  net,
  activation,
  hovered,
  selected,
  onHover,
  onSelect,
  onAddLayer,
  onRemoveLayer,
  onAddNeuron,
  onRemoveNeuron,
}: Props) {
  const hiddenLayers = net.slice(0, -1);
  const columns = hiddenLayers.length + 2;
  const tallest = Math.max(1, ...hiddenLayers.map((layer) => layer.length));
  const width = Math.max(COL_MIN * columns, 400);
  const height = Math.max(tallest, 3) * ROW + 26;
  const curves = neuronCurves(net, THUMB_XS, activation);

  // Whichever neuron the detail panel is currently describing.
  const focus = hovered ?? selected;

  const colX = (col: number) => ((col + 0.5) * width) / columns;
  const rowY = (index: number, count: number) => height / 2 + (index - (count - 1) / 2) * ROW;
  const layerColumn = (l: number) => l + 1;

  return (
    <div className="network">
      <div className="layer-bar" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        <div className="layer-cell">
          <span className="layer-caption">Input</span>
        </div>
        {hiddenLayers.map((layer, l) => (
          <div className="layer-cell" key={l}>
            <div className="stepper">
              <button
                type="button"
                aria-label={`Add neuron to layer ${l + 1}`}
                disabled={layer.length >= MAX_NEURONS}
                onClick={() => onAddNeuron(l)}
              >
                +
              </button>
              <button
                type="button"
                aria-label={`Remove neuron from layer ${l + 1}`}
                disabled={layer.length <= 1}
                onClick={() => onRemoveNeuron(l)}
              >
                −
              </button>
            </div>
            <span className="layer-caption">{layer.length}</span>
          </div>
        ))}
        <div className="layer-cell">
          <span className="layer-caption">Output</span>
        </div>
      </div>

      <svg
        className="network-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Network architecture with per-neuron response curves"
      >
        {net.map((layer, l) =>
          layer.map((neuron, j) =>
            neuron.w.map((w, i) => {
              const fromCol = colX(layerColumn(l) - 1);
              const toCol = colX(layerColumn(l));
              const fromCount = l === 0 ? 1 : net[l - 1].length;
              const y1 = rowY(i, fromCount);
              const y2 = rowY(j, layer.length);
              const mid = (fromCol + toCol) / 2;
              // Highlight the edges entering and leaving the focused neuron.
              const related =
                focus !== null &&
                ((focus.layer === l && focus.index === j) ||
                  (focus.layer === l - 1 && focus.index === i));
              const dim = focus !== null && !related;
              return (
                <path
                  key={`${l}-${j}-${i}`}
                  d={`M${fromCol + NODE / 2},${y1} C${mid},${y1} ${mid},${y2} ${toCol - NODE / 2},${y2}`}
                  className={`edge ${w >= 0 ? "edge-pos" : "edge-neg"}`}
                  strokeWidth={edgeWidth(w)}
                  opacity={dim ? 0.15 : 0.75}
                />
              );
            }),
          ),
        )}

        <g>
          <rect
            x={colX(0) - NODE / 2}
            y={rowY(0, 1) - NODE / 2}
            width={NODE}
            height={NODE}
            rx="7"
            className="node node-io"
          />
          <text x={colX(0)} y={rowY(0, 1) + 6} textAnchor="middle" className="node-label">
            x
          </text>
        </g>

        {hiddenLayers.map((layer, l) =>
          layer.map((neuron, j) => {
            const cx = colX(layerColumn(l));
            const cy = rowY(j, layer.length);
            const isFocus = focus?.layer === l && focus.index === j;
            const isPinned = selected?.layer === l && selected.index === j;
            return (
              <g
                key={`n-${l}-${j}`}
                onMouseEnter={() => onHover({ layer: l, index: j })}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(isPinned ? null : { layer: l, index: j })}
                className="node-group"
                // Everything but the neuron under inspection recedes.
                opacity={focus !== null && !isFocus ? 0.3 : 1}
              >
                <rect
                  x={cx - NODE / 2}
                  y={cy - NODE / 2}
                  width={NODE}
                  height={NODE}
                  rx="7"
                  className={`node ${isFocus ? "node-active" : ""} ${isPinned ? "node-pinned" : ""}`}
                />
                <polyline
                  points={polyline(curves[l][j], NODE, NODE)}
                  className="node-curve"
                  transform={`translate(${cx - NODE / 2},${cy - NODE / 2})`}
                />
              </g>
            );
          }),
        )}

        <g>
          <rect
            x={colX(columns - 1) - NODE / 2}
            y={rowY(0, 1) - NODE / 2}
            width={NODE}
            height={NODE}
            rx="7"
            className="node node-io"
          />
          <text x={colX(columns - 1)} y={rowY(0, 1) + 6} textAnchor="middle" className="node-label">
            F
          </text>
        </g>
      </svg>

      <div className="layer-controls">
        <button type="button" onClick={onRemoveLayer} disabled={hiddenLayers.length === 0}>
          −
        </button>
        <strong>
          {hiddenLayers.length} hidden layer{hiddenLayers.length === 1 ? "" : "s"}
        </strong>
        <button type="button" onClick={onAddLayer} disabled={hiddenLayers.length >= MAX_LAYERS}>
          +
        </button>
      </div>
    </div>
  );
}
