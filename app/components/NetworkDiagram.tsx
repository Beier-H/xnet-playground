"use client";

import { MAX_LAYERS, MAX_NEURONS, neuronCurves, type Activation, type Network } from "../lib/model";

export type NeuronRef = { layer: number; index: number };

const THUMB_XS = Array.from({ length: 25 }, (_, i) => -1 + (2 * i) / 24);

const NODE = 46;
const ROW = 66;
const COL_MIN = 108;

type Props = {
  net: Network;
  activation: Activation;
  hovered: NeuronRef | null;
  onHover: (ref: NeuronRef | null) => void;
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
 * Output magnitude that fills most of a thumbnail before saturating. Low enough
 * that a quiet neuron early in training is still legible, high enough that a
 * unit doing real work has not yet saturated.
 */
const THUMB_SCALE = 0.6;

/**
 * Draws a neuron's response on a *fixed* vertical scale.
 *
 * Auto-ranging each box to its own min/max is tempting but unusable while
 * training: every box is stretched to full height regardless of amplitude, so
 * a neuron that barely moves still renders full-height and the tiniest change
 * makes the curve leap across the box on the next frame. A fixed squash keeps
 * the drawing still unless the neuron's output genuinely moves, and lets you
 * compare amplitudes across neurons. tanh rather than a hard clamp so large
 * Cauchy spikes saturate smoothly instead of clipping flat.
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
  onHover,
  onAddLayer,
  onRemoveLayer,
  onAddNeuron,
  onRemoveNeuron,
}: Props) {
  const hiddenLayers = net.slice(0, -1);
  const columns = hiddenLayers.length + 2; // input + hidden + output
  const tallest = Math.max(1, ...hiddenLayers.map((layer) => layer.length));
  const width = Math.max(COL_MIN * columns, 420);
  const height = Math.max(tallest, 3) * ROW + 30;
  const curves = neuronCurves(net, THUMB_XS, activation);

  const colX = (col: number) => ((col + 0.5) * width) / columns;
  const rowY = (index: number, count: number) =>
    height / 2 + (index - (count - 1) / 2) * ROW;

  // Column index of each layer in `net`: hidden layers first, then the output.
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
            <span className="layer-caption">
              {layer.length} neuron{layer.length === 1 ? "" : "s"}
            </span>
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
        {/* Edges are drawn first so the node boxes sit on top of them. */}
        {net.map((layer, l) =>
          layer.map((neuron, j) =>
            neuron.w.map((w, i) => {
              const fromCol = colX(layerColumn(l) - 1);
              const toCol = colX(layerColumn(l));
              const fromCount = l === 0 ? 1 : net[l - 1].length;
              const y1 = rowY(i, fromCount);
              const y2 = rowY(j, layer.length);
              const mid = (fromCol + toCol) / 2;
              // Highlight the edges entering and leaving the hovered neuron.
              const related =
                hovered !== null &&
                ((hovered.layer === l && hovered.index === j) ||
                  (hovered.layer === l - 1 && hovered.index === i));
              const dim = hovered !== null && !related;
              return (
                <path
                  key={`${l}-${j}-${i}`}
                  d={`M${fromCol + NODE / 2},${y1} C${mid},${y1} ${mid},${y2} ${toCol - NODE / 2},${y2}`}
                  className={`edge ${w >= 0 ? "edge-pos" : "edge-neg"}`}
                  strokeWidth={edgeWidth(w)}
                  opacity={dim ? 0.18 : 0.75}
                />
              );
            }),
          ),
        )}

        {/* Input node */}
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

        {/* Hidden neurons, each showing its own response curve over x ∈ [−1, 1] */}
        {hiddenLayers.map((layer, l) =>
          layer.map((neuron, j) => {
            const cx = colX(layerColumn(l));
            const cy = rowY(j, layer.length);
            const isHovered = hovered?.layer === l && hovered.index === j;
            return (
              <g
                key={`n-${l}-${j}`}
                onMouseEnter={() => onHover({ layer: l, index: j })}
                onMouseLeave={() => onHover(null)}
                className="node-group"
              >
                <rect
                  x={cx - NODE / 2}
                  y={cy - NODE / 2}
                  width={NODE}
                  height={NODE}
                  rx="7"
                  className={`node ${isHovered ? "node-active" : ""}`}
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

        {/* Output node */}
        <g>
          <rect
            x={colX(columns - 1) - NODE / 2}
            y={rowY(0, 1) - NODE / 2}
            width={NODE}
            height={NODE}
            rx="7"
            className="node node-io"
          />
          <text
            x={colX(columns - 1)}
            y={rowY(0, 1) + 6}
            textAnchor="middle"
            className="node-label"
          >
            F
          </text>
        </g>
      </svg>

      <p className="network-hint">
        Each box plots that neuron&rsquo;s own output across x. Line thickness is weight
        magnitude; <span className="swatch-pos">orange</span> is positive,{" "}
        <span className="swatch-neg">blue</span> negative. Hover a neuron to trace it in the
        approximation plot.
      </p>

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
