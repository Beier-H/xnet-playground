"use client";

import { kanEdgeCurves, type KanNet } from "../lib/kan";

/** Which learnable edge function is under inspection. 0 = input→hidden. */
export type EdgeRef = { layer: 0 | 1; index: number };

const THUMB_XS = Array.from({ length: 25 }, (_, i) => -1 + (2 * i) / 24);

const BOX = 42;
const ROW = 62;
const NODE_R = 9;

/** Same fixed squash as the MLP diagram, so the two are visually comparable. */
const SCALE = 0.6;

function polyline(values: number[], size: number): string {
  const pad = 5;
  const usable = size - pad * 2;
  return values
    .map((v, i) => {
      const x = pad + (i / (values.length - 1)) * (size - pad * 2);
      const unit = Math.tanh((Number.isFinite(v) ? v : 0) / SCALE);
      const y = pad + (1 - (unit + 1) / 2) * usable;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

type Props = {
  net: KanNet;
  hovered: EdgeRef | null;
  selected: EdgeRef | null;
  onHover: (ref: EdgeRef | null) => void;
  onSelect: (ref: EdgeRef | null) => void;
};

/**
 * KAN's signature view: the learnable function lives on the *edge*, so each
 * edge carries its own little plot rather than a thickness. Nodes are plain
 * sums and are drawn as dots.
 *
 * Both plots are drawn against x rather than against each edge's own input.
 * That keeps them comparable with every other chart here, and it makes the
 * outer row literally "this path's contribution to F(x)" — the outer curves
 * sum to the network output.
 */
export default function KanDiagram({ net, hovered, selected, onHover, onSelect }: Props) {
  const n = net.width;
  const W = 520;
  const H = Math.max(n, 3) * ROW + 26;
  const curves = kanEdgeCurves(net, THUMB_XS);

  const xIn = 40;
  const xHidden = W / 2;
  const xOut = W - 40;
  const xInner = (xIn + xHidden) / 2;
  const xOuter = (xHidden + xOut) / 2;
  const rowY = (i: number) => H / 2 + (i - (n - 1) / 2) * ROW;
  const midY = H / 2;

  const focus = hovered ?? selected;
  const isFocus = (layer: 0 | 1, i: number) => focus?.layer === layer && focus.index === i;
  const dim = (layer: 0 | 1, i: number) => focus !== null && !isFocus(layer, i);

  const edgeBox = (layer: 0 | 1, i: number, cx: number, cy: number, values: number[]) => {
    const pinned = selected?.layer === layer && selected.index === i;
    return (
      <g
        key={`${layer}-${i}`}
        className="node-group"
        opacity={dim(layer, i) ? 0.28 : 1}
        onMouseEnter={() => onHover({ layer, index: i })}
        onMouseLeave={() => onHover(null)}
        onClick={() => onSelect(pinned ? null : { layer, index: i })}
      >
        <rect
          x={cx - BOX / 2}
          y={cy - BOX / 2}
          width={BOX}
          height={BOX}
          rx="7"
          className={`node ${isFocus(layer, i) ? "node-active" : ""} ${pinned ? "node-pinned" : ""}`}
        />
        <polyline
          points={polyline(values, BOX)}
          className="kan-curve"
          transform={`translate(${cx - BOX / 2},${cy - BOX / 2})`}
        />
      </g>
    );
  };

  return (
    <div className="network">
      <div className="kan-legend">
        <span>φᵢ(x) — input edge</span>
        <span>ψᵢ(hᵢ) — output edge</span>
        <span className="muted">both plotted over x</span>
      </div>

      <svg
        className="network-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="KAN architecture with learnable functions on the edges"
      >
        {Array.from({ length: n }, (_, i) => {
          const y = rowY(i);
          return (
            <g key={`wire-${i}`} opacity={focus === null ? 0.5 : 0.2}>
              <path
                d={`M${xIn + NODE_R},${midY} C${xInner},${midY} ${xInner},${y} ${xHidden - NODE_R},${y}`}
                className="kan-wire"
              />
              <path
                d={`M${xHidden + NODE_R},${y} C${xOuter},${y} ${xOuter},${midY} ${xOut - NODE_R},${midY}`}
                className="kan-wire"
              />
            </g>
          );
        })}

        {/* Input and output nodes */}
        <circle cx={xIn} cy={midY} r={NODE_R} className="kan-node" />
        <text x={xIn} y={midY - 16} textAnchor="middle" className="node-label">
          x
        </text>
        <circle cx={xOut} cy={midY} r={NODE_R} className="kan-node" />
        <text x={xOut} y={midY - 16} textAnchor="middle" className="node-label">
          F
        </text>

        {/* Hidden nodes are bare sums */}
        {Array.from({ length: n }, (_, i) => (
          <circle key={`h-${i}`} cx={xHidden} cy={rowY(i)} r={NODE_R - 2} className="kan-node" />
        ))}

        {Array.from({ length: n }, (_, i) => edgeBox(0, i, xInner, rowY(i), curves.inner[i]))}
        {Array.from({ length: n }, (_, i) => edgeBox(1, i, xOuter, rowY(i), curves.outer[i]))}
      </svg>

      <div className="layer-controls">
        <strong>
          KAN [1, {n}, 1] · {n * 2} learnable edge functions
        </strong>
      </div>
    </div>
  );
}
