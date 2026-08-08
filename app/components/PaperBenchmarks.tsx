"use client";

import { useState } from "react";

import { PAPER_SOURCES, PAPER_TASKS } from "../lib/paperBenchmarks";

function fmt(value: number): string {
  return value < 1e-2 ? value.toExponential(2) : value.toString();
}

/**
 * Published reference figures, kept rigidly separate from anything the
 * playground computes. Every row carries the "Paper Result" label and the
 * source it came from; no value here is ever produced by a live run.
 */
export default function PaperBenchmarks() {
  const [open, setOpen] = useState(false);

  return (
    <section className="paper-panel">
      <button type="button" className="paper-toggle" onClick={() => setOpen(!open)}>
        <span className="badge paper">Paper Result</span>
        Reference benchmarks
        <span className="paper-count">
          {PAPER_TASKS.length} tasks · {PAPER_SOURCES.length} sources
        </span>
        <span className="chevron">{open ? "▴" : "▾"}</span>
      </button>

      {open && (
        <div className="paper-body">
          <p className="paper-note strong">
            Reported in the papers below — <strong>not</strong> produced by this playground,
            and not comparable with the Live Result tables. Those papers train much larger
            networks, for longer, with different optimisers.
          </p>

          {PAPER_TASKS.map((group) => (
            <div className="paper-group" key={`${group.source}-${group.task}`}>
              <div className="paper-group-head">
                <strong>{group.task}</strong>
                <cite>{group.source}</cite>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Architecture</th>
                    <th>Metric</th>
                    <th>Reported</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {group.entries.map((entry, i) => (
                    <tr key={i} className={entry.isXNet ? "xnet" : undefined}>
                      <td>{entry.method}</td>
                      <td className="muted mono">{entry.architecture}</td>
                      <td className="muted">{entry.metric}</td>
                      <td className="emph">{fmt(entry.value)}</td>
                      <td>{entry.timeSeconds === undefined ? "—" : `${entry.timeSeconds}s`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          <p className="paper-note">
            Blank cells mean the paper did not report that quantity. Nothing here is
            estimated or filled in. Values live in <code>app/lib/paperBenchmarks.ts</code>.
          </p>
        </div>
      )}
    </section>
  );
}
