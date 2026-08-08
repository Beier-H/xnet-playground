"use client";

import { useState } from "react";

import { PAPER_BENCHMARKS } from "../lib/paperBenchmarks";

/**
 * Published reference figures, kept rigidly separate from anything the
 * playground computes. Every row is labelled "Paper Result" and no value is
 * ever derived from a live run.
 */
export default function PaperBenchmarks() {
  const [open, setOpen] = useState(false);
  const { citation, rows } = PAPER_BENCHMARKS;
  const filled = rows.filter((r) => r.value !== null).length;

  return (
    <section className="paper-panel">
      <button type="button" className="paper-toggle" onClick={() => setOpen(!open)}>
        <span className="paper-badge">Paper Result</span>
        Reference benchmarks
        <span className="paper-count">{filled === 0 ? "not supplied" : `${filled}/${rows.length}`}</span>
        <span className="chevron">{open ? "▴" : "▾"}</span>
      </button>

      {open && (
        <div className="paper-body">
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Metric</th>
                <th>Cauchy / XNet</th>
                <th>Baseline</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.task}>
                  <td>{row.task}</td>
                  <td className="muted">{row.metric}</td>
                  <td>{row.value === null ? <em className="empty">not supplied</em> : `${row.value}${row.unit ?? ""}`}</td>
                  <td>
                    {row.baseline
                      ? row.baseline.value === null
                        ? <em className="empty">not supplied</em>
                        : `${row.baseline.label}: ${row.baseline.value}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="paper-note">
            {citation ?? (
              <>
                No citation set. These rows stay empty on purpose — filling them with
                invented figures under a &ldquo;Paper Result&rdquo; label would be a fabricated
                citation. Add the reported values in{" "}
                <code>app/lib/paperBenchmarks.ts</code>.
              </>
            )}
          </p>
          <p className="paper-note">
            Reference figures only. They are not comparable with the live numbers above,
            which use a much smaller network and a different training budget.
          </p>
        </div>
      )}
    </section>
  );
}
