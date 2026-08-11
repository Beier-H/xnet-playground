"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ActivationShapeChart from "./components/ActivationShapeChart";
import ApproximationPlot, { type PlotRun } from "./components/ApproximationPlot";
import LossChart, { HISTORY_LIMIT, type LossPoint, type LossSeries } from "./components/LossChart";
import MetricsPanel, { type RunMetrics } from "./components/MetricsPanel";
import KanDiagram, { type EdgeRef } from "./components/KanDiagram";
import NetworkDiagram, { type NeuronRef } from "./components/NetworkDiagram";
import NeuronInspector from "./components/NeuronInspector";
import PaperBenchmarks from "./components/PaperBenchmarks";
import PdeDemo from "./components/PdeDemo";
import {
  buildModelState,
  kanWidthFor,
  modelColor,
  modelLabel,
  stateFunctionMse,
  stateLocalMse,
  stateLoss,
  stateParams,
  statePredict,
  trainModelEpoch,
  type ModelId,
  type ModelState,
} from "./lib/benchmark";
import { kanEdgeContribution } from "./lib/kan";
import DiscontinuityInset from "./components/DiscontinuityInset";
import WidthBenchmark from "./components/WidthBenchmark";
import {
  ACTIVATIONS,
  ACTIVATION_COLORS,
  activationMeta,
  type ActivationId,
} from "./lib/activations";
import {
  BATCH_SIZES,
  COMPARE_SET,
  ERROR_TARGETS,
  LEARNING_RATES,
  MAX_LAYERS,
  MAX_NEURONS,
  NOISE_LEVELS,
  PLOT_XS,
  REG_RATES,
  TARGETS,
  effectiveLocalization,
  makeDataset,
  mulberry32,
  neuronContribution,
  shuffledOrder,
  targetValue,
  type Regularization,
} from "./lib/model";
import {
  DEFAULT_CONFIG,
  readConfig,
  writeConfig,
  type Mode,
  type PlaygroundConfig,
} from "./lib/urlState";

/**
 * Config keys that invalidate whatever has been trained so far. Learning rate,
 * regularization and batch size are deliberately absent: those can change
 * mid-run, which is half the fun.
 */
const REBUILD_KEYS = [
  "shape",
  "netSeed",
  "activation",
  "target",
  "noise",
  "percentTrain",
  "dataSeed",
  "compare",
  "l1",
  "l2",
  "d",
] as const satisfies readonly (keyof PlaygroundConfig)[];

type Run = {
  model: ModelId;
  state: ModelState;
  history: LossPoint[];
  epochsToTarget: number | null;
  runtimeMs: number;
};

/** Compare pits the trainable-shape families against the fixed-shape ones. */
const COMPARE_MODELS: ModelId[] = [...COMPARE_SET, "kan"];

function modelsFor(config: PlaygroundConfig): ModelId[] {
  return config.compare ? COMPARE_MODELS : [config.activation];
}

function buildRuns(config: PlaygroundConfig): Run[] {
  const init = { l1: config.l1, l2: config.l2, d: config.d };
  return modelsFor(config).map((model) => ({
    model,
    // Identical shape and seed across models, so a comparison is fair.
    state: buildModelState(model, config.shape, config.netSeed, init),
    history: [],
    epochsToTarget: null,
    runtimeMs: 0,
  }));
}

export default function Playground() {
  const [config, setConfig] = useState<PlaygroundConfig>(DEFAULT_CONFIG);
  const [runs, setRuns] = useState<Run[]>(() => buildRuns(DEFAULT_CONFIG));
  const [epoch, setEpoch] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hovered, setHovered] = useState<NeuronRef | null>(null);
  const [selected, setSelected] = useState<NeuronRef | null>(null);
  const [focusModel, setFocusModel] = useState<ModelId>("cauchy");
  const [edgeHover, setEdgeHover] = useState<EdgeRef | null>(null);
  const [edgeSelect, setEdgeSelect] = useState<EdgeRef | null>(null);

  const runsRef = useRef(runs);
  useEffect(() => {
    runsRef.current = runs;
  }, [runs]);
  const epochRef = useRef(0);
  useEffect(() => {
    epochRef.current = epoch;
  }, [epoch]);
  const shuffleRng = useRef(mulberry32(7));

  const {
    mode,
    compare,
    activation,
    target,
    shape,
    learningRate,
    regularization,
    regRate,
    batchSize,
    noise,
    percentTrain,
    dataSeed,
    showTest,
    errorTarget,
    showDerivative,
  } = config;

  // Restore a shared experiment from the URL once, after mount. This has to be
  // an effect: `location.hash` does not exist during the server render, so
  // seeding state from it directly would break hydration.
  useEffect(() => {
    const fromUrl = readConfig(window.location.hash);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConfig(fromUrl);
    setRuns(buildRuns(fromUrl));
    setEpoch(0);
  }, []);

  useEffect(() => {
    window.history.replaceState(null, "", writeConfig(config));
  }, [config]);

  const dataset = useMemo(
    () => makeDataset(target, noise, percentTrain, dataSeed),
    [target, noise, percentTrain, dataSeed],
  );

  const restart = useCallback((next: PlaygroundConfig) => {
    setPlaying(false);
    setConfig(next);
    setRuns(buildRuns(next));
    setEpoch(0);
    setSelected(null);
  }, []);

  const update = useCallback(
    (patch: Partial<PlaygroundConfig>) => {
      const next = { ...config, ...patch };
      if (REBUILD_KEYS.some((key) => patch[key] !== undefined)) restart(next);
      else setConfig(next);
    },
    [config, restart],
  );

  const step = useCallback(() => {
    // Read the counter from a ref rather than state: taking `epoch` as a
    // dependency would rebuild `step` every epoch, which would tear down and
    // re-subscribe the animation frame on every single frame.
    const nextEpoch = epochRef.current + 1;
    epochRef.current = nextEpoch;
    // One shared visiting order, so every activation sees identical batches.
    const order = shuffledOrder(dataset.train.length, shuffleRng.current);

    const next = runsRef.current.map((run) => {
      const started = performance.now();
      const state = trainModelEpoch(
        run.state,
        run.model,
        dataset.train,
        {
          activation: run.model as ActivationId,
          learningRate,
          batchSize,
          regularization,
          regRate,
        },
        order,
      );
      const elapsed = performance.now() - started;
      const fnMse = stateFunctionMse(state, run.model, target);
      return {
        ...run,
        state,
        runtimeMs: run.runtimeMs + elapsed,
        epochsToTarget:
          run.epochsToTarget === null && fnMse <= errorTarget ? nextEpoch : run.epochsToTarget,
        history: [
          ...run.history,
          {
            train: stateLoss(state, run.model, dataset.train),
            test: stateLoss(state, run.model, dataset.test),
          },
        ].slice(-HISTORY_LIMIT),
      };
    });

    runsRef.current = next;
    setRuns(next);
    setEpoch(nextEpoch);
  }, [dataset, learningRate, batchSize, regularization, regRate, target, errorTarget]);

  useEffect(() => {
    if (!playing || mode !== "fit") return;
    let frame = 0;
    const loop = () => {
      step();
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [playing, step, mode]);

  // ------------------------------------------------------------- derived data

  const focusRun = useMemo(
    () => runs.find((r) => r.model === focusModel) ?? runs[0],
    [runs, focusModel],
  );
  const mlpNet = focusRun?.state.kind === "mlp" ? focusRun.state.net : null;
  const kanNet = focusRun?.state.kind === "kan" ? focusRun.state.net : null;

  const focusNeuron = hovered ?? selected;
  const focusEdge = edgeHover ?? edgeSelect;

  const contribution = useMemo(() => {
    // KAN's output is a plain sum of its outer edge functions, so one edge's
    // contribution is exact rather than measured by ablation.
    if (kanNet && focusEdge) {
      if (focusEdge.index >= kanNet.width) return null;
      return kanEdgeContribution(kanNet, PLOT_XS, focusEdge.index);
    }
    if (!focusNeuron || !mlpNet || !focusRun) return null;
    if (focusNeuron.layer >= mlpNet.length - 1) return null;
    if (focusNeuron.index >= mlpNet[focusNeuron.layer].length) return null;
    return neuronContribution(
      mlpNet,
      PLOT_XS,
      focusRun.model as ActivationId,
      focusNeuron.layer,
      focusNeuron.index,
    );
  }, [focusNeuron, focusEdge, focusRun, mlpNet, kanNet]);

  const localization = useMemo(() => {
    if (!focusNeuron || !mlpNet || !focusRun) return null;
    if (focusNeuron.layer >= mlpNet.length - 1) return null;
    return effectiveLocalization(
      mlpNet,
      PLOT_XS,
      focusRun.model as ActivationId,
      focusNeuron.layer,
      focusNeuron.index,
    );
  }, [focusNeuron, focusRun, mlpNet]);

  const isStep = target === "step";

  const metrics: RunMetrics[] = runs.map((run) => ({
    model: run.model,
    trainLoss: stateLoss(run.state, run.model, dataset.train),
    testLoss: stateLoss(run.state, run.model, dataset.test),
    functionMse: stateFunctionMse(run.state, run.model, target),
    localMse: stateLocalMse(run.state, run.model, target),
    epochsToTarget: run.epochsToTarget,
    runtimeMs: run.runtimeMs,
    params: stateParams(run.state, run.model, shape),
  }));

  const plotRuns: PlotRun[] = runs.map((run) => ({
    id: run.model,
    color: modelColor(run.model),
    predict: (x: number) => statePredict(run.state, run.model, x),
  }));

  const lossSeries: LossSeries[] = compare
    ? runs.map((run) => ({
        id: run.model,
        color: modelColor(run.model),
        values: run.history.map((h) => h.train),
      }))
    : [
        { id: "test", color: "#2b4257", values: runs[0]?.history.map((h) => h.test) ?? [], dashed: true },
        {
          id: "train",
          color: modelColor(runs[0]?.model ?? "cauchy"),
          values: runs[0]?.history.map((h) => h.train) ?? [],
        },
      ];

  const primary = ACTIVATIONS.filter((a) => a.primary);
  const secondary = ACTIVATIONS.filter((a) => !a.primary);
  const cauchyVisible = compare || activation === "cauchy";
  const shapeParams = { l1: config.l1, l2: config.l2, d: config.d };

  if (mode === "width") {
    return (
      <div className="page">
        <Masthead mode={mode} onMode={(m) => update({ mode: m })} />
        <WidthBenchmark
          target={target}
          learningRate={learningRate}
          batchSize={batchSize}
          regularization={regularization}
          regRate={regRate}
          noise={noise}
          percentTrain={percentTrain}
          dataSeed={dataSeed}
          netSeed={config.netSeed}
          errorTarget={errorTarget}
          init={shapeParams}
          onLearningRate={(lr) => setConfig({ ...config, learningRate: lr })}
          onErrorTarget={(v) => setConfig({ ...config, errorTarget: v })}
        />
        <PaperBenchmarks />
      </div>
    );
  }

  if (mode === "pde") {
    return (
      <div className="page">
        <Masthead mode={mode} onMode={(m) => update({ mode: m })} />
        <PdeDemo
          pde={config.pde}
          learningRate={config.pdeLearningRate}
          onChangePde={(id) => setConfig({ ...config, pde: id })}
          onChangeLearningRate={(lr) => setConfig({ ...config, pdeLearningRate: lr })}
        />
        <PaperBenchmarks />
      </div>
    );
  }

  return (
    <div className="page">
      <Masthead mode={mode} onMode={(m) => update({ mode: m })} />

      <section className="control-bar" aria-label="Training controls">
        <div className="transport">
          <button
            type="button"
            className="icon-button"
            title="Reset the network"
            aria-label="Reset the network"
            onClick={() => restart({ ...config, netSeed: Math.floor(Math.random() * 1e9) })}
          >
            ↺
          </button>
          <button
            type="button"
            className="play-button"
            aria-label={playing ? "Pause training" : "Start training"}
            onClick={() => setPlaying((value) => !value)}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <button
            type="button"
            className="icon-button"
            title="Train one epoch"
            aria-label="Train one epoch"
            onClick={() => {
              setPlaying(false);
              step();
            }}
          >
            ▶❘
          </button>
        </div>

        <div className="readout">
          <span>Epoch</span>
          <strong>{epoch.toLocaleString("en-US", { minimumIntegerDigits: 6, useGrouping: true })}</strong>
        </div>

        <div className="field activation-field">
          Activation
          <div className="activation-picker">
            {primary.map((a) => (
              <button
                key={a.id}
                type="button"
                title={a.note}
                disabled={compare}
                className={!compare && activation === a.id ? "seg selected" : "seg"}
                style={
                  !compare && activation === a.id
                    ? { borderColor: ACTIVATION_COLORS[a.id], color: ACTIVATION_COLORS[a.id] }
                    : undefined
                }
                onClick={() => update({ activation: a.id })}
              >
                {a.label}
              </button>
            ))}
            <select
              aria-label="More activations"
              disabled={compare}
              value={secondary.some((a) => a.id === activation) ? activation : ""}
              onChange={(e) => e.target.value && update({ activation: e.target.value as ActivationId })}
            >
              <option value="">More…</option>
              {secondary.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="button"
          className={compare ? "compare-toggle on" : "compare-toggle"}
          onClick={() => update({ compare: !compare })}
          title="Train Cauchy, ReLU and Tanh together on the same data, seed and optimiser"
        >
          Compare activations
        </button>

        <label className="field">
          Learning rate
          <select value={learningRate} onChange={(e) => update({ learningRate: Number(e.target.value) })}>
            {LEARNING_RATES.map((rate) => (
              <option key={rate} value={rate}>
                {rate}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Regularization
          <select
            value={regularization}
            onChange={(e) => update({ regularization: e.target.value as Regularization })}
          >
            <option value="none">None</option>
            <option value="l1">L1</option>
            <option value="l2">L2</option>
          </select>
        </label>

        <label className="field">
          Reg. rate
          <select
            value={regRate}
            disabled={regularization === "none"}
            onChange={(e) => update({ regRate: Number(e.target.value) })}
          >
            {REG_RATES.map((rate) => (
              <option key={rate} value={rate}>
                {rate}
              </option>
            ))}
          </select>
        </label>
      </section>

      <main className="playground">
        <section className="column" aria-label="Data">
          <h2>Data</h2>
          <div className="target-grid">
            {TARGETS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`target-chip ${target === item.id ? "selected" : ""} ${item.featured ? "featured" : ""}`}
                onClick={() => update({ target: item.id })}
                title={item.note}
              >
                <svg viewBox="0 0 60 34" aria-hidden="true">
                  <polyline
                    points={PLOT_XS.filter((_, i) => i % 2 === 0)
                      .map((x, i, arr) => {
                        const y = targetValue(item.id, x);
                        const mid = item.id === "step" || item.id === "runge" ? 24 : 17;
                        return `${((i / (arr.length - 1)) * 56 + 2).toFixed(1)},${(mid - y * 12).toFixed(1)}`;
                      })
                      .join(" ")}
                  />
                </svg>
                <span>{item.label}</span>
                {item.featured && <em className="chip-badge">XNet</em>}
              </button>
            ))}
          </div>

          <div className="noise-row">
            <span>Noise</span>
            <div className="segmented">
              {NOISE_LEVELS.map((level) => (
                <button
                  key={level}
                  type="button"
                  className={noise === level ? "seg selected" : "seg"}
                  onClick={() => update({ noise: level })}
                >
                  {level.toFixed(2)}
                </button>
              ))}
            </div>
          </div>

          <label className="slider">
            Train / test split: <strong>{percentTrain}%</strong>
            <input
              type="range"
              min="10"
              max="90"
              step="10"
              value={percentTrain}
              onChange={(e) => update({ percentTrain: Number(e.target.value) })}
            />
          </label>
          <label className="slider">
            Batch size: <strong>{batchSize}</strong>
            <input
              type="range"
              min="0"
              max={BATCH_SIZES.length - 1}
              step="1"
              value={BATCH_SIZES.indexOf(batchSize)}
              onChange={(e) => update({ batchSize: BATCH_SIZES[Number(e.target.value)] })}
            />
          </label>
          <button
            type="button"
            className="wide-button"
            onClick={() => update({ dataSeed: Math.floor(Math.random() * 1e9) })}
          >
            Regenerate
          </button>

          <h2 className="spaced">Activation shape</h2>
          <ActivationShapeChart
            activation={compare ? "cauchy" : activation}
            params={shapeParams}
            showDerivative={showDerivative}
            caption={compare ? "Cauchy (initial)" : activationMeta(activation).note}
          />
          <label className="checkbox">
            <input
              type="checkbox"
              checked={showDerivative}
              onChange={(e) => setConfig({ ...config, showDerivative: e.target.checked })}
            />
            Show φ′(z)
          </label>

          {cauchyVisible && (
            <div className="cauchy-params">
              <span className="formula">φ(x) = (λ₁x + λ₂) / (x² + d²)</span>
              <ParamSlider
                label="λ₁"
                value={config.l1}
                min={-2}
                max={2}
                step={0.05}
                onChange={(v) => update({ l1: v })}
              />
              <ParamSlider
                label="λ₂"
                value={config.l2}
                min={-2}
                max={2}
                step={0.05}
                onChange={(v) => update({ l2: v })}
              />
              <ParamSlider
                label="d"
                value={config.d}
                min={0.1}
                max={2}
                step={0.05}
                onChange={(v) => update({ d: v })}
              />
              <small>Initial values — training adapts them per neuron.</small>
            </div>
          )}
        </section>

        <section className="column column-network" aria-label="Network">
          <h2>Network</h2>
          {compare && (
            <div className="segmented run-tabs">
              {runs.map((run) => (
                <button
                  key={run.model}
                  type="button"
                  className={focusModel === run.model ? "seg selected" : "seg"}
                  style={
                    focusModel === run.model
                      ? { borderColor: modelColor(run.model), color: modelColor(run.model) }
                      : undefined
                  }
                  onClick={() => {
                    setFocusModel(run.model);
                    setSelected(null);
                    setEdgeSelect(null);
                  }}
                >
                  {modelLabel(run.model)}
                </button>
              ))}
            </div>
          )}

          {kanNet && (
            <KanDiagram
              net={kanNet}
              hovered={edgeHover}
              selected={edgeSelect}
              onHover={setEdgeHover}
              onSelect={setEdgeSelect}
            />
          )}

          {mlpNet && (
            <NetworkDiagram
              net={mlpNet}
              activation={focusRun.model as ActivationId}
              hovered={hovered}
              selected={selected}
              onHover={setHovered}
              onSelect={setSelected}
              onAddLayer={() => shape.length < MAX_LAYERS && update({ shape: [...shape, 3] })}
              onRemoveLayer={() => shape.length > 0 && update({ shape: shape.slice(0, -1) })}
              onAddNeuron={(layer) => {
                if (shape[layer] >= MAX_NEURONS) return;
                const next = [...shape];
                next[layer] += 1;
                update({ shape: next });
              }}
              onRemoveNeuron={(layer) => {
                if (shape[layer] <= 1) return;
                const next = [...shape];
                next[layer] -= 1;
                update({ shape: next });
              }}
            />
          )}

          {kanNet && focusEdge && contribution ? (
            <div className="inspector">
              <div className="inspector-head">
                <strong>
                  {focusEdge.layer === 0 ? "φ" : "ψ"}
                  <sub>{focusEdge.index + 1}</sub> · {focusEdge.layer === 0 ? "input" : "output"} edge
                </strong>
                <span className={edgeSelect ? "pin-badge on" : "pin-badge"}>
                  {edgeSelect ? "pinned" : "hover"}
                </span>
              </div>
              <div className="param-chips">
                <span className="chip">{kanNet.e1[0].c.length} spline coefficients</span>
                <span className="chip">w_b {(focusEdge.layer === 0 ? kanNet.e1 : kanNet.e2)[focusEdge.index].wb.toFixed(2)}</span>
                <span className="chip">w_s {(focusEdge.layer === 0 ? kanNet.e1 : kanNet.e2)[focusEdge.index].ws.toFixed(2)}</span>
              </div>
              <div className="influence">
                <span>Contribution to output</span>
                <strong>{contribution.peak.toFixed(3)}</strong>
                <small>
                  {contribution.band
                    ? `strongest on x ∈ [${contribution.band.lo.toFixed(2)}, ${contribution.band.hi.toFixed(2)}]`
                    : "negligible — this path is not doing much"}
                </small>
              </div>
            </div>
          ) : mlpNet && focusNeuron && contribution ? (
            <NeuronInspector
              net={mlpNet}
              activation={focusRun.model as ActivationId}
              ref_={focusNeuron}
              contribution={contribution}
              localization={localization}
              showDerivative={showDerivative}
              pinned={selected !== null && hovered === null}
            />
          ) : kanNet ? (
            <p className="network-hint">
              KAN puts a learnable function on every <em>edge</em>. Each box is one such
              function; hover for detail, click to pin. Single hidden layer by construction —
              width {kanWidthFor(shape)} is taken from the first hidden layer.
            </p>
          ) : (
            <p className="network-hint">
              Each box plots that neuron&rsquo;s own output across x. Thickness is weight
              magnitude; <span className="swatch-pos">orange</span> positive,{" "}
              <span className="swatch-neg">blue</span> negative. Hover a neuron for detail,
              click to pin it.
            </p>
          )}
        </section>

        <section className="column column-output" aria-label="Output">
          <h2>Output</h2>
          {isStep && (
            <div className="bench-status">
              <span className="badge discontinuity">Discontinuity Benchmark</span>
              <span className="muted">Heaviside — the XNet reference task</span>
            </div>
          )}
          <MetricsPanel rows={metrics} errorTarget={errorTarget} showLocal={isStep} />

          <div className="output-controls">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={showTest}
                onChange={(e) => setConfig({ ...config, showTest: e.target.checked })}
              />
              Show test data
            </label>
            <label className="checkbox">
              Target error
              <select
                value={errorTarget}
                onChange={(e) => setConfig({ ...config, errorTarget: Number(e.target.value) })}
              >
                {ERROR_TARGETS.map((t) => (
                  <option key={t} value={t}>
                    {t.toExponential(0)}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <LossChart series={lossSeries} />
          <ApproximationPlot
            runs={plotRuns}
            target={target}
            train={dataset.train}
            test={dataset.test}
            showTest={showTest}
            overlay={
              contribution
                ? { delta: contribution.delta, band: contribution.band, localization }
                : null
            }
          />
          {isStep && <DiscontinuityInset runs={plotRuns} target={target} />}
        </section>
      </main>

      <PaperBenchmarks />
    </div>
  );
}

const MODES: { id: Mode; label: string }[] = [
  { id: "fit", label: "Function Approximation" },
  { id: "width", label: "Neuron Efficiency" },
  { id: "pde", label: "PDE Demo" },
];

function Masthead({ mode, onMode }: { mode: Mode; onMode: (m: Mode) => void }) {
  return (
    <header className="masthead">
      <div className="masthead-text">
        <p className="eyebrow">Cauchy activation playground</p>
        <h1>
          Watch a network learn the <em>shape</em> of its own nonlinearity.
        </h1>
      </div>
      <div className="mode-switch">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            className={mode === m.id ? "selected" : ""}
            onClick={() => onMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
    </header>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="param-slider">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <strong>{value.toFixed(2)}</strong>
    </label>
  );
}
