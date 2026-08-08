"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ActivationShapeChart from "./components/ActivationShapeChart";
import ApproximationPlot, { type PlotRun } from "./components/ApproximationPlot";
import LossChart, { HISTORY_LIMIT, type LossPoint, type LossSeries } from "./components/LossChart";
import MetricsPanel, { type RunMetrics } from "./components/MetricsPanel";
import NetworkDiagram, { type NeuronRef } from "./components/NetworkDiagram";
import NeuronInspector from "./components/NeuronInspector";
import PaperBenchmarks from "./components/PaperBenchmarks";
import PdeDemo from "./components/PdeDemo";
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
  buildNetwork,
  functionMse,
  loss as computeLoss,
  makeDataset,
  mulberry32,
  neuronContribution,
  parameterCount,
  shuffledOrder,
  targetValue,
  trainEpoch,
  type Network,
  type Regularization,
} from "./lib/model";
import { DEFAULT_CONFIG, readConfig, writeConfig, type PlaygroundConfig } from "./lib/urlState";

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
  activation: ActivationId;
  net: Network;
  history: LossPoint[];
  epochsToTarget: number | null;
  runtimeMs: number;
};

function activationsFor(config: PlaygroundConfig): ActivationId[] {
  return config.compare ? COMPARE_SET : [config.activation];
}

function buildRuns(config: PlaygroundConfig): Run[] {
  return activationsFor(config).map((activation) => ({
    activation,
    // Identical shape and seed across activations, so a comparison is fair.
    net: buildNetwork(config.shape, config.netSeed, {
      l1: config.l1,
      l2: config.l2,
      d: config.d,
    }),
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
  const [focusActivation, setFocusActivation] = useState<ActivationId>("cauchy");

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
      const net = trainEpoch(
        run.net,
        dataset.train,
        {
          activation: run.activation,
          learningRate,
          batchSize,
          regularization,
          regRate,
        },
        order,
      );
      const elapsed = performance.now() - started;
      const fnMse = functionMse(net, run.activation, target);
      return {
        ...run,
        net,
        runtimeMs: run.runtimeMs + elapsed,
        epochsToTarget:
          run.epochsToTarget === null && fnMse <= errorTarget ? nextEpoch : run.epochsToTarget,
        history: [
          ...run.history,
          {
            train: computeLoss(net, dataset.train, run.activation),
            test: computeLoss(net, dataset.test, run.activation),
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
    () => runs.find((r) => r.activation === focusActivation) ?? runs[0],
    [runs, focusActivation],
  );

  const focusNeuron = hovered ?? selected;

  const contribution = useMemo(() => {
    if (!focusNeuron || !focusRun) return null;
    if (focusNeuron.layer >= focusRun.net.length - 1) return null;
    if (focusNeuron.index >= focusRun.net[focusNeuron.layer].length) return null;
    return neuronContribution(
      focusRun.net,
      PLOT_XS,
      focusRun.activation,
      focusNeuron.layer,
      focusNeuron.index,
    );
  }, [focusNeuron, focusRun]);

  const metrics: RunMetrics[] = runs.map((run) => ({
    activation: run.activation,
    trainLoss: computeLoss(run.net, dataset.train, run.activation),
    testLoss: computeLoss(run.net, dataset.test, run.activation),
    functionMse: functionMse(run.net, run.activation, target),
    epochsToTarget: run.epochsToTarget,
    runtimeMs: run.runtimeMs,
    params: parameterCount(shape, run.activation),
  }));

  const plotRuns: PlotRun[] = runs.map((run) => ({
    activation: run.activation,
    net: run.net,
    color: ACTIVATION_COLORS[run.activation],
  }));

  const lossSeries: LossSeries[] = compare
    ? runs.map((run) => ({
        id: run.activation,
        color: ACTIVATION_COLORS[run.activation],
        values: run.history.map((h) => h.train),
      }))
    : [
        { id: "test", color: "#2b4257", values: runs[0]?.history.map((h) => h.test) ?? [], dashed: true },
        {
          id: "train",
          color: ACTIVATION_COLORS[runs[0]?.activation ?? "cauchy"],
          values: runs[0]?.history.map((h) => h.train) ?? [],
        },
      ];

  const primary = ACTIVATIONS.filter((a) => a.primary);
  const secondary = ACTIVATIONS.filter((a) => !a.primary);
  const cauchyVisible = compare || activation === "cauchy";
  const shapeParams = { l1: config.l1, l2: config.l2, d: config.d };

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
                  key={run.activation}
                  type="button"
                  className={focusActivation === run.activation ? "seg selected" : "seg"}
                  style={
                    focusActivation === run.activation
                      ? { borderColor: ACTIVATION_COLORS[run.activation], color: ACTIVATION_COLORS[run.activation] }
                      : undefined
                  }
                  onClick={() => {
                    setFocusActivation(run.activation);
                    setSelected(null);
                  }}
                >
                  {activationMeta(run.activation).label}
                </button>
              ))}
            </div>
          )}

          {focusRun && (
            <NetworkDiagram
              net={focusRun.net}
              activation={focusRun.activation}
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

          {focusRun && focusNeuron && contribution ? (
            <NeuronInspector
              net={focusRun.net}
              activation={focusRun.activation}
              ref_={focusNeuron}
              contribution={contribution}
              showDerivative={showDerivative}
              pinned={selected !== null && hovered === null}
            />
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
          <MetricsPanel rows={metrics} errorTarget={errorTarget} />

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
            overlay={contribution ? { delta: contribution.delta, band: contribution.band } : null}
          />
        </section>
      </main>

      <PaperBenchmarks />
    </div>
  );
}

function Masthead({ mode, onMode }: { mode: "fit" | "pde"; onMode: (m: "fit" | "pde") => void }) {
  return (
    <header className="masthead">
      <div className="masthead-text">
        <p className="eyebrow">Cauchy activation playground</p>
        <h1>
          Watch a network learn the <em>shape</em> of its own nonlinearity.
        </h1>
      </div>
      <div className="mode-switch">
        <button
          type="button"
          className={mode === "fit" ? "selected" : ""}
          onClick={() => onMode("fit")}
        >
          Function Approximation
        </button>
        <button
          type="button"
          className={mode === "pde" ? "selected" : ""}
          onClick={() => onMode("pde")}
        >
          PDE Demo
        </button>
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
