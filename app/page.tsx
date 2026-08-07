"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ApproximationPlot from "./components/ApproximationPlot";
import LossChart, { HISTORY_LIMIT, type LossPoint } from "./components/LossChart";
import NetworkDiagram, { type NeuronRef } from "./components/NetworkDiagram";
import {
  ACTIVATIONS,
  BATCH_SIZES,
  LEARNING_RATES,
  MAX_LAYERS,
  MAX_NEURONS,
  PLOT_XS,
  REG_RATES,
  TARGETS,
  buildNetwork,
  loss as computeLoss,
  makeDataset,
  mulberry32,
  neuronCurves,
  networkShape,
  targetValue,
  trainEpoch,
  type Activation,
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
] as const satisfies readonly (keyof PlaygroundConfig)[];

export default function Playground() {
  const [config, setConfig] = useState<PlaygroundConfig>(DEFAULT_CONFIG);
  const [net, setNet] = useState<Network>(() => buildNetwork(DEFAULT_CONFIG.shape, DEFAULT_CONFIG.netSeed));
  const [epoch, setEpoch] = useState(0);
  const [history, setHistory] = useState<LossPoint[]>([]);
  const [playing, setPlaying] = useState(false);
  const [hovered, setHovered] = useState<NeuronRef | null>(null);

  // The live network is mirrored into a ref so the training loop can read it
  // without re-subscribing the animation frame on every epoch.
  const netRef = useRef(net);
  useEffect(() => {
    netRef.current = net;
  }, [net]);
  // A single shuffling stream, kept out of React state so it never triggers a render.
  const shuffleRng = useRef(mulberry32(7));

  const {
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
  } = config;

  // Restore a shared experiment from the URL once, after mount.
  //
  // This has to be an effect rather than a lazy initializer: `location.hash`
  // does not exist during the server render, so seeding state from it directly
  // would make the hydrating client render disagree with the server HTML. The
  // set-state-in-effect rule is aimed at cascading renders; here it is a single
  // batched update that runs exactly once, which is the accepted trade-off for
  // reading browser-only state under SSR.
  useEffect(() => {
    const fromUrl = readConfig(window.location.hash);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConfig(fromUrl);
    setNet(buildNetwork(fromUrl.shape, fromUrl.netSeed));
    setEpoch(0);
    setHistory([]);
  }, []);

  useEffect(() => {
    window.history.replaceState(null, "", writeConfig(config));
  }, [config]);

  const dataset = useMemo(
    () => makeDataset(target, noise, percentTrain, dataSeed),
    [target, noise, percentTrain, dataSeed],
  );

  const trainLoss = useMemo(
    () => computeLoss(net, dataset.train, activation),
    [net, dataset.train, activation],
  );
  const testLoss = useMemo(
    () => computeLoss(net, dataset.test, activation),
    [net, dataset.test, activation],
  );

  const restart = useCallback((next: PlaygroundConfig) => {
    setPlaying(false);
    setConfig(next);
    setNet(buildNetwork(next.shape, next.netSeed));
    setEpoch(0);
    setHistory([]);
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
    const next = trainEpoch(
      netRef.current,
      dataset.train,
      { activation, learningRate, batchSize, regularization, regRate },
      shuffleRng.current,
    );
    netRef.current = next;
    setNet(next);
    setEpoch((value) => value + 1);
    setHistory((current) =>
      [
        ...current,
        {
          train: computeLoss(next, dataset.train, activation),
          test: computeLoss(next, dataset.test, activation),
        },
      ].slice(-HISTORY_LIMIT),
    );
  }, [dataset, activation, learningRate, batchSize, regularization, regRate]);

  // Continuous training: one epoch per animation frame while playing.
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

  const hoveredCurve = useMemo(() => {
    if (!hovered) return null;
    const curves = neuronCurves(net, PLOT_XS, activation);
    return curves[hovered.layer]?.[hovered.index] ?? null;
  }, [hovered, net, activation]);

  const hiddenCount = shape.length;

  return (
    <div className="page">
      <header className="masthead">
        <p className="eyebrow">Cauchy activation playground</p>
        <h1>
          Watch a network learn the <em>shape</em> of its own nonlinearity.
        </h1>
        <p className="lede">
          Fit a 1-D target with Cauchy, ReLU, or tanh units. Cauchy neurons carry learnable
          shape parameters — φ(z) = (λ₁z + λ₂)/(z² + σ²) — so each one adapts its own curve
          instead of reusing a fixed kink.
        </p>
      </header>

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

        <label className="field">
          Learning rate
          <select
            value={learningRate}
            onChange={(event) => update({ learningRate: Number(event.target.value) })}
          >
            {LEARNING_RATES.map((rate) => (
              <option key={rate} value={rate}>
                {rate}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Activation
          <select
            value={activation}
            onChange={(event) => update({ activation: event.target.value as Activation })}
          >
            {ACTIVATIONS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          Regularization
          <select
            value={regularization}
            onChange={(event) =>
              update({ regularization: event.target.value as Regularization })
            }
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
            onChange={(event) => update({ regRate: Number(event.target.value) })}
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
          <p className="column-note">Which target should the network fit?</p>
          <div className="target-grid">
            {TARGETS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`target-chip ${target === item.id ? "selected" : ""}`}
                onClick={() => update({ target: item.id })}
                title={item.note}
              >
                <svg viewBox="0 0 60 34" aria-hidden="true">
                  <polyline
                    points={PLOT_XS.filter((_, i) => i % 4 === 0)
                      .map((x, i, arr) => {
                        const y = targetValue(item.id, x);
                        return `${((i / (arr.length - 1)) * 56 + 2).toFixed(1)},${(17 - y * 12).toFixed(1)}`;
                      })
                      .join(" ")}
                  />
                </svg>
                <span>{item.label}</span>
              </button>
            ))}
          </div>

          <label className="slider">
            Ratio of training to test data: <strong>{percentTrain}%</strong>
            <input
              type="range"
              min="10"
              max="90"
              step="10"
              value={percentTrain}
              onChange={(event) => update({ percentTrain: Number(event.target.value) })}
            />
          </label>
          <label className="slider">
            Noise: <strong>{noise.toFixed(2)}</strong>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              value={noise}
              onChange={(event) => update({ noise: Number(event.target.value) })}
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
              onChange={(event) => update({ batchSize: BATCH_SIZES[Number(event.target.value)] })}
            />
          </label>
          <button
            type="button"
            className="wide-button"
            onClick={() => update({ dataSeed: Math.floor(Math.random() * 1e9) })}
          >
            Regenerate
          </button>
        </section>

        <section className="column column-network" aria-label="Network">
          <h2>Network</h2>
          <NetworkDiagram
            net={net}
            activation={activation}
            hovered={hovered}
            onHover={setHovered}
            onAddLayer={() =>
              hiddenCount < MAX_LAYERS && update({ shape: [...shape, 3] })
            }
            onRemoveLayer={() => hiddenCount > 0 && update({ shape: shape.slice(0, -1) })}
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
        </section>

        <section className="column column-output" aria-label="Output">
          <h2>Output</h2>
          <div className="loss-readout">
            <div>
              <span>Test loss</span>
              <strong className="loss-test-value">{testLoss.toFixed(4)}</strong>
            </div>
            <div>
              <span>Training loss</span>
              <strong className="loss-train-value">{trainLoss.toFixed(4)}</strong>
            </div>
          </div>
          <LossChart history={history} />

          <ApproximationPlot
            net={net}
            activation={activation}
            target={target}
            train={dataset.train}
            test={dataset.test}
            showTest={showTest}
            hoveredCurve={hoveredCurve}
          />

          <label className="checkbox">
            <input
              type="checkbox"
              checked={showTest}
              onChange={(event) => setConfig((c) => ({ ...c, showTest: event.target.checked }))}
            />
            Show test data
          </label>
        </section>
      </main>

      <footer className="formula-strip">
        <div>
          <span>Neuron input</span>
          <strong>z = wᵀa + b</strong>
        </div>
        <div>
          <span>Cauchy activation</span>
          <strong>φ(z) = (λ₁z + λ₂)/(z² + σ²)</strong>
        </div>
        <div>
          <span>Learned per neuron</span>
          <strong>{activation === "cauchy" ? "w, b, λ₁, λ₂, σ" : "w, b only"}</strong>
        </div>
        <div>
          <span>Network</span>
          <strong>
            {networkShape(net).join(" → ") || "no hidden layer"} → 1
          </strong>
        </div>
      </footer>
    </div>
  );
}
