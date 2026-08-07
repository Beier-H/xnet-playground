"use client";

import { useMemo, useState } from "react";

type Activation = "cauchy" | "relu" | "tanh";
type Target = "sin" | "exp" | "abs";

type Unit = { w: number; b: number; a: number; l1: number; l2: number; sigmaRaw: number };

const samples = Array.from({ length: 81 }, (_, i) => -1 + (2 * i) / 80);

function targetValue(target: Target, x: number) {
  if (target === "sin") return Math.sin(Math.PI * x);
  if (target === "exp") return Math.exp(x) / Math.E;
  return Math.abs(x);
}

function softplus(x: number) { return x > 18 ? x : Math.log1p(Math.exp(x)); }
function sigmoid(x: number) { return 1 / (1 + Math.exp(-x)); }

function seededUnit(index: number): Unit {
  const r = Math.sin((index + 1) * 91.27);
  return {
    w: 0.55 + r * 0.7,
    b: Math.sin((index + 3) * 4.7) * 0.4,
    a: Math.cos((index + 5) * 6.2) * 0.22,
    l1: Math.sin((index + 9) * 2.4) * 0.35,
    l2: Math.cos((index + 11) * 3.9) * 0.55,
    sigmaRaw: -0.2 + Math.sin((index + 2) * 8.1) * 0.25,
  };
}

function makeUnits(width: number) { return Array.from({ length: width }, (_, index) => seededUnit(index)); }

function unitValue(unit: Unit, z: number, activation: Activation) {
  if (activation === "relu") return Math.max(0, z);
  if (activation === "tanh") return Math.tanh(z);
  const sigma = softplus(unit.sigmaRaw) + 0.08;
  return (unit.l1 * z + unit.l2) / (z * z + sigma * sigma);
}

function predict(units: Unit[], outputBias: number, x: number, activation: Activation) {
  return outputBias + units.reduce((sum, unit) => sum + unit.a * unitValue(unit, unit.w * x + unit.b, activation), 0);
}

function mse(units: Unit[], outputBias: number, target: Target, activation: Activation) {
  return samples.reduce((sum, x) => {
    const err = predict(units, outputBias, x, activation) - targetValue(target, x);
    return sum + err * err;
  }, 0) / samples.length;
}

function SvgPlot({ units, outputBias, target, activation }: { units: Unit[]; outputBias: number; target: Target; activation: Activation }) {
  const values = samples.flatMap((x) => [targetValue(target, x), predict(units, outputBias, x, activation)]);
  const yMin = Math.min(-0.2, ...values);
  const yMax = Math.max(1.05, ...values);
  const padding = (yMax - yMin) * 0.12;
  const low = yMin - padding;
  const high = yMax + padding;
  const width = 720;
  const height = 330;
  const xToSvg = (x: number) => 48 + ((x + 1) / 2) * (width - 68);
  const yToSvg = (y: number) => 18 + ((high - y) / (high - low)) * (height - 58);
  const toPath = (fn: (x: number) => number) => samples.map((x, index) => `${index === 0 ? "M" : "L"}${xToSvg(x).toFixed(2)},${yToSvg(fn(x)).toFixed(2)}`).join(" ");
  const targetPath = toPath((x) => targetValue(target, x));
  const modelPath = toPath((x) => predict(units, outputBias, x, activation));
  const yTicks = [low, (low + high) / 2, high];

  return (
    <svg className="plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Target function and neural-network prediction">
      <rect x="0" y="0" width={width} height={height} rx="14" className="plot-background" />
      {[-1, -0.5, 0, 0.5, 1].map((tick) => <g key={tick}><line x1={xToSvg(tick)} x2={xToSvg(tick)} y1="18" y2={height - 40} className="grid" /><text x={xToSvg(tick)} y={height - 18} textAnchor="middle" className="axis-label">{tick}</text></g>)}
      {yTicks.map((tick, index) => <g key={index}><line x1="48" x2={width - 20} y1={yToSvg(tick)} y2={yToSvg(tick)} className="grid" /><text x="38" y={yToSvg(tick) + 4} textAnchor="end" className="axis-label">{tick.toFixed(1)}</text></g>)}
      <line x1="48" x2={width - 20} y1={yToSvg(0)} y2={yToSvg(0)} className="zero-line" />
      <path d={targetPath} className="target-line" /><path d={modelPath} className="model-line" />
      <g transform={`translate(${width - 182},30)`}><line x1="0" x2="26" y1="0" y2="0" className="target-line" /><text x="34" y="4" className="legend-label">target</text><line x1="0" x2="26" y1="20" y2="20" className="model-line" /><text x="34" y="24" className="legend-label">network</text></g>
    </svg>
  );
}

export default function Home() {
  const [activation, setActivation] = useState<Activation>("cauchy");
  const [target, setTarget] = useState<Target>("sin");
  const [width, setWidth] = useState(8);
  const [learningRate, setLearningRate] = useState(0.035);
  const [units, setUnits] = useState<Unit[]>(() => makeUnits(8));
  const [outputBias, setOutputBias] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([mse(makeUnits(8), 0, "sin", "cauchy")]);
  const loss = useMemo(() => mse(units, outputBias, target, activation), [units, outputBias, target, activation]);

  function reset(nextWidth = width, nextTarget = target, nextActivation = activation) {
    const fresh = makeUnits(nextWidth);
    setUnits(fresh); setOutputBias(0); setEpoch(0); setLossHistory([mse(fresh, 0, nextTarget, nextActivation)]);
  }
  function updateWidth(nextWidth: number) { setWidth(nextWidth); reset(nextWidth); }
  function changeTarget(nextTarget: Target) { setTarget(nextTarget); reset(width, nextTarget, activation); }
  function changeActivation(nextActivation: Activation) { setActivation(nextActivation); reset(width, target, nextActivation); }

  function train(iterations = 200) {
    let nextUnits = units.map((unit) => ({ ...unit }));
    let nextBias = outputBias;
    const nextHistory = [...lossHistory];
    for (let step = 0; step < iterations; step += 1) {
      const gradients = nextUnits.map(() => ({ w: 0, b: 0, a: 0, l1: 0, l2: 0, sigmaRaw: 0 }));
      let biasGradient = 0;
      for (const x of samples) {
        const hidden = nextUnits.map((unit) => { const z = unit.w * x + unit.b; return { z, value: unitValue(unit, z, activation) }; });
        const output = nextBias + hidden.reduce((sum, item, index) => sum + nextUnits[index].a * item.value, 0);
        const dLossOutput = (2 * (output - targetValue(target, x))) / samples.length;
        biasGradient += dLossOutput;
        nextUnits.forEach((unit, index) => {
          const { z, value } = hidden[index];
          gradients[index].a += dLossOutput * value;
          const dLossValue = dLossOutput * unit.a;
          let dValueZ = 0;
          if (activation === "relu") dValueZ = z > 0 ? 1 : 0;
          else if (activation === "tanh") dValueZ = 1 - value * value;
          else {
            const sigma = softplus(unit.sigmaRaw) + 0.08;
            const denominator = z * z + sigma * sigma;
            const numerator = unit.l1 * z + unit.l2;
            dValueZ = (unit.l1 * denominator - 2 * z * numerator) / (denominator * denominator);
            gradients[index].l1 += dLossValue * z / denominator;
            gradients[index].l2 += dLossValue / denominator;
            gradients[index].sigmaRaw += dLossValue * (-2 * sigma * numerator) / (denominator * denominator) * sigmoid(unit.sigmaRaw);
          }
          const dLossZ = dLossValue * dValueZ;
          gradients[index].w += dLossZ * x; gradients[index].b += dLossZ;
        });
      }
      nextUnits = nextUnits.map((unit, index) => ({
        w: unit.w - learningRate * gradients[index].w,
        b: unit.b - learningRate * gradients[index].b,
        a: unit.a - learningRate * gradients[index].a,
        l1: unit.l1 - learningRate * gradients[index].l1,
        l2: unit.l2 - learningRate * gradients[index].l2,
        sigmaRaw: Math.max(-6, Math.min(5, unit.sigmaRaw - learningRate * gradients[index].sigmaRaw)),
      }));
      nextBias -= learningRate * biasGradient;
      if ((step + 1) % 20 === 0) nextHistory.push(mse(nextUnits, nextBias, target, activation));
    }
    setUnits(nextUnits); setOutputBias(nextBias); setEpoch((value) => value + iterations); setLossHistory(nextHistory.slice(-32));
  }

  const recentLoss = lossHistory.at(-1) ?? loss;
  const maxLoss = Math.max(...lossHistory, recentLoss, 0.0001);
  const lossPoints = lossHistory.map((item, index) => {
    const x = lossHistory.length === 1 ? 0 : (index / (lossHistory.length - 1)) * 100;
    const y = 100 - (Math.log1p(item) / Math.log1p(maxLoss)) * 88;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return <main className="lab-shell">
    <section className="hero"><p className="eyebrow">Cauchy activation research lab</p><h1>Compare function approximation, one neuron family at a time.</h1><p>Fit the same one-dimensional target with Cauchy, ReLU, or tanh units. The curves make width, training loss, and approximation error visible.</p></section>
    <section className="control-panel" aria-label="Experiment controls">
      <label>Target function<select value={target} onChange={(event) => changeTarget(event.target.value as Target)}><option value="sin">sin(πx) — analytic</option><option value="exp">eˣ / e — analytic</option><option value="abs">|x| — non-analytic at 0</option></select></label>
      <label>Activation<select value={activation} onChange={(event) => changeActivation(event.target.value as Activation)}><option value="cauchy">Cauchy</option><option value="relu">ReLU</option><option value="tanh">tanh</option></select></label>
      <label>Hidden neurons <strong>{width}</strong><input type="range" min="2" max="24" step="1" value={width} onChange={(event) => updateWidth(Number(event.target.value))} /></label>
      <label>Learning rate <strong>{learningRate.toFixed(3)}</strong><input type="range" min="0.005" max="0.08" step="0.005" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
      <div className="actions"><button className="primary" onClick={() => train(200)}>Train 200 steps</button><button onClick={() => reset()}>Reset</button></div>
    </section>
    <section className="workspace"><div className="plot-card"><div className="card-heading"><div><p className="eyebrow">Approximation</p><h2>Target vs. learned network</h2></div><div className="metric"><span>current MSE</span><strong>{loss.toExponential(2)}</strong></div></div><SvgPlot units={units} outputBias={outputBias} target={target} activation={activation} /></div>
      <aside className="side-panel"><div className="metric-card"><span>Training epoch</span><strong>{epoch.toLocaleString()}</strong><small>Each click runs 200 full-batch gradient updates.</small></div><div className="metric-card"><span>Model</span><strong>{width} hidden units</strong><small>{activation === "cauchy" ? "Each unit learns its Cauchy shape parameters." : "The activation shape is fixed."}</small></div><div className="loss-card"><span>Loss history</span><svg viewBox="0 0 100 100" role="img" aria-label="Recent training loss history"><polyline points={lossPoints} className="loss-line" /></svg><small>latest: {recentLoss.toExponential(2)}</small></div></aside>
    </section>
    <section className="formula-strip"><div><span>Neuron input</span><strong>z = wᵀx + b</strong></div><div><span>Cauchy activation</span><strong>φ(z) = (λ₁z + λ₂)/(z² + σ²)</strong></div><div><span>Network output</span><strong>F(x) = Σ aⱼφⱼ(wⱼᵀx + bⱼ) + c</strong></div></section>
  </main>;
}
