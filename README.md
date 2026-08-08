# Cauchy Activation Playground

An interactive lab, in the spirit of
[TensorFlow Playground](https://playground.tensorflow.org), for the **Cauchy
activation function**:

```
φ(x) = (λ₁x + λ₂) / (x² + d²)
```

Unlike ReLU or tanh, whose shape is fixed, every Cauchy unit carries its own
learnable `λ₁, λ₂, d`, so each neuron adapts the shape of its nonlinearity
during training. The playground puts that side by side with fixed-shape
activations so the difference is visible rather than asserted.

## Quick start

Requires **Node.js 22.13 or newer** (`node -v` to check).

```bash
git clone https://github.com/Beier-H/xnet-playground.git
cd xnet-playground
npm install
npm run dev
```

Then open **http://localhost:3000**. Everything runs in the browser — there is
no backend, no database and no account.

If port 3000 is already taken, pick another:

```bash
npm run dev -- --port 5199
```

Then press **▶** to train. Good first things to try:

1. Select the **step(x)** target (the one badged *XNet*) and hit **Compare
   activations** — Cauchy tracks the discontinuity, Tanh smooths it away, ReLU
   flat-lines.
2. Click any neuron to pin it and see the shape it has learned.
3. Switch to **PDE Demo** for the PINN comparison.

Other commands:

| Command | What it does |
| --- | --- |
| `npm run dev` | Local development server with hot reload |
| `npm run build` | Production build |
| `npm run lint` | ESLint |

## Three modes

**Function Approximation** fits a 1-D target. **Neuron Efficiency** sweeps
network width and plots test MSE against neuron count. **PDE Demo** solves a
small PINN problem. All three compare Cauchy against fixed-shape activations
under identical conditions.

## What you can do

- **Train continuously** — play/pause runs one epoch per animation frame; step
  runs a single epoch.
- **Pick an activation** — Cauchy, ReLU and Tanh are the primary comparison and
  sit on segmented buttons; Sigmoid, GELU, SiLU/Swish and Sine are in the
  dropdown.
- **Shape the Cauchy neuron** — λ₁, λ₂ and d sliders set the initialization and
  redraw the activation chart live. Training then adapts them per neuron.
- **See the activation itself** — a small chart plots φ(z), optionally with
  φ′(z), making the localization and smoothness explicit.
- **Compare activations** — trains Cauchy, ReLU and Tanh on the *same*
  architecture, dataset, seed, optimizer, learning rate and mini-batch order,
  and overlays their loss curves and fits.
- **Measure the trade-off** — train loss, test loss, function MSE against the
  noiseless target, epochs to reach a target error, runtime, and trainable
  parameter count.
- **Inspect a neuron** — hover or click to pin one and see its learned λ₁, λ₂,
  d, its own response curve, its contribution to the output (measured by
  ablation), and the x-band where it matters most. For Cauchy units the
  effective centre μ and width are derived from the existing parameters and
  drawn as a localization band; every other neuron dims.
- **Sweep width** — Neuron Efficiency trains widths 8→256 for Cauchy, ReLU and
  Tanh and plots test MSE against neuron count on a log axis, to test whether
  Cauchy reaches a target error with fewer neurons.
- **Zoom the discontinuity** — with step(x) selected you get a zoomed inset over
  x ∈ [−0.15, 0.15] and a Local MSE column measured only in that window, where
  a global MSE would be swamped by the two flat halves.
- **Change the problem** — six targets including the Heaviside step (the XNet
  benchmark), sin(10πx) and the Runge function; discrete noise levels 0 / 0.05
  / 0.10 / 0.20; train/test split; batch size; L1 or L2 regularization.
- **Share an experiment** — the full configuration lives in the URL hash.

## Code layout

| Path | Role |
| --- | --- |
| `app/lib/activations.ts` | 7 activations with analytic φ, φ′, φ″ and shape gradients |
| `app/lib/model.ts` | Dataset, network, forward pass, analytic backprop, SGD |
| `app/lib/pde.ts` | PINN prototype: residuals, collocation, Adam |
| `app/lib/paperBenchmarks.ts` | Published reference figures (see below) |
| `app/lib/urlState.ts` | Config ⇄ URL hash, with validation of every field |
| `app/components/` | Diagram, plots, activation chart, inspector, metrics, PDE |
| `app/page.tsx` | State, the training loop, and the control surface |

Backpropagation is written by hand — there is no ML dependency — including the
analytic gradients for the Cauchy shape parameters. Every derivative is
verified against central differences (φ′ to ~5e-7, φ″ to ~4e-6, full backprop
to ~9e-8 relative).

## Live Result vs Paper Result

Every number in the interface carries one of two labels and they are never
mixed:

- **Live Result** — computed in your browser, right now, by this code.
- **Paper Result** — quoted from published work, in `app/lib/paperBenchmarks.ts`,
  with its source, task, architecture, metric and reported value.

They are not comparable. The papers train much larger networks, for longer, with
different optimisers, so a paper MSE of 1e-8 next to a live MSE of 1e-3 says
nothing about this implementation. Where a paper did not report a quantity the
cell is blank rather than estimated.

## The PDE demo is a prototype

One hidden layer of 6 units, identical seed and optimizer across activations.
The residual uses analytic φ″; parameter gradients are central finite
differences with Adam. Each problem has a closed-form solution, so absolute
error is real rather than a comparison against another approximation. ReLU has
φ″ = 0 and therefore cannot represent a second-order operator at all — that is
a property of the activation, not a bug.

Results are genuinely mixed, and the UI is built to show that rather than to
argue a conclusion: Cauchy wins clearly on the heat equation and on the
Heaviside step, while Tanh wins on Poisson with fewer parameters.

### Two things worth knowing

- The output neuron is linear, so the network can span any range.
- Rendered SVG coordinates are rounded. `Math.sin`/`exp`/`log` are
  implementation-defined and may differ by an ulp between the SSR runtime and
  the browser, which is enough to trip React hydration.
- **Every chart axis is on a fixed scale, deliberately.** Auto-ranging any of
  them to the current data makes the UI shake during training: the axis is
  recomputed each epoch, so everything already drawn moves. This applies to the
  neuron thumbnails, the loss chart's log axis, its epoch window, and the
  approximation plot's y range. Do not "improve" these by fitting them to the
  data.

---

## Platform notes

Built from the `vinext-starter` template, so a couple of files belong to the
[vinext](https://github.com/cloudflare/vinext) hosting stack rather than to the
playground. Neither affects running it locally.

**There is no authentication.** The playground has no login, no server-side
identity and no user data — it is entirely client-side. The template's optional
ChatGPT sign-in helper (`app/chatgpt-auth.ts`) was unused and has been removed.
If you ever want it back, it read the `oai-authenticated-user-*` request
headers that OpenAI Sites injects; the platform owns the `/signin-with-chatgpt`,
`/signout-with-chatgpt` and `/callback` routes, so an app never implements them
itself.

**`.openai/hosting.json`** — configuration for OpenAI Sites hosting.
`project_id` ties the repo to a hosting project; `d1` and `r2` would declare a
Cloudflare D1 database or R2 bucket, and both are `null` here because the
playground stores nothing. `vite.config.ts` simulates declared bindings during
local development. Kept because deleting it would unlink the hosting project;
it is safe to remove if you deploy somewhere else.

`db/schema.ts` is intentionally empty, `examples/d1/` is an unused optional
sample, and this starter does not use `wrangler.jsonc`.
