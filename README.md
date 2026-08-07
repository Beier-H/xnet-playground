# Cauchy Activation Playground

An interactive lab, in the spirit of
[TensorFlow Playground](https://playground.tensorflow.org), for the **Cauchy
activation function**:

```
φ(z) = (λ₁z + λ₂) / (z² + σ²)
```

Unlike ReLU or tanh, whose shape is fixed, every Cauchy unit carries its own
learnable `λ₁, λ₂, σ`, so each neuron adapts the shape of its nonlinearity
during training. The playground fits a 1-D target with Cauchy, ReLU, or tanh
units side by side, so the difference is visible rather than asserted.

## What you can do

- **Train continuously** — play/pause runs one epoch per animation frame; step
  runs a single epoch.
- **Edit the architecture live** — up to 4 hidden layers of up to 8 neurons.
- **See inside the network** — every neuron box plots its own response across
  x; edge thickness is weight magnitude, orange positive, blue negative. Hover
  a neuron to trace its curve on the approximation plot.
- **Change the problem** — four targets (smooth, monotone, kinked, and
  discontinuous), adjustable noise, train/test split, and batch size.
- **Regularize** — L1 or L2, to watch a wide net stop chasing noise.
- **Share an experiment** — the full configuration lives in the URL hash, so a
  link reproduces the exact setup.

## Code layout

| Path | Role |
| --- | --- |
| `app/lib/model.ts` | Dataset, network, forward pass, analytic gradients, SGD |
| `app/lib/urlState.ts` | Config ⇄ URL hash, with validation of every field |
| `app/components/NetworkDiagram.tsx` | Architecture, weights, per-neuron curves |
| `app/components/ApproximationPlot.tsx` | Target, samples, and model output |
| `app/components/LossChart.tsx` | Train/test loss on a fixed log scale |
| `app/page.tsx` | State, the training loop, and the control surface |

Backpropagation is written by hand — there is no ML dependency — including the
analytic gradients for the Cauchy shape parameters.

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

This project runs on [vinext](https://github.com/cloudflare/vinext), with
optional Cloudflare D1 and Drizzle support.

## Prerequisites

- Node.js `>=22.13.0`

## Quick Start

```bash
npm install
npm run dev
npm run build
```

This starter does not use `wrangler.jsonc`.

## Included Shape

- edit site code under `app/`
- `.openai/hosting.json` declares optional Sites D1 and R2 bindings
- `vite.config.ts` simulates declared bindings for local development
- `db/schema.ts` starts intentionally empty
- `examples/d1/` contains an optional D1 example surface
- `drizzle.config.ts` supports local migration generation when needed

## Workspace Auth Headers

Signed-in visitors receive both `oai-authenticated-user-id` and `oai-authenticated-user-email`. Private Sites require every visitor to sign in; public Sites may also have anonymous visitors, for whom neither header is present.

The user ID is stable for the same user on the same Site and different across Sites. Email and name are intended for display or contact purposes.

SIWC-authenticated workspace sites may also receive
`oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty
`name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by
`oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const userId = requestHeaders.get("oai-authenticated-user-id");
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Optional Dispatch-Owned ChatGPT Sign-In

Import the ready-to-use helpers from `app/chatgpt-auth.ts` when the site needs
optional or required ChatGPT sign-in:

- Use `getChatGPTUser()` for optional signed-in UI.
- Use `requireChatGPTUser(returnTo)` for server-rendered pages that should send
  anonymous visitors through Sign in with ChatGPT.
- Use `chatGPTSignInPath(returnTo)` and `chatGPTSignOutPath(returnTo)` for
  browser links or actions.
- Pass a same-origin relative `returnTo` path for the destination after sign-in
  or sign-out. The helper validates and safely encodes it.
- Mark protected pages with `export const dynamic = "force-dynamic"` because
  they depend on per-request identity headers.

Dispatch owns `/signin-with-chatgpt`, `/signout-with-chatgpt`, `/callback`, the
OAuth cookies, and identity header injection. Do not implement app routes for
those reserved paths. Routes that do not import and call the helper remain
anonymous-compatible.

SIWC establishes identity only; it does not prove workspace membership. Use the
Sites hosting platform's access policy controls for workspace-wide restrictions,
or enforce explicit server-side membership or allowlist checks.

Use SIWC for account pages, user-specific dashboards, saved records, and write
actions tied to the current ChatGPT user. Leave public content anonymous.

## Useful Commands

- `npm run dev`: start local development
- `npm run build`: verify the vinext build output
- `npm test`: build the starter and verify its rendered loading skeleton
- `npm run db:generate`: generate Drizzle migrations after schema changes

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)
