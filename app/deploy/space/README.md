---
title: AuscultTrust API
emoji: 🩺
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Inference API for AuscultTrust (lung-sound classification with spectrum correction and conformal sets), CPU only.
Routes: `/v1/health`, `/v1/model`, `/v1/samples`, `/v1/infer`; interactive docs at `/docs`.

Settings (Space settings, Variables and secrets):

| Name | Kind | Meaning |
| --- | --- | --- |
| `AT_APP_TOKEN` | secret | access token every route except `/v1/health` requires |
| `AT_APP_ORIGINS` | variable | the web app's origin, e.g. `https://auscult-trust.pages.dev` |
| `AT_WEIGHTS_REPO` | variable | private model repo holding the weights, e.g. `<user>/auscult-trust-weights` |
| `HF_TOKEN` | secret | read token for that repo |

Without the weights repo the API starts and reports `unavailable`; it never returns a made-up prediction.
