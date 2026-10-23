# AuscultTrust

Research code and a demo app for lung-sound classification under stethoscope/device shift. The research studies spectrum correction (SC) and split-conformal prediction. The selected **baseline** is fine-tuned **BEATs + PAFA** without SC; **AuscultTrust** uses the same recipe with **static SC at training and inference**. Exploratory extensions are the equal-weight **E1** and **E1+E3** ensembles.

This is a research prototype for sound classification and screening support. It is not a medical device and does not diagnose disease.

## Project map

| Path | Purpose |
| --- | --- |
| `src/processing/` | Audio, ICBHI/KAUH data, patient splits, spectrum correction |
| `src/training/patchmix_cl/` | Training and export using the Patch-Mix CL reference implementation |
| `src/evaluation/` | Metrics, conformal prediction, screening, device-shift experiments |
| `experiments/train/` | Composable YAML training recipes |
| `scripts/` | Data preparation, campaigns, analysis, and deployment staging |
| `api/` | FastAPI inference service and serving-bundle builder |
| `app/web/` | Vite/TypeScript browser app |
| `package.json` | Root Wrangler CLI dependency for Cloudflare Pages deployment |
| `docs/` | Setup, protocols, decisions, results, and handoffs |
| `outputs/`, `checkpoints/`, `logs/` | Local, git-ignored experiment artifacts |

Training outputs and checkpoints mirror each other by run, model variant, and fit:

```text
outputs/train/<run>/<cell>/<unit>/       reports, predictions, training arguments
checkpoints/train/<run>/<cell>/<unit>/   model weights
```

Serving is separate: `api/bundle.py` assembles selected checkpoints and metadata into `app/bundle/`. The API loads that bundle; it does not scan experiment output folders. `AT_APP_BUNDLE` can point to another bundle directory.

## Setup and training

See [Setup](docs/SETUP.md) for environments, datasets, and reference repositories, and [Quickstart](docs/QUICKSTART.md) for the registered training workflow. The training environment uses Python 3.10; analysis and training dependencies are intentionally separate.

Typical campaign:

```bash
export RUN_ID=$(date +%Y%m%d-%H%M%S)
bash scripts/run.sh screen
bash scripts/run.sh summary
bash scripts/run.sh final
bash scripts/run.sh report
```

Results and checkpoints are local artifacts and are not committed. Keep each experiment's `RUN_ID` fixed when resuming it.

## Inference app

The app has two independently run parts: a FastAPI model service and a static browser UI. Deployment instructions, environment variables, API routes, security settings, and hosting options are in [app/README.md](app/README.md).

To run locally after installing the documented dependencies and preparing `app/bundle/`:

```bash
PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
cd app/web && npm ci && npm run dev
```

The API reports model readiness at `GET /v1/health`; inference is unavailable until a valid bundle and its referenced weights load. Do not expose a public API without choosing appropriate authentication, rate limits, and sample-serving settings.

The final decision and execution provenance are recorded in [the model decision log](docs/decision_logs/2026-10-23-final-model.md).

## Research and documentation

- [Research proposal v10](docs/updates/research-proposal-v10-en.md) (current), the registered [proposal v9](docs/updates/research-proposal-v9-en.md) and [pre-registration](docs/updates/prereg-v8.md) define the study and amendments.
- [Handoff](docs/handoff/HANDOFF.md) gives the current state, where results live and the next steps.
- [Results](report.md) and dated reports in `docs/reports/` contain findings and limitations.
- [Decision logs](docs/decision_logs/) record protocol changes; [handoffs](docs/handoff/) track active work.
- [CLAUDE.md](CLAUDE.md) contains repository-specific working conventions.

The registered core experiments used AST with Patch-Mix CL; the selected demo model is a later model decision. See the dated results and handoff documents for the evidence and current experiment status. Do not treat the demo model as a replacement for registered analyses.
