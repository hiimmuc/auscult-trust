# AuscultTrust

**Spectrum correction + AST (Patch-Mix CL) + split-conformal prediction for lung-sound classification that survives a change of stethoscope.**

KHKT 2026-2027 · school round 17/10/2026 · research cut-off 31/01/2027

> *Nghiên cứu giải pháp cải thiện độ tin cậy của mô hình phân loại âm thanh phổi dưới sự dịch chuyển thiết bị bằng hiệu chỉnh phổ và dự đoán Conformal cho hỗ trợ sàng lọc bệnh đường hô hấp*

---

## Research question

Does spectrum correction (SC) let a benchmark-level lung-sound classifier **keep its predictions and its conformal coverage** when the stethoscope changes, **without labels from the new stethoscope**?

| Piece | What it does |
| --- | --- |
| **SC front end** | Rescales each device's mean spectrum to a shared reference (STFT 1024/512, 16 kHz). Uses unlabelled target audio only. |
| **AST + Patch-Mix CL** | Benchmark-level 4-class cycle classifier (normal / crackle / wheeze / both). |
| **Split-conformal layer** | LAC score, alpha = 0.1. Cycle 4-class view and recording-level screening view (normal / abnormal). |

**Hypotheses** (full definitions: `docs/research-proposal-v9-en.md`, pre-registration `docs/prereg-v8.md` + `docs/prereg-v9-amendment.md`)

| ID | Claim |
| --- | --- |
| G | Stage 1 gate: AuscultTrust is not worse than plain Patch-Mix CL by more than 1.5 Score points (10 seeds). |
| H1 | SC lowers the prediction flip rate between devices on the same sound. |
| H2 | SC lowers \|coverage - 0.90\| on an unseen device, set size not larger (ICBHI leave-one-device-out). |
| H3 | SC explains at least 80 % of the paired log-mel shift (phantom transducers, pending). |

Not claimed: a new algorithm, SOTA Score, disease diagnosis, patient-level guarantees, KAUH software filters as real hardware.

## Pipeline

```
 wav ──► SC (per-device spectrum / reference) ──► log-mel ──► AST + Patch-Mix CL ──► softmax
                                                                                      │
                          calibration patients ──► split conformal (LAC, alpha 0.1) ◄──┘
                                                          │
                              cycle 4-class sets  /  recording screening triage
```

- **Stage 1 (model):** official ICBHI 60/40 split, arms P0 / P1 / P1+P3 / P4, 5 seeds (10 for the gate), grouped 3-fold CV for selection.
- **Stage 2 (experiments):** frozen Stage-1 model. Test 1 ICBHI LODO, Test 2 KAUH filters, Test 3 phantom (pending).

## Quick start

```bash
# 1. data + reference code
bash scripts/download_data.sh          # ICBHI, KAUH, AST weights, Patch-Mix CL repo
python scripts/prepare_data.py         # fix names, build training folder, check datasets

# 2. environments (two: analysis and training, timm==0.4.5 needs Python 3.10)
uv venv .venv && uv pip install -r requirements.txt
uv venv .venv-train --python 3.10 && uv pip install --python .venv-train/bin/python -r requirements-train.txt

# 3. campaign
export RUN_ID=$(date +%Y%m%d-%H%M%S)   # export the same id again to resume
bash scripts/run.sh screen             # 3-fold CV of every arm (SC arms with both sc_mode)
bash scripts/run.sh summary            # ranking, CV epochs, AuscultTrust arm
bash scripts/run.sh final              # refit at CV epoch, test once
bash scripts/run.sh report             # final.md: tables, paired differences, gate G

# 4. tests
uv run --no-project --python .venv/bin/python python -m pytest -q tests
```

Step-by-step: **[docs/SETUP.md](docs/SETUP.md)** (environments, data sources) and **[docs/QUICKSTART.md](docs/QUICKSTART.md)** (campaign, single run, freeze and export).

## Arms

| Arm | Config | Code name |
| --- | --- | --- |
| P0 Patch-Mix CL (baseline) | `experiments/train/baseline.yaml` | `baseline` |
| P1 = P0 + SC | `sc.yaml` | `sc` |
| P1+P3 = SC + random per-bin gain (6 dB SD) | `sc_gain.yaml` | `sc_gain` |
| P4 Freq-MixStyle p = 0.5 (competitor) | `freq_mixstyle.yaml` | `freq_mixstyle` |

All configs layer on `base.yaml`; `sc_static.yaml` switches the SC clipping mode. See `experiments/train/README.md`.

## Layout

```
auscult-trust/
├── src/
│   ├── paths.py, runs.py        data/repo locations; run dirs outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>
│   ├── processing/              ICBHI + KAUH loaders, patient-disjoint splits, SC, gain augmentation
│   ├── training/patchmix_cl/    dataset, loop, reports, frozen-softmax export (model/loss from upstream repo)
│   └── evaluation/              metrics, coverage under shift, conformal layer, export reader
├── experiments/train/           base recipe + one YAML per arm
├── scripts/                     run.sh, download_data.sh, prepare_data.py, sweep.py
├── tests/                       pytest suite
├── notebooks/                   exploration
├── docs/                        setup, quickstart, proposal v9, pre-registration, decision logs
├── report.md                    results log (Phase 0 + current)
├── data/                        README only (data lives next to the repo)
└── checkpoints/ outputs/ logs/  gitignored run artefacts
```

Wrapper directory (not versioned): `../data/raw/` (ICBHI, KAUH, HF_Lung_V1), `../repos/` (read-only references: OPERA, patch-mix_contrastive_learning, SG-SCL), `../archive/` (Phase 0 code).

## Documentation map

| Need | Read |
| --- | --- |
| Install, data sources | [docs/SETUP.md](docs/SETUP.md) |
| Run and compare arms | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Full design, hypotheses | [docs/research-proposal-v9-en.md](docs/research-proposal-v9-en.md) |
| Registered plan | [docs/prereg-v8.md](docs/prereg-v8.md), [docs/prereg-v9-amendment.md](docs/prereg-v9-amendment.md) |
| Vietnamese change list | [docs/v9-change-list-vi.md](docs/v9-change-list-vi.md) |
| Results so far | [report.md](report.md) |
| Config conventions (for agents) | [CLAUDE.md](CLAUDE.md) |

## Rules that matter

- Splits are **patient-disjoint**, always. Calibration patients never train, select or evaluate.
- Tune on grouped CV over training patients, never on test. Test-selected numbers appear only in a labelled literature column.
- SC uses unlabelled target audio only. Target labels are for evaluation.
- Trained arms use at least 5 paired seeds; report mean ± SD; bootstrap by patient.
- Every run logs config, seed, data hash, checkpoint hash and `s_ref` hash. A finished unit is skipped; a different config in an existing unit is refused.
