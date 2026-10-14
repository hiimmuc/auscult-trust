# Amendment 3 to `prereg-v8.md` (proposal v9): Lung-SRAD reproduction and SC on DASS

**Status: registered 14/10/2026, committed before the first run (smoke or training) of either arm below.**

Nothing here changes a registered rule of Amendment 1 or Amendment 2. Both arms are **exploratory: reported with the same statistics, no verdict**. They do not enter Gate G, H1, H2 or H3.

## 1. Why

Our DASS row of the encoder benchmark is plain DASS with the AST-style 2-D Patch-Mix CL (60.5 ± 1.2, best epoch on test, seeds 0-4, `repro-wave1/encoder_dass+baseline`). Lung-SRAD reports 64.48 ± 0.25 for DASS with two changes. The report of 09/10 marked this as "not run". This campaign (C8) does two things:
- It checks that our pipeline reaches the published Lung-SRAD number when the settings match.
- It tests whether SC still helps on a different and stronger encoder (state-space model, per-cycle normalised input).

## 2. Arms

| Cell | Config | Change from plain DASS |
|---|---|---|
| `lung_srad` | `base.yaml,encoder_dass.yaml,lung_srad.yaml` | Lung-SRAD as in its training script `icbhi_dual_patchmix_gaussian.sh` |
| `lung_srad+sc+sc_static` | same `+ sc.yaml,sc_static.yaml` | Lung-SRAD with SC (static bound) at train and test time |

The Lung-SRAD settings match the script. Where the script leaves an option at its default, the Lung-SRAD `main.py` default is used.
- **Gaussian blur:** inside VMamba blocks 2 and 3 of stage 2, kernel 5, sigma 3. It is part of the network, so it is active at train and test time.
- **Dual-axis Patch-Mix CL:** there are two contrastive losses, each with weight 1.0. One uses a time-only Patch-Mix view (one contiguous segment of time tokens is swapped). The other uses a frequency-only view (one contiguous band of frequency tokens is swapped). Before the projector, the mixed features are detached, as in the Lung-SRAD code. The target is the projector output with gradient (`project_flow`, the Lung-SRAD default).
- **Unchanged from our DASS recipe** (and the same in the Lung-SRAD script): batch 16, temperature 0.2, Adam, lr 5e-5, cosine schedule, weight decay 1e-6, 50 epochs, moving average 0.5, repeat padding, 8 s cycles, 128 mel bins, per-cycle normalisation, SpecAugment, `DASS_medium_v2.pth` AudioSet weights.
- **Known difference:** the classifier is a fresh linear head on the pooled features; Lung-SRAD uses the network's own linear head on the same features. Both are a single linear layer on the same features.

Code: `--gaussian_blur` and `--cl_mode dual` in `src/training/patchmix_cl/config.py`; both are refused for any encoder other than DASS. With neither option set, the network and the training loop are unchanged.

## 3. Protocol

- RUN_ID `lung-srad`, official split, seeds 0-4. As in the paper, the reported Score is that of the best epoch on test (optimistic). The epoch-50 Score is reported next to it.
- Smoke run first: 1 epoch of each cell (`smoke-a`, `queue.sh smoke:smoke_c8`). A cell that fails is not trained.
- Runs on machine A, after the C7 queue.

## 4. Readings (no verdict)

- **Reproduction:** mean ± SD of `lung_srad` against 64.48 ± 0.25. A gap of 1.0 point or less counts as reproduced, the same acceptance window as for AST Patch-Mix CL.
- **Lung-SRAD vs plain DASS:** paired by seed against `repro-wave1/encoder_dass+baseline` (same seeds 0-4, same selection). Plain DASS used the `grad_block` target, so this contrast combines all three changes: blur, dual-axis mixing and target type.
- **SC on DASS:** `lung_srad+sc+sc_static` against `lung_srad`, paired by seed. Reported at the best epoch on test and at epoch 50, with a paired t-interval and a patient bootstrap.
- **Not done here:** LODO and KAUH for these arms. They are considered only if time is left after the 15/11 report.
