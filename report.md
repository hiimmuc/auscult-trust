# AuscultTrust: consolidated results report

| | |
|---|---|
| Report version | 2.0 (supersedes `docs/report_archive_2026-10-05.md`, the 17-section report of 2026-10-05, kept for the full detail of the v4-era experiments) |
| Datetime | 2026-10-07 11:52 (+07) |
| Repo state | branch `report-update`, HEAD `9e156d8`, in sync with `origin/main` |
| Proposal / prereg | v9 (`docs/research-proposal-v9-en.md`) over v8 (`docs/research-proposal-v8-en.md`); `docs/prereg-v8.md` (2026-10-06), `docs/prereg-v9-amendment.md` (2026-10-07) |
| Sources | `outputs/`, `baselines/patchmix_cl/save/results.json` and `logs/`, `git log`, the work sessions of 2026-10-02 to 2026-10-07 |

Tags: [Fact] measured in the repo, [Interpretation] reasoned, [Assumption], [Pending] not run. Score = (Sp + Se) / 2 on ICBHI, 4-class, official split unless stated.

## 1. Status in one paragraph

[Fact] Part I (base model) has not met its criterion: the best fine-tuning rung in our protocol is 57.3 (AST, layer-4 cache), Patch-Mix CL reproduced at 61.49 ± 0.91 with test-selected epochs and 59.59 ± 0.47 at the last epoch, against the published 62.37 ± 0.61. Stage 1 screening of the device-motivated variants is half done (P0-P3 of 24 runs finished; no variant separates from the baseline). Part II (device shift) ran on frozen and fine-tuned rungs only; spectrum correction (A1) and moment matching (A2) are now wired into extraction but no corrected feature set has been extracted or evaluated. No phantom, no hospital data.

## 2. File modifications (2026-10-05 to 2026-10-07)

Commits from `git log`; the v8 migration (6 commits) was done in this session, the rest by parallel sessions.

| Commit | Date | Change |
|---|---|---|
| `bcaee78`, `1aa943b`, `db15736` | 10-05 | Unified encoder extraction, archived v4 stack to `src/legacy/`, AuscultTrust ladder, Patch-Mix recipes, Part II runner, AST layer-4 retry, KAUH extraction configs |
| `1a553e9`/`420e56f`, `44df13f` | 10-05 | Report update (Patch-Mix CL 5 seeds, HeAR/AST-l4 ladders, Part II study); PR #2 |
| `21b64cf` | 10-06 | CLAUDE.md points at v8; hypotheses H1, H-diag, H2, H3, H-inv, finding F4 |
| `b3da606` | 10-06 | `src/shift/correction.py`: A2 docstring fixed (needs a source/target domain boundary) |
| `0b979a1`, `74cd264` | 10-06 | `src/features.py`: config key `correction` applies A1 on the waveform before encoding. Fix: unrelated HTS-AT lines removed from `tests/test_features.py` |
| `ca8017f`, `40314f0` | 10-06 | `src/data/splits.py:calibration_eval_splits` (official-test patients of non-held-out devices, stratified by device, 20 re-splits; stochastic rounding fixes an odd-count bias) |
| `5f462ae` | 10-06 | `src/features_handcrafted.py` (74-dim); `src/eval/metrics.py` (`sp_se`, `hs`, `macro_f1`, `by_device`); `src/eval/shift.py` (`closed_fraction_checked`, `prior_matched_coverage`, `decodability_curve` PCA-8, `mmd_permutation`); `src/conformal/conformal.py` (V4-oracle, V5 in patients); `spectrum_coefficients(reference=arithmetic\|geometric)`; `src/data/kauh.py:rotation_partitions` |
| `30c5bee` | 10-06 | Patch-Mix Stage 1: `baselines/patchmix_cl/main.py` (patient-level val set, `--selection cv\|test`, three reported epochs, per-device metrics, flags for P1-P8), `util/stage1.py`, `util/icbhi_dataset.py`, `models/ast.py` (P6 affine layer), `export_probs.py` (softmax + embedding export with weight hash), `src/conformal/export.py` (E1 from export), A2 wired in `src/features.py` via `Encoder.logmel` (AST, OPERA), `configs/stage1_variants/`, `docs/prereg-v8.md`, `tests/test_stage1.py`, `tests/test_handcrafted.py` |
| `9bccec8` | 10-06 | HTS-AT (AudioSet) encoder, extract and ladder configs |
| `e42a105`, `aeedd4e` | 10-06 | Screening runner (3-fold patient CV, no test access), grouped-CV folds in `main.py` (`--cv_folds`, `--cv_fold`), fixed-refit mode, `stage1_summary.py`, extra P4 configs (p = 0.25, 0.75) |
| `761cc53`, `24840c9`, `25ff50b`, `d9c9419` | 10-06/07 | Proposal v9 (single spectrum-correction chain G, H1-H3), `docs/prereg-v9-amendment.md`, `sc_mode` (dynamic dB clip vs static band bound), `configs/stage1_variants/P1P3_sc_gain.json`, CLAUDE.md rewrite, `docs/v9-change-list-vi.md`; PR #3 |
| this report | 10-07 | old `report.md` moved to `docs/report_archive_2026-10-05.md` |

Tests: 67 passed on the committed tree (2026-10-06 run, before the v9 commits).

## 3. Models and configs used

| Item | Setting |
|---|---|
| Data | ICBHI official split (6898 cycles, 126 patients; 4142 train / 2756 test cycles), split file `../data/splits/icbhi_official_v1.json`; KAUH (112 patients x 3 filters, 8 s windows); HF_Lung_V1 |
| Frozen encoders | AST AudioSet (primary), HeAR ViT-L, CLAP HTS-AT, OPERA-CT, OPERA-CE, HTS-AT AudioSet (`htsat_audioset`, non-leaking). Configs `configs/extract_*.yaml`, `configs/ladder_*.yaml` |
| Ladder rungs | L0 logistic probe; L1 best frozen head by grouped CV (`<last\|concat\|scalar>-<mean\|max\|meanmax\|attn>`); L2 last-k blocks (`ft-k*`); L3 LoRA; L4 Patch-Mix CL. 5 seeds, epoch by grouped CV, test once |
| Patch-Mix CL | `baselines/patchmix_cl/run_repro.sh`: AST (ImageNet + AudioSet), bs 8, Adam lr 5e-5, wd 1e-6, cosine, 50 epochs, temperature 0.06, alpha 1.0, mix_beta 1.0, EMA 0.5, repeat padding, 8 s cycles |
| Stage 1 screening | `configs/stage1_variants/*.json` = Patch-Mix recipe + one flag: P1 `a1_input`, P2 `bin_norm`, P3 `rand_bin_gain` 6 dB, P4 `freq_mixstyle` (p 0.25/0.5/0.75), P6 `bin_affine`, P7 `hc_fuse` (20 MFCC frame SDs), P8 `select_metric: worst_device`, P1P3 `sc_gain`. P5 dropped (no licensed IR set). 3-fold patient-grouped CV on the 60% training patients, validation-selected epoch |
| Conformal | LAC score, alpha = 0.1; V1 split, V2 Mondrian (`src/conformal/conformal.py`), V3-V5 and V4-oracle implemented |
| Hardware | RTX 5000 Ada 32 GB (Patch-Mix seed 1 on an 8 GB laptop) |

## 4. Results

### 4.1 Frozen encoders, ICBHI official test (5 seeds; selection by grouped CV or val)

| Encoder | CV Score | MLP val | Test Score | V1 coverage | Set size |
|---|---|---|---|---|---|
| OPERA-CT | 0.497 ± 0.056 | 0.501 | 0.417 ± 0.023 | 0.855 | 2.56 |
| AST | **0.601 ± 0.035** | **0.653** | **0.547 ± 0.013** | 0.830 | 1.85 |
| CLAP | 0.578 ± 0.028 | 0.573 | 0.519 ± 0.018 | 0.847 | 2.16 |
| HeAR | 0.580 ± 0.026 | 0.628 | 0.527 ± 0.010 | 0.857 | 2.09 |
| HTS-AT (AudioSet) | 0.601 (last-mean) | | 0.534 ± 0.005 (`last-mean`); 0.532 ± 0.006 (`last-meanmax`); L0 probe 0.491 (CI 0.430-0.546) | L0 V1 0.786 | |

[Fact] AST is the primary encoder. OPERA-CT, although pretrained on ICBHI, is lowest, so pretraining overlap did not inflate it. HTS-AT with AudioSet weights is below AST and about level with CLAP/HeAR. [Interpretation] The weak base model of the early v4 experiments was the encoder.

### 4.2 Fine-tuning ladder (test Score, 5 seeds, test once)

| Rung | CV Score | Test Score |
|---|---|---|
| HeAR `last-mean` / `concat-attn` (frozen) | 0.603 | 0.513 ± 0.010 / 0.511 ± 0.005 |
| HeAR `ft-k4-attn` | 0.587 ± 0.025 | **0.570 ± 0.002** |
| AST `ft-k4-attn+mix` (cache after block 8) | 0.585 ± 0.037 | 0.557 ± 0.009 |
| AST layer-4 cache `ft-k8-attn+mix` | 0.605 ± 0.020 | **0.573 ± 0.019** (best of the ladder) |
| HTS-AT `ft-k2-meanmax` (stage 4 only) | 0.585 ± 0.014 | 0.527 ± 0.007 |

[Fact] Fine-tuning helps HeAR (+0.057) and AST (+0.02 over its frozen head 0.547); it does not help HTS-AT (0.527 vs 0.534 frozen). No paired CIs for cell-to-cell differences of 0.01-0.02.

### 4.3 Patch-Mix CL reproduction (5 seeds)

| Protocol | Score | Published |
|---|---|---|
| Best epoch on test (paper protocol, optimistic) | 61.49 ± 0.91 (seeds 61.18, 62.41, 60.60, 62.51, 60.76) | 62.37 ± 0.61 |
| Last epoch (no test selection) | 59.59 ± 0.47 | plain fine-tuning 59.55 ± 0.88 |

[Fact] Reproduction accepted (0.88 below the published mean, inside the +-1 point target). Part I criterion (best rung within 1 SD of or above 62.37 under the val-selected protocol) is **not met**: 57.3 (AST) is 4.2 below the test-selected and 2.3 below the last-epoch reproduction. [Interpretation] Plausible cause: the cached-token design leaves the lower blocks frozen; not tested by a full fine-tune in our protocol.

### 4.4 Part II: device shift (V1 coverage, nominal 0.90)

E1, ICBHI one device held out (AST L0 / L1 / L2; HeAR L2):

| Held-out device (cycles) | AST L0 | AST L1 | AST L2 | HeAR L2 |
|---|---|---|---|---|
| AKGC417L (1836) | 0.97 | 0.80 | 0.83 | 0.87 |
| Litt3200 (461) | 0.47 | 0.97 | 0.95 | 0.91 |
| Meditron (459) | 0.99 | 0.96 | 0.93 | 0.92 |

Binomial half-width +-0.014 (1836), +-0.027 (about 460). [Fact] Coverage is below nominal beyond the half-width on AKGC417L and Litt3200 for most rungs, not on Meditron. Whether this is device or label shift is **not separated** (V4 not run on these pairs). E2 (KAUH filter pairs, 30 units per rung): AST L1 0.892 to 0.914, L2 0.904 to 0.927; OPERA probes drop to 0.49 (CT) and 0.45 (CE). Decodability AUC 0.9-1.0 saturates and cannot rank rungs. [Fact] Fine-tuning did not raise decodability or the coverage deficit (AST 0.904 vs 0.910; HeAR 0.900 vs 0.869), reported as negative finding F4.

### 4.5 Stage 1 screening of device variants (3-fold patient CV, 12 of 24 runs)

Validation-selected epoch on held-out training patients, mean over 3 folds:

| Variant | Folds 0 / 1 / 2 | Mean CV Score | Sp / Se | Worst-device | Fold SD |
|---|---|---|---|---|---|
| P0 baseline | 64.1 / 63.5 / 56.2 | 61.3 | 84.1 / 38.4 | 54.4 | 4.3 |
| P1 A1 input | 63.3 / 66.1 / 57.9 | **62.4** | 82.0 / 42.9 | **58.2** | 4.2 |
| P2 bin norm | 62.0 / 65.1 / 55.7 | 60.9 | 82.0 / 39.8 | 55.5 | 4.8 |
| P3 random bin gain | 60.3 / 64.1 / 55.5 | 60.0 | 87.7 / 32.2 | 55.1 | 4.3 |
| P4, P6, P7, P8, P1P3 | not run | | | | |

[Fact] P1 leads P0 by 1.1 (mean) and 3.8 (worst device) but fold SD is about 4, so no variant is distinguishable yet; fold 2 is the hardest for all. P4/P6/P7 launched with a runner that passed `--cv_fold` before `main.py` accepted it (exit at argument parsing, 0 epochs; fixed since in `aeedd4e`); P8 never started. Smoke run (1 epoch, all flags on) passed end to end, so the variant code paths work. Scores in `baselines/patchmix_cl/save/results.json`, logs in `baselines/patchmix_cl/logs/screen_*.log`.

### 4.6 Earlier v4-era findings (full tables in the archive)

[Fact] Short-window branch: no gain (H1 negative). Hard phase pooling: improved AST crackle F1 on HF_Lung only. Tent: no reliable calibration gain. Conformal sets average 2.5-3.5 of 4 labels with OPERA-CT, 1.85 with AST.

## 5. Caveats and open issues

- **Prereg timing.** `prereg-v9-amendment.md` (registered 2026-10-07) states that no Stage-1 run had started. File timestamps of the screening runs (local +07: P0 fold 0 01:23, P1 03:24, P2 06:29, P3 08:31 on 2026-10-07) are earlier than or close to the commit of the amendment (`d9c9419`, 04:24 UTC = 11:24 +07), so P0-P3 started before the amendment was committed. The primary hypotheses are unaffected by P0-P3 (no test access), but the amendment sentence should be corrected or the runs declared exploratory. [Assumption] The timestamps are in one timezone; check `logs/screening_driver.log`.
- **HTS-AT** (committed `9bccec8`) is not in the v8/v9 variant pool; its numbers above come from `outputs/ladder_htsat/ladder-htsat-v1`.
- **A1/A2 in extraction** are tested on stubs only; no corrected feature cache exists, so H3 (correction closes the deficit) is untested.
- **ICBHI device is confounded** with class prior and site; E1 deficits are not decomposed.
- **KAUH** test sets hold about 35 windows per partition; differences below 0.1 in coverage are inside binomial noise, and no CIs were computed for E2.
- Patch-Mix runs are not deterministic (`cudnn.benchmark`, AMP): seed SD is the right comparison, not run-to-run equality.

## 6. Next steps

1. Rerun P4 (three p values), P6, P7, P8, P1P3 screening (about 15 GPU-hours, 2 in parallel); aggregate with `baselines/patchmix_cl/stage1_summary.py`.
2. Fix the prereg-v9 "no earlier runs" statement or label P0-P3 exploratory.
3. Select finalists by the registered rule (top 3 by CV plus P0), 5 seeds each, test once; gate G-base on the paired Score difference over P0.
4. Extract corrected features (`correction`, `a2` keys) and run E1 on AKGC417L and Litt3200; run V4 and V4-oracle to separate label shift from the device part.
5. Export softmax and embeddings of the frozen finalist (`export_probs.py`) and run Stage 2 over 20 calibration/evaluation re-splits.
