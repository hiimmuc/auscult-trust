# Tier 0 report: RQ1 (H1, H2) and RQ3 (H4)

Date: 2026-10-03. Spec: `../docs/research-proposal-lung-sound-v4-en.md`. All numbers: mean ± SD over 5 seeds; CI = paired patient-bootstrap 95% CI of the seed-averaged difference (1000 resamples).

## 1. Setup

- **Data.** ICBHI official split: test = official test (2756 cycles, 49 patients); train 65 / val 12 patients carved from official train. Two patients on both official sides stay in test; their 13 train recordings are dropped. HF_Lung_V1: official train/test (date groups disjoint), val = 15% of train date groups.
- **Encoder.** OPERA-CT, frozen. Input matches OPERA pretraining: raw audio, mel 50–8000 Hz, cycles repeat-padded to 8 s (cache `opera_ct_v2`).
- **Head.** Per-frame Linear+LN+GELU, mean or phase pooling, LN+Linear. AdamW, 50 epochs, epoch picked on val ICBHI Score. No class weights (they lowered grouped-CV score on train patients by 0.05–0.10).
- **Phase masks (ICBHI).** BiGRU detector on HF_Lung (all 7809 train files; test inspiration F1 0.71, expiration F1 0.36). HF_Lung labels expiration in only ~half of the breaths, so each ICBHI cycle gets one inspiration + one expiration segment from p(inspiration). 88% of cycles decode inspiration-first.
- **Changes from v1** (random 88/19/19 split, zero padding, 2 kHz mel; archived in `outputs/tier0_icbhi_baseline/20261003-005846_v1-random-split/`). v1 and v2 scores are not comparable: the official split is harder.

## 2. RQ1 on ICBHI: 2×2 ablation + controls

Run `outputs/tier0_icbhi_baseline/20261003-020002_ablation-2x2-v2/`, follow-up `outputs/tier0_icbhi_baseline/20261003-143337_h1-short-vs-long/`.

| Cell | ICBHI Score | Crackle F1 | Wheeze F1 | Δ Score vs FM [CI] | Δ Crackle F1 vs FM [CI] |
|---|---|---|---|---|---|
| FM only | 0.417 ± 0.023 | 0.249 ± 0.032 | 0.180 ± 0.069 | — | — |
| + branch 5 ms | 0.472 ± 0.017 | 0.306 ± 0.035 | 0.153 ± 0.034 | +0.056 [+0.025, +0.084] | +0.057 [−0.009, +0.114] |
| + branch 64 ms (control) | 0.458 ± 0.016 | 0.248 ± 0.065 | 0.162 ± 0.036 | +0.041 [+0.019, +0.064] | −0.001 [−0.027, +0.030] |
| + phase | 0.417 ± 0.024 | 0.240 ± 0.041 | 0.165 ± 0.046 | +0.001 [−0.013, +0.012] | −0.009 [−0.030, +0.009] |
| + branch + phase | 0.475 ± 0.014 | 0.281 ± 0.065 | 0.167 ± 0.016 | +0.059 [+0.032, +0.083] | +0.032 [−0.021, +0.075] |
| + branch + shuffled phase | 0.468 ± 0.025 | 0.348 ± 0.030 | 0.187 ± 0.037 | +0.051 [+0.017, +0.083] | +0.099 [+0.025, +0.159] |

**H1 direct test, 5 ms minus 64 ms branch:** ICBHI Score +0.015 [−0.008, +0.036]; crackle F1 +0.058 [−0.019, +0.116]; wheeze F1 −0.009 [−0.043, +0.018].

## 3. RQ1 on HF_Lung: true vs predicted phase (H2 mechanism control)

Run `outputs/hf_phase_ablation/20261003-143739_h2-true-phase/`. 8 s windows, 4 classes (crackle = any D event; CAS = any Wheeze/Stridor/Rhonchi). No branch. Mask = inspiration vs rest. Predicted masks are cross-fitted (no window is masked by a detector that saw it); test frame agreement with the true mask 0.874.

| Cell | ICBHI Score | Crackle F1 | Wheeze F1 | Δ Crackle F1 vs FM [CI] |
|---|---|---|---|---|
| FM only | 0.478 ± 0.010 | 0.129 ± 0.068 | 0.115 ± 0.041 | — |
| + true phase | 0.473 ± 0.035 | 0.152 ± 0.098 | 0.135 ± 0.025 | +0.023 [+0.003, +0.041] |
| + predicted phase | 0.473 ± 0.022 | 0.109 ± 0.064 | 0.137 ± 0.023 | −0.020 [−0.051, +0.003] |
| + shuffled true phase | 0.469 ± 0.032 | 0.150 ± 0.086 | 0.120 ± 0.032 | +0.021 [+0.008, +0.036] |

True minus predicted phase: crackle F1 +0.044 [+0.020, +0.069]; ICBHI Score −0.000 [−0.005, +0.009].

## 4. RQ3 on ICBHI: conformal under the official-split shift, ± Tent (H4)

Run `outputs/rq3_conformal/20261003-151124_conformal-tent/`. Plain FM (per G1). Source calibration = val. Official test split per seed into 10 labelled patients (V5 k-shot calibration only) and 39 evaluation patients (all variants). Tent updates LayerNorm affine parameters on unlabelled evaluation inputs, one pass; calibration scores recomputed after TTA. Nominal coverage 0.90.

| Variant | Coverage | Set size (of 4) | Worst-class coverage |
|---|---|---|---|
| V1 split | 0.855 ± 0.020 | 2.55 ± 0.16 | 0.32 ± 0.07 |
| V2 Mondrian | 0.921 ± 0.009 | 3.49 ± 0.03 | 0.78 ± 0.05 |
| V3 weighted (covariate) | 0.889 ± 0.019 | 3.18 ± 0.09 | 0.53 ± 0.05 |
| V4 label shift (BBSE) | 0.778 ± 0.035 | 2.15 ± 0.21 | 0.18 ± 0.08 |
| V5 k-shot (10 patients) | 0.903 ± 0.033 | 2.91 ± 0.26 | 0.44 ± 0.14 |

No TTA: ICBHI Score 0.418 ± 0.023, ECE 0.247 ± 0.041. Tent at lr 0.001 changes every metric by at most 0.016 (label-shift set size), coverage by at most 0.001.

Tent at lr 0.01, paired per-seed change vs no TTA (mean; sign over 5 seeds):

| Metric | Δ mean | Seeds with this sign |
|---|---|---|
| ICBHI Score | +0.006 | 4/5 up, 1 zero |
| ECE | +0.003 | 3/5 up |
| V1 split coverage | −0.008 | 5/5 down |
| V1 split set size | −0.064 | 5/5 down |
| V4 label-shift coverage | −0.020 | 5/5 down |
| V3 weighted coverage | −0.003 | 4/5 down |
| V5 k-shot coverage | −0.001 | mixed |

## 5. Findings

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: short window raises crackle F1 more than wheeze F1 | **Not shown** (G1) | Branch gain is real (+0.056 Score) but mostly also present at 64 ms. Short-specific crackle gain +0.058, CI includes 0, below seed SD 0.065 |
| H2: phase pooling raises crackle F1; shuffled removes it | **Not supported** | ICBHI: phase ±0, shuffled ≥ real. HF_Lung with true phase: +0.023 vs shuffled +0.021, both below seed SD |
| H4: TTA raises accuracy, moves coverage off nominal; recalibration restores it | **Direction consistent, effect too small** | Tent lr 0.01: +0.006 Score, split coverage −0.008 in 5/5 seeds; both below seed SD. V3/V5 barely move |
| Shift alone breaks split conformal | **Yes** | V1 0.855 < 0.90 without any TTA; V3 (0.889) and V5 (0.903) restore marginal coverage, at larger sets |

## 6. What holds and what breaks

_Superseded for H1, H2 and encoder choice by sections 13-14 (four encoders)._

Status of every hypothesis and design assumption of the proposal against current Tier 0 evidence.

| Claim (proposal section) | Status | Evidence |
|---|---|---|
| H1 short window raises crackle F1 more than wheeze F1 (§3) | **Breaks** (G1) | Short vs 64 ms: crackle F1 +0.058, CI includes 0, below seed SD. Direction holds |
| H2 phase pooling raises crackle F1; shuffled removes the gain (§3) | **Breaks** | Shuffled = true phase, on ICBHI and on HF_Lung with true labels |
| H3 crackle recall drops more than wheeze under device shift (§3) | Not tested | Needs KAUH loader and device hold-out |
| H4 TTA raises accuracy and moves coverage; recalibration restores it (§3) | **Partly holds**, too small | Same direction in 5/5 seeds, effect below seed SD. Recalibration (V3, V5) does restore coverage |
| H5, H6 (Tier 2) | Not testable yet | Patient data |
| P1 64 ms window smears crackles (§1) | Not supported | Extra spectral input helps at 64 ms as much as at 5 ms for ICBHI Score |
| P2 shift breaks reliability (§1) | **Holds** | Split conformal 0.855 vs 0.90 on the official split, no TTA needed |
| Conformal validity breaks under shift unless corrected (§1, [18]–[20]) | **Holds** | V1 0.855; V3 0.889, V5 0.903 |
| Entropy-minimising TTA degrades calibration (§1, [Hypothesis] for lung sounds) | Not shown | ECE +0.003, 3/5 seeds |
| V4 label-shift correction is needed because device is confounded with class prior (§4.3) | **Breaks** in practice | BBSE lowers coverage to 0.778; weights need an accurate classifier and the shift is not pure label shift |
| V2 Mondrian gives per-class coverage (§4.3) | Holds, uninformative | Worst-class coverage 0.78 but 3.49 of 4 labels per set |
| Phase detector reaches published F1 0.79–0.87 (§4.2) | **Breaks** | Frame F1: inspiration 0.71, expiration 0.36 (HF_Lung expiration labels incomplete; 0.25 s frames) |
| [Assumption] one ICBHI cycle = inspiration + expiration (this work) | Holds | 88% of cycles decode inspiration-first |
| Frozen FM: low compute, low overfit (§4.2) | **Holds for AST/CLAP, breaks for OPERA-CT** | Frozen test Score: OPERA-CT 0.417, CLAP 0.519, AST 0.547 (§8); pooling, context and layer tricks do not lift OPERA-CT (§7) |
| Encoder choice: OPERA-CT primary, AST/CLAP only as leakage control (§4.2, B4) | **Breaks** | Non-OPERA encoders are better by 0.10-0.13 Score; G2 rule "report non-OPERA as primary" applies in effect |
| Baselines reproduce within ±1 point (§4.2) | Pending | Patch-Mix CL run in progress (§8) |

## 7. Improving the base model: screening

Grouped 5-fold CV over the 77 official-train patients only (official test untouched); logistic probe on standardised, PCA-256 frozen OPERA-CT v2 features. Mean ± SD over folds.

| Change | CV ICBHI Score | Crackle F1 |
|---|---|---|
| Mean-pooled final embedding (baseline) | 0.497 ± 0.050 | 0.340 |
| + std pooling / + max pooling | 0.488 / 0.501 | — |
| + recording-mean context / minus recording mean | 0.481 / 0.479 | — |
| + previous and next cycle of the same recording | 0.487 | — |
| ICBHI-optimal class bias (fitted on training fold) | 0.498 ± 0.058 | — |
| HTS-AT stage 1 / 2 / 3 / 4 (mean-pooled) | 0.505 / 0.506 / 0.488 / 0.497 | 0.334 / 0.376 / 0.360 / 0.340 |
| Stages 1–4 concatenated | 0.476 ± 0.042 | 0.340 |
| Inverse-frequency class weights (MLP head, earlier screen) | −0.05 to −0.10 | — |

No change moves CV Score beyond ±0.01 of baseline (fold SD 0.05). The ceiling is the frozen representation, not pooling, context or the decision rule. Stage 2 has higher crackle F1 (+0.036), a weak hint that finer encoder time resolution carries crackle information.

Fine-tuning breaks the ceiling: Patch-Mix CL (AST, fine-tuned) reaches 0.570 on the official test after its first epoch (§9), against 0.42 for the frozen probe.

### Proposals, ranked

1. **Switch the base model to a fine-tuned AST, then freeze it.** Fine-tune AST (AudioSet + ImageNet, Patch-Mix CL recipe) on official train with val-based epoch selection, then freeze it and run H1–H4 on top. AST was not pretrained on ICBHI or HF_Lung, so it also serves as the non-OPERA control for G2. It changes the proposal's "frozen FM" design; that change is justified by the 0.15 Score gap.
2. **Retest H1 on the stronger base.** Run the 5 ms and 64 ms branches next to AST patch tokens. The crackle-specific signal (+0.058) may cross seed SD once the base classifier separates classes.
3. **Report split variance.** Repeat grouped CV over train patients next to the official test, so effects are judged against split variance, not only seed SD.
4. **Test H4 under a stronger shift and a larger adaptable part.** KAUH filter renderings and ICBHI device hold-out; adapt the encoder's norm layers too (allowed once the encoder is the fine-tuned AST, not OPERA), and add the calibration-aware TTA variant.
5. **Smaller conformal sets.** A better base classifier is the main lever. Also try APS/RAPS scores and a two-level label set (normal vs abnormal, then type) so sets stay informative.
6. **Drop or demote phase pooling.** It failed with true labels. If kept, test phase as a feature (inspiratory vs expiratory crackle rate), not as pooling.

## 8. Encoder comparison (frozen, plain FM)

Run `outputs/encoder_compare/20261003-222400_opera-ast-clap/`. Same protocol for every encoder: 32 frames per 8 s cycle, mean-pooled, no branch, no phase. **Selection uses grouped 5-fold CV over the train+val patients** (logistic probe) and the MLP-head val Score; the official test Score is shown but not used to choose. Mean ± SD (CV: over folds; others: 5 seeds).

| Encoder | CV Score | CV crackle F1 | CV wheeze F1 | MLP val Score | Test Score | Test crackle F1 | Test wheeze F1 | Coverage | Set size |
|---|---|---|---|---|---|---|---|---|---|
| OPERA-CT | 0.497 ± 0.056 | 0.337 | 0.182 | 0.501 ± 0.015 | 0.417 ± 0.023 | 0.249 ± 0.032 | 0.180 ± 0.069 | 0.855 | 2.56 |
| **AST (AudioSet)** | **0.601 ± 0.035** | 0.441 | **0.503** | **0.653 ± 0.008** | **0.547 ± 0.013** | 0.470 ± 0.043 | **0.356 ± 0.021** | 0.830 | **1.85** |
| CLAP (HTS-AT) | 0.578 ± 0.028 | **0.461** | 0.408 | 0.573 ± 0.015 | 0.519 ± 0.018 | 0.426 ± 0.061 | 0.339 ± 0.026 | 0.847 | 2.16 |
| HeAR (ViT-L, 1024-d patch tokens) | 0.580 ± 0.026 | 0.431 | 0.408 | 0.628 ± 0.015 | 0.527 ± 0.010 | 0.397 ± 0.053 | 0.308 ± 0.063 | 0.857 | 2.09 |

- Four-encoder run: `outputs/encoder_compare/20261003-224347_four-encoders/`. HeAR ties CLAP on CV (0.580 vs 0.578) and sits between CLAP and AST on val (0.628) and test (0.527). Published anchors (official split): SG-SCL 61.71, Patch-Mix CL 62.37, BTS 63.54; Lung-SRAD 64.48 and Meta-Ensemble 66.49 are abstract-level claims, protocol not checked.
- **AST is selected** on both selection criteria (CV +0.104 and val +0.152 over OPERA-CT). OPERA-CT is last on every criterion.
- AST vs CLAP: CV difference 0.023 is within fold SD; MLP val (+0.080, seed SD 0.01) and test (+0.028, seed SD 0.02) favour AST. CLAP is slightly better on CV crackle F1 (0.461 vs 0.441, within SD).
- **A frozen AST (0.547) recovers about 60% of the gap to the fine-tuned Patch-Mix CL (0.59-0.61),** with no fine-tuning and 5 seeds in a few minutes.
- **The weak base model of sections 2-4 was the encoder.** OPERA-CT was pretrained on ICBHI and HF_Lung, yet scores lowest, so pretraining overlap did not inflate it. Non-OPERA encoders are better, so by the G2 rule they should be the primary results.
- Sets are smaller with a better encoder (1.85 vs 2.56 of 4 labels) and coverage is still below 0.90 (0.83), so the shift finding in section 4 holds on AST.

## 9. Patch-Mix CL reproduction (proposal ±1 point check)

Status: **5 of 5 seeds done** (seed 1 on the RTX 4060 8 GB laptop, ~6.5 h; seeds 2–5 on an RTX 5000 Ada 32 GB, ~1 h each, same script and arguments). A first attempt died at epoch 3 when the laptop rebooted (test Score 0.570 and 0.577 after epochs 1–2; log `logs/seed1_interrupted_reboot.log`); seed 1 restarted from scratch with full resume. A CUDA OOM at epoch 8 (best weights were kept on the GPU) was fixed by keeping them on the CPU; `run_until_done.sh` now re-runs the resume until the seed finishes. Manual resume: `setsid nohup ./run_repro.sh 1 >> logs/seed1.log 2>&1 &`. Summary: `.venv/bin/python summarize_repro.py`. Code: `../repos/patch-mix_contrastive_learning`, exact arguments of `scripts/icbhi_patchmix_cl.sh` (`baselines/patchmix_cl/run_repro.sh`), log `baselines/patchmix_cl/logs/seed1.log`.

Deviations needed to run on this machine (numerics unchanged):
- torch 2.9.1 + cu128 instead of the original torch 1.x; `timm==0.4.5` as required.
- Audio loaded with `soundfile` instead of `torchaudio.load` (torchaudio 2.9 needs torchcodec). Same float32 PCM.
- timm attention routed through `scaled_dot_product_attention` (same scale, max abs difference 9e-8) to fit batch 8 in 8 GB.
- Full resume: `last.pth` every epoch holds model, classifier, projector, optimizer, AMP scaler, best score and best weights (the original `--resume` restored model and optimizer only). RNG state is not restored.
- The script is not deterministic (`cudnn.benchmark=True`, AMP): the same seed gave epoch-1 test Score 0.570 and 0.499 in two runs. Seed SD over 5 seeds is the right comparison.

Protocol note: the script picks the best epoch **on the official test set** (its validation loader is the test set). The published 62.37 ± 0.61 uses that protocol, which is optimistic. Report both best-on-test (comparable to the paper) and last epoch (no test selection).

| Seed | Best-on-test Score (epoch) | Last-epoch Score | Published |
|---|---|---|---|
| 1 | 61.18 (epoch 14; Sp 75.81, Se 46.56) | 59.51 | |
| 2 | 62.41 (epoch 18; Sp 81.06, Se 43.76) | 59.63 | |
| 3 | 60.60 (epoch 12; Sp 74.48, Se 46.73) | 59.74 | |
| 4 | 62.51 (epoch 21; Sp 81.70, Se 43.33) | 60.19 | |
| 5 | 60.76 (epoch 18; Sp 75.05, Se 46.47) | 58.88 | |
| **Mean ± SD** | **61.49 ± 0.91** | **59.59 ± 0.47** | 62.37 ± 0.61 (5 seeds) |

Reading: with the paper's own (test-selected) protocol the reproduction is 0.88 points below the published mean, inside the ±1 point target, so the reproduction is accepted. The seed SD (0.91) is larger than the published SD (0.61), as expected for a non-deterministic script (seed 1 ran on another GPU, seeds 2–5 on another; no seed-to-GPU effect is claimed). Without test selection (last epoch) the score is 59.59 ± 0.47, 2.8 points below the published mean and about equal to the published plain fine-tuning (59.55 ± 0.88). Fine-tuned AST is ~0.19 above the frozen OPERA-CT probe (0.42) either way. Logs: `baselines/patchmix_cl/logs/seed<N>.log`, summary `.venv/bin/python summarize_repro.py`, scores in `baselines/patchmix_cl/save/results.json`.

## 10. Discussion

- **Weak base model limits every result.** Frozen OPERA-CT + small head reaches ICBHI Score 0.42 on the official split (crackle F1 0.25 on ICBHI, 0.13 on HF_Lung). Effects of pooling or TTA are small relative to seed noise when the classifier barely separates classes. Conformal sets average 2.5–3.5 of 4 labels, so they carry little information.
- **The branch helps through extra spectral input, not resolution.** The 64 ms control recovers ~75% of the Score gain. The crackle-specific signal of the 5 ms window is suggestive (+0.057 vs FM, +0.058 vs 64 ms) and worth retesting with more test patients or a stronger base, but not a result now.
- **Phase information is not used.** With true phase, the gain equals the shuffled control: the head benefits from two pooled groups (more capacity), not from which frames are inspiration. Predicted masks hurt crackle F1, so detector errors add structured noise. Caveat: 0.25 s encoder frames are coarse relative to crackle timing within a phase.
- **The official split is a real shift.** Val → test lowers split-conformal coverage by 4.5 points with no TTA. The covariate-weighted variant restores marginal coverage; k-shot restores it exactly but with high seed SD (10 patients). Mondrian gives class-conditional coverage only by returning almost all labels.
- **Label-shift correction (BBSE) makes coverage worse.** The BBSE weights rely on the confusion matrix of a weak classifier, and the ICBHI shift is not pure label shift (device and site change too). This supports reporting V4 alongside V3, not alone.
- **Tent barely moves this model.** It updates only LayerNorm parameters of a small head on a frozen encoder. The direction matches H4 in every seed (higher Score, lower split coverage, smaller sets), but the size is below seed SD. Stronger shift (KAUH filters, device hold-out) is needed to test H4.

## 11. Limitations

- One test split (official); seed SD excludes split variance; bootstrap CIs cover test-patient sampling only.
- Val (12 patients) both picks the epoch and calibrates conformal.
- OPERA was pretrained on ICBHI and HF_Lung train data; no non-OPERA control encoder yet (G2).
- Phase mask = inspiration vs rest; [Assumption] one ICBHI cycle = one inspiration + one expiration.
- Calibration-aware TTA variant not implemented.

## 12. Next steps

0. Done: HeAR added; ablation, phase and RQ3 rerun on all four encoders (sections 13-14). Ranked next steps: section 16.
1. Patch-Mix CL seeds 2–5 (~6.5 h each) to close the ±1 point check; optional.
2. Fine-tune AST with val-based epoch selection, freeze it, rerun H1 (5 ms vs 64 ms) and RQ3 on top (§7, proposals 1–2).
3. KAUH loader and ICBHI device hold-out for H3/H4 (§7, proposal 4).

## 13. Branch and phase across encoders (ICBHI official test, 5 seeds)

Runs: `outputs/ablate_<encoder>/<run_id>_sweep/` (OPERA-CT: `outputs/tier0_icbhi_baseline/20261003-233355_sweep/`, identical to section 2). Phase masks come from a detector trained on each encoder's own HF_Lung frames (frame F1 inspiration / expiration on HF_Lung test: OPERA-CT 0.71 / 0.36, AST 0.76 / 0.47, CLAP 0.66 / 0.36, HeAR 0.72 / 0.43). Cell = ICBHI Score, crackle F1, wheeze F1 (mean; SDs in the run summaries, typically 0.01-0.02 Score, 0.03-0.09 F1).

| Cell | OPERA-CT | AST | CLAP | HeAR |
|---|---|---|---|---|
| FM only | .417 / .249 / .180 | **.547** / .470 / .356 | .519 / .426 / .339 | .527 / .397 / .308 |
| + branch 5 ms | .472 / .306 / .153 | .553 / .462 / .359 | .534 / .406 / .319 | .524 / .386 / .255 |
| + branch 64 ms (control) | .458 / .248 / .162 | .545 / .434 / .357 | .528 / .464 / .358 | .527 / .430 / .285 |
| + phase | .417 / .240 / .165 | .544 / .481 / .363 | .527 / .460 / .336 | .530 / .371 / .327 |
| + branch + phase | .475 / .281 / .167 | .527 / .456 / .364 | .520 / .403 / .326 | .522 / .413 / .301 |
| + branch + shuffled phase | .468 / .348 / .187 | .529 / .437 / .359 | .518 / .427 / .351 | .530 / .384 / .308 |

Paired patient-bootstrap CIs (95%) against FM only, ICBHI Score: branch OPERA-CT +0.056 [+0.025, +0.084]; AST +0.006 [-0.004, +0.016]; CLAP +0.015 [-0.002, +0.037]; HeAR -0.003 [-0.018, +0.013]. Branch + phase on AST: -0.020 [-0.042, +0.003]. No phase cell improves any encoder's Score.

**H1 direct test, 5 ms minus 64 ms branch, crackle F1:** OPERA-CT +0.058 [-0.019, +0.116]; AST +0.028 [+0.001, +0.054]; CLAP **-0.058** [-0.086, -0.022]; HeAR **-0.044** [-0.076, -0.012]. Wheeze F1: CLAP -0.038, HeAR -0.030 [-0.049, -0.009].

### HF_Lung windows: true vs predicted phase (crackle F1, mean over 5 seeds; Δ vs FM with 95% CI)

Runs: `outputs/hfphase_<encoder>/` and `outputs/hf_phase_ablation/20261004-010640_sweep/`. Mask = inspiration vs rest.

| Encoder | FM | + true phase | + predicted phase | + shuffled true phase | Wheeze F1 Δ, true phase |
|---|---|---|---|---|---|
| OPERA-CT | .129 | +0.023 [+0.003, +0.041] | -0.020 [-0.051, +0.003] | +0.021 | +0.020 |
| **AST** | .386 ± .060 | **+0.086 [+0.049, +0.127]** | **+0.065 [+0.037, +0.097]** | +0.029 [-0.001, +0.061] | **-0.041** [-0.056, -0.004] |
| CLAP | .393 | +0.010 [-0.023, +0.062] | -0.006 | -0.019 | +0.003 |
| HeAR | .420 | -0.037 [-0.082, +0.011] | -0.135 [-0.184, -0.086] | -0.033 | -0.035 |

### RQ3 per encoder (nominal 0.90; 10 labelled patients for k-shot; evaluation on the other 39 test patients)

Runs: `outputs/rq3_<encoder>/`. Coverage (set size of 4):

| Encoder | ECE | V1 split | V2 Mondrian | V3 weighted | V4 label shift | V5 k-shot | Tent lr 0.01: split cov. |
|---|---|---|---|---|---|---|---|
| OPERA-CT | .247 | .855 (2.55) | .921 (3.49) | .889 (3.18) | .778 (2.15) | .903 (2.91) | .847 |
| AST | .250 | .833 (1.84) | .896 (2.96) | .981 (3.77) | .808 (1.72) | .905 (2.28) | .831 |
| CLAP | .136 | .856 (2.17) | .875 (3.39) | .951 (3.39) | .852 (2.23) | .926 (2.72) | .857 |
| HeAR | **.089** | .861 (2.09) | .892 (3.40) | .911 (2.93) | .820 (1.96) | .908 (2.43) | .863 |

## 14. What the cross-encoder results change

| Claim | Status now | Evidence |
|---|---|---|
| H1 short window helps crackle | **Breaks** | 5 ms is worse than 64 ms for CLAP and HeAR (CIs exclude 0), borderline +0.028 for AST. On strong encoders the branch adds nothing to the Score |
| The branch gain in section 2 | Encoder-specific | Only the weak OPERA-CT gains (+0.056); the gain vanishes with AST, CLAP, HeAR. The branch compensated for OPERA-CT's 2 kHz-trained frontend, not for time resolution |
| H2 phase raises crackle F1; shuffled removes it | **Holds on HF_Lung with AST only** | True phase +0.086 [+0.049, +0.127] vs shuffled +0.029; gain above FM seed SD (0.060). Wheeze F1 falls (-0.041), matching the class-specific shape of H2 |
| H2 on ICBHI | **Breaks** | No encoder gains Score or crackle F1 from phase pooling; AST branch+phase is -0.020 |
| Detector error costs performance | Holds | AST: predicted +0.065 vs true +0.086 on crackle F1. HeAR: predicted -0.135 vs FM |
| Shift breaks split conformal | **Holds for all 4 encoders** | V1 coverage 0.83-0.86 |
| k-shot recalibration restores coverage | **Holds for all 4** | 0.903-0.926 with the smallest sets that reach 0.90 (2.3-2.9) |
| V3 covariate weighting restores coverage | Unstable | 0.889 (OPERA-CT), 0.981 (AST), 0.951, 0.911: over-covers with larger sets (up to 3.77) |
| V4 label-shift correction | **Breaks** | Lowest coverage for every encoder (0.78-0.85) |
| H4 TTA moves coverage | Not shown | Tent changes split coverage by at most 0.01 on every encoder, below seed SD |
| Better calibration with larger/more health-specific encoder | Partly | HeAR ECE 0.089 and CLAP 0.136 vs 0.25 for AST and OPERA-CT; AST has the best Score but is the worst calibrated |

## 15. Model architecture review

[Fact] = measured here; [Hypothesis] = untested proposal.

| Component | Finding | Proposed change |
|---|---|---|
| Frozen encoder, last layer, 32 frames | 4 encoders span test Score 0.417-0.547; fine-tuned AST (Patch-Mix) reaches 0.595-0.612 (best-on-test 61.18, last 59.51) [Fact]. The head has only 0.10-0.17 M parameters and sees one layer | [Hypothesis] Use AST features from several layers (learned weights), then fine-tune the last 1-2 blocks or add LoRA. Select by grouped CV, not test |
| Temporal pooling: mean over all 32 frames | About 66% of each 8 s input is repeat padding (median cycle 2.5 s; real-frame fraction 0.34) [Fact]. A mean dilutes brief events | [Hypothesis] Masked attention or max+mean pooling over real frames only (mask from cycle length) |
| Phase pooling: mean per phase, concatenated | Helps AST crackle F1 on HF_Lung (+0.086) but not on ICBHI; hurts wheeze [Fact]. ICBHI masks use one inspiration + one expiration per cycle (88% of cycles decode inspiration-first with OPERA-CT, 86% AST) | [Hypothesis] Use phase as an auxiliary training target (multi-task) or as an additional input to attention pooling, instead of hard pooling; restrict the claim to crackle |
| Short-window branch (5 ms STFT, +42 k parameters) | No gain on AST/CLAP/HeAR; worse than 64 ms for CLAP/HeAR [Fact] | Drop from the main model; keep as reported negative result |
| Training: unweighted CE, no augmentation, 50 epochs, epoch picked on val | Val has 12 patients, 6 "both" cycles; it picks the epoch and calibrates conformal [Fact] | Pick epoch count by grouped CV over train+val patients, then refit; calibrate conformal on held-out patients |
| Data split | Litt3200: 0 train cycles, 461 test cycles (17% of test) [Fact]. Device shift is built into the official split, which explains coverage 0.83-0.86 | Report device-held-out and KAUH filter shift separately (H3/H4); do not tune on the official test |
| HeAR input | 2 s clips; 4 clips per cycle, 12 time tokens each, pooled to 32 frames; mel PCEN preprocessing from `repos/hear` [Fact] | None |

## 16. Comparison with published results and next steps

Official ICBHI split, Score = (Sp + Se) / 2. Published rows are as reported by the sources listed in the search (not re-run); the last two are abstract-level claims whose protocol I did not verify.

| Method | Score | Source / protocol |
|---|---|---|
| This work, frozen OPERA-CT + MLP head | 41.7 ± 2.3 | val-selected epoch |
| This work, frozen CLAP / HeAR + MLP head | 51.9 ± 1.8 / 52.7 ± 1.0 | val-selected epoch |
| **This work, frozen AST + MLP head** | **54.7 ± 1.3** | val-selected epoch, 5 seeds |
| DAT (AST domain adaptation) | 59.81 ± 1.25 | published |
| This work, Patch-Mix CL reproduction | 61.49 ± 0.91 (best-on-test) / 59.59 ± 0.47 (last epoch) | 5 seeds, paper protocol |
| This work, AST fine-tuned (layer-4 cache, `ft-k8-attn+mix`) | 57.3 ± 1.9 | val-selected epoch, 5 seeds (section 17) |
| SG-SCL | 61.71 | published |
| Patch-Mix CL | 62.37 ± 0.61 | published (best epoch chosen on test) |
| BTS (audio + text metadata) | 63.54 | published |
| Lung-SRAD | 64.48 | abstract claim |
| Meta-Ensemble Learning with Diverse Data Splits | 66.49 | abstract claim; split protocol unverified |

Frozen AST is 0.048-0.075 below Patch-Mix CL (depending on epoch-selection protocol) and 0.19 above frozen OPERA-CT. Sources: arXiv 2305.14032, 2312.09603, 2606.11922, 2604.24096; ICBHI numbers for BTS and DAT from the ICBHI literature search results.

### Next steps (ranked)

1. **Make frozen AST the primary encoder; keep CLAP and HeAR as the second and third.** Update proposal B4: OPERA-CT is last on every criterion, so the leakage control no longer constrains the primary result.
2. **Improve the AST head** (section 15): multi-layer features, masked attention/max pooling, then partial fine-tuning. Select by grouped CV. Target: close the 0.05-0.07 gap to Patch-Mix CL under the same val-selected protocol, so the comparison with published numbers is fair.
3. **Follow up the one positive phase signal.** Repeat AST + true/predicted phase on HF_Lung with more seeds, then test phase as a multi-task target on ICBHI. If it does not transfer, report H2 as dataset-specific.
4. **Close H1 as negative.** Remove the short-window branch from the main model.
5. **Test H3/H4 under real device shift** (ICBHI device hold-out including Litt3200, KAUH filter renderings) on AST and HeAR (best calibrated), with k-shot and Mondrian as the recalibration baselines; add the calibration-aware TTA variant and a larger adaptable part.
6. ~~Optional: Patch-Mix CL seeds 2-5~~ done, see section 9 (61.49 ± 0.91). The fine-tuning comparison in our own protocol is in section 17.

## 17. R-plan update (2026-10-05): HeAR ladder, AST layer-4 retry, Part II shift study

Run on an RTX 5000 Ada 32 GB (project moved from the 8 GB laptop). Official ICBHI split, Score = (Sp + Se) / 2, test evaluated once per cell after grouped-CV epoch selection (no test selection), 5 seeds unless stated.

### 17.1 Fine-tuned ladder, HeAR and AST (Part I)

| Encoder / cell | CV Score | Test Score |
|---|---|---|
| HeAR `last-mean` (frozen) | | 0.513 ± 0.010 |
| HeAR `concat-attn` (best frozen head, CV 0.603) | 0.603 | 0.511 ± 0.005 |
| HeAR `ft-k4-attn` (blocks 21-24 tuned) | 0.587 ± 0.025 | **0.570 ± 0.002** |
| AST `ft-k4-attn+mix` (cache after block 8, section F3 of `anchors.md`) | 0.585 ± 0.037 | 0.557 ± 0.009 |
| AST layer-4 cache `ft-k4-attn` (13 ep) | 0.603 ± 0.020 | 0.546 ± 0.009 |
| AST layer-4 cache `ft-k4-attn+mix` (8 ep) | 0.612 ± 0.012 | 0.555 ± 0.006 |
| AST layer-4 cache `ft-k8-attn` (8 ep) | 0.607 ± 0.012 | 0.569 ± 0.013 |
| AST layer-4 cache `ft-k8-attn+mix` (21 ep) | 0.605 ± 0.020 | **0.573 ± 0.019** |

AST layer-4 retry: token cache after block 4 (blocks 5-12 trainable), `lr_block` 5e-5 (was 2e-5), 30-epoch budget, 3-fold grouped CV picks the epoch (`outputs/ladder_ast_l4/ladder-ast_l4-v1/`, config `configs/ladder_ast_l4.yaml`).

- [Fact] Best result of the R-plan is AST `ft-k8-attn+mix` at 0.573 ± 0.019, up from 0.557 (layer-8 cache). Tuning 8 blocks helps more than the `+mix` flag (k8 vs k4: +0.023 without mix, +0.018 with). No paired CIs were computed for the layer-4 cells, so the cell-to-cell differences within 0.01-0.02 are not claimed.
- [Fact] HeAR fine-tuning raises the test Score by 0.057 over its frozen heads (0.513 → 0.570), the largest fine-tuning gain of any encoder, although CV does not rank `ft-k4-attn` above the frozen heads (0.587 vs 0.603). CV scores are above the test Scores for every cell (by 0.02-0.09; HeAR `ft-k4-attn` is the smallest gap, frozen HeAR heads the largest): the official test split is harder than the grouped-CV folds of the train+val patients.
- [Fact] Part I criterion not met. Best rung 57.3 vs Patch-Mix CL 62.37 ± 0.61 (published) and 61.49 ± 0.91 (our reproduction, test-selected); vs plain AST fine-tuning 59.55 ± 0.88 (published). Gap to the reproduction under the stricter protocol: 4.2 points; to the last-epoch reproduction (59.59, no test selection): 2.3 points.
- [Interpretation] The remaining gap is plausibly the cached-token design (the lower blocks stay frozen at the cached layer) and the short schedule, as in section F3 of `anchors.md`; not tested by a full fine-tune in our protocol.
- Per the proposal (G8), Part II runs on the frozen rungs L0/L1, with L2 added for AST and HeAR.

### 17.2 Part II: device shift, coverage and decodability per rung

Runner `src/shift_study.py` (`RUN_ID` `shift-v1`; tables `outputs/shift_table_v1.txt`, `outputs/shift_summary_v1.txt`). Rungs: L0 logistic probe, L1 best frozen head, L2 best plain `ft-k*` cell (AST, HeAR); cell and epoch from the ICBHI ladder CV, nothing chosen on a shifted target. Each unit trains on source-train patients, calibrates the conformal layer on source-calibration patients (V1 split, V2 Mondrian, alpha = 0.1) and evaluates on the shifted set. Decodability = AUC of a domain classifier on the rung's embedding, within class, patient-grouped folds.

**E1, ICBHI one device held out** (train and calibrate on the other devices, test on the held device's official-test patients; `device_holdout_official`). V1 coverage (nominal 0.90; seeds: 1 for L0, 5 for L1/L2):

| Held-out device (test cycles) | L0 probe, AST / CLAP / HeAR / OPERA-CT / CE | L1 head, AST / CLAP / HeAR / OPERA-CT / CE | L2 fine-tuned, AST / HeAR |
|---|---|---|---|
| AKGC417L (1836) | 0.97 / 0.96 / 0.95 / 0.90 / 0.92 | 0.80 / 0.68 / 0.86 / 0.73 / 0.75 | 0.83 / 0.87 |
| Litt3200 (461) | 0.47 / 0.47 / 0.61 / 0.72 / 0.68 | 0.97 / 0.78 / 0.79 / 0.98 / 0.91 | 0.95 / 0.91 |
| Meditron (459) | 0.99 / 1.00 / 0.95 / 0.88 / 0.87 | 0.96 / 0.96 / 0.95 / 0.91 / 0.91 | 0.93 / 0.92 |

LittC2SE is skipped (no official-test recordings). Binomial half-width of the nominal at these sizes: ±0.014 (1836), ±0.027 (459-461). Decodability AUC of the held-out device: 0.9-1.0 for AKGC417L and Litt3200, 0.72-0.88 for Meditron.

**E2, KAUH filter pairs** (5 patient-disjoint 60/20/20 partitions x 6 ordered pairs of Bell/Diaphragm/Extended; train and calibrate on the source filter, test on the same test patients under the target filter; 8 s windows, ~35 test windows per partition). V1 coverage, source filter → target filter, mean over 30 units:

| Rung | AST | HeAR | CLAP | OPERA-CT | OPERA-CE |
|---|---|---|---|---|---|
| L0 probe | 0.881 → 0.841 | 0.875 → 0.782 | 0.894 → 0.815 | 0.892 → 0.491 | 0.907 → 0.445 |
| L1 head | 0.892 → 0.914 | 0.878 → 0.880 | 0.870 → 0.810 | 0.856 → 0.617 | 0.894 → 0.861 |
| L2 fine-tuned | 0.904 → 0.927 | 0.896 → 0.870 | | | |

The Score falls under the filter change for every rung (AST head 0.579 → 0.499). KAUH decodability AUC is 0.94-1.00 for every rung.

**Hypothesis status**

| Claim | Status | Evidence |
|---|---|---|
| H2: coverage falls below nominal under device shift | Supported in part | [Fact] On two of three real devices (AKGC417L, Litt3200) V1 coverage is below 0.90 by more than the binomial half-width for most rungs; none on Meditron. [Fact] On KAUH the deficit is large only for the OPERA probes and moderate for CLAP/HeAR probes; AST/HeAR heads and fine-tuned rungs show little or none. |
| H2, correction part (A1/A2 close part of the deficit) | Not tested | A1/A2 are not wired into extraction. |
| H4: fine-tuning raises decodability and the coverage deficit | Not supported | [Fact] AST E1 coverage 0.904 (L2) vs 0.910 (L1), AUC 0.91 vs 0.92; HeAR E1 coverage 0.900 vs 0.869, AUC 0.93 vs 0.93; KAUH AST 0.927 vs 0.914, AUC 0.99 vs 0.99. Fine-tuned coverage is never lower by more than 0.03 on average and decodability does not rise. |
| H1, H3 (phantom, matched IR augmentation) | Not tested | No phantom or extra-hardware data. |

**Caveats**
- [Fact] ICBHI device is confounded with class prior and site (CLAUDE.md data rules). The E1 deficit may be label shift rather than a gain difference; the V4 label-shift variant exists in `src/conformal/conformal.py` and was not run here.
- [Fact] KAUH test sets hold ~35 windows per partition and the 30 units of a rung share patients, so they are not independent. No CIs are reported for E2; differences below ~0.1 in coverage are inside the binomial noise.
- [Fact] Decodability is saturated (AUC 0.9-1.0), so it cannot separate the rungs. The proposal's Spearman test of decodability against deficit needs more shift pairs than the 3 held-out devices; Meditron is the only low-AUC device and also the only one with no deficit, which is suggestive only.
- [Assumption] KAUH training sets are ~40x smaller than ICBHI, so epochs are step-matched to the ICBHI CV epoch (`n_dev / n_train` scaling) rather than selected on the target. This choice is untested against alternatives.
- [Fact] E1 L0 probes are deterministic (1 run per device); L1/L2 use 5 seeds, and the table gives the seed mean without SD (SDs are in the per-unit JSONs under `outputs/shift_<enc>/shift-v1/`).
- The first AST E2 run lacked decodability because of a size guard that skipped sets under 100 rows; the results were discarded (`outputs/shift_ast_noauc_discarded/`) and E2 was re-run for AST. Other encoders ran with the fixed code.

### 17.3 Next steps (ranked)

1. Wire A1 (spectrum correction) and A2 (ISA moment matching) into extraction and re-run E1 on AKGC417L and Litt3200: the only way to test the second half of H2.
2. Run the V4 label-shift recalibration on the same two devices to separate label shift from the gain-like part of the deficit.
3. Add paired patient-level bootstrap CIs to the E1 rung comparisons; add a decodability measure that does not saturate (e.g. a low-capacity probe or a per-class variant on matched labels) before testing H4 further.
4. Phantom recordings and a second hardware (H1, H3) remain blocked on data.
