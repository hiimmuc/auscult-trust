# Training sweep: definition checked against the papers (2026-10-07)

## What the papers do [Fact, read from the papers and their scripts]
- Patch-Mix CL, SG-SCL, Lung-SRAD: ICBHI official 60/40 split (4,142 train / 2,756 test cycles), **5 random seeds, no cross-validation**.
  Lung-SRAD (https://arxiv.org/abs/2606.11922): Adam, lr 5e-5, batch 16, 50 epochs, 8 s cycles, SpecAugment (160 frames, 48 bins), mean over 5 seeds.
  Patch-Mix CL reference script: batch 8, lr 5e-5, wd 1e-6, cosine, 50 epochs, moving average 0.5, temperature 0.06 (Lung-SRAD DASS script: batch 16, temperature 0.2, seeds 1-5; the Patch-Mix script shows seed 1 only).
- Reported best epoch is chosen on the test set (optimistic). Metrics: Sp, Se, Score = (Sp + Se) / 2 (4-class primary, 2-class secondary: normal vs abnormal).
- OPERA (frozen encoder, linear probe, 5 runs) uses ICBHI only for COPD vs healthy; it has no cycle-level ICBHI score.
- Consequence: "5 seeds" matches the papers; **cross-validation does not**. We keep CV only for our own choices (epoch, SC bound, arm), reported next to the paper protocol (best epoch on test, labelled optimistic).

## ICBHI subtasks
1. 4-class lung sound per cycle (normal, crackle, wheeze, both): primary, trained.
2. 2-class normal vs abnormal: evaluation view of the same 4-class predictions (what the papers' "2-class eval" reports); no separate model.
3. Not in the papers, not in the sweep unless requested: diagnosis per patient (healthy / chronic / non-chronic, 3-class; healthy / unhealthy, 2-class), OPERA T7 COPD vs healthy, T11 COPD severity.

## Sweep axes
- Encoder: AST, HTS-AT, OPERA-CT, HeAR, CLAP, DASS. Method: Patch-Mix CL, Patch-Mix. Mode: full fine-tune, frozen encoder (train only the new parts).
- Arms: baseline, SC (`sc`), SC + random gain (`sc_gain`), Freq-MixStyle (`freq_mixstyle`). Metrics: Sp, Se, Score, HS, per-class recall (in every report).
- Phases: (1) paper-protocol comparison, (2) leave-one-device-out, (3) frozen-weight external evaluation (KAUH positive controls, isolation contrast) with our models, RespireNet, pretrained Patch-Mix CL, pretrained Lung-SRAD.

## Cost [Interpretation, measured on one RTX 5000 Ada]
AST fine-tune: about 1 h per seed (50 epochs, test scored every epoch); frozen AST: about 20 min. Full grid = 6 encoders x 2 methods x 4 arms x 2 modes x 5 seeds = 480 runs, roughly 400-500 GPU-hours before LODO: not feasible as one sweep. Proposed order: AST first (done end to end), then add one encoder at a time; Patch-Mix only for baseline and the best arm; the three arms on all encoders only with Patch-Mix CL.

## Encoder readiness (updated)
Done and smoke-tested (1 epoch, screening fold): AST, HTS-AT (fine-tune), OPERA-CT, CLAP, HeAR (frozen only, as decided). Each encoder in `src/training/patchmix_cl/encoders/` has its own spectrogram front end and Patch-Mix on its patch tokens (HTS-AT, OPERA-CT: HTS-AT Swin network of the OPERA repository; CLAP: HF audio tower; HeAR: four 2 s clips per cycle). Frozen mode: no gradients through the encoder, eval mode, learning rate 1e-3 for the new parts. HeAR weights are used from the local cache (account not on the gated list). DASS still open. Old notes below.

## Encoder readiness (earlier)
- AST: ready (reference code). Patch-Mix, Patch-Mix CL, frozen mode work (smoke tested).
- DASS: needs the DASS selective-scan CUDA kernel built with CUDA 12.8 (nvcc here is 13.0) and the v0.2 checkpoints; Lung-SRAD code in `repos/Lung-SRAD`.
- HTS-AT / CLAP / OPERA-CT: checkpoints in `data/models/{htsat,opera}`; Swin-style windows need a Patch-Mix adaptation (not in the reference code).
- HeAR: ViT-L, weights not downloaded (gated Hugging Face model, token needed).

## Config check against the papers in docs/papers (2026-10-07)

Read from the PDFs (Patch-Mix CL 2305.14032, SG-SCL 2312.09603, Lung-SRAD 2606.11922, OPERA 2406.16148) and the reference scripts.

| Item | Papers | Ours (`experiments/train/base.yaml`) | Verdict |
|---|---|---|---|
| Split, input | official 60/40, 8 s cycles, 16 kHz, 128-mel fbank (798 frames) | same | match |
| Optimiser | Adam, lr 5e-5, cosine, batch 8 (Lung-SRAD DASS: 16), 50 epochs | same; DASS uses batch 16 | match |
| Weight decay | 1e-6 (reference scripts) | 1e-6 | match |
| Moving average | coefficient 0.5 on all learnable parameters | `ma_beta` 0.5 on encoder, head, projector | match |
| SpecAugment | max mask 160 frames, 48 bins, mean fill, no time warp | same | match |
| Patch-Mix CL | beta 1.0, tau 0.06, alpha 1.0 (SG-SCL tau 0.06; Lung-SRAD tau 0.2) | same; DASS tau 0.2 | match |
| Loss | weighted cross-entropy only when no mixing is used | unweighted (every arm mixes) | match for mixing arms; a plain CE arm would need class weights (option removed) |
| Seeds | five seeds, no cross-validation (SG-SCL: "a fixed set of five seeds"; Lung-SRAD scripts use 1-5, Patch-Mix script shows seed 1) | seeds 0-4 | five seeds match; the exact numbers are not stated, ours are 0-4 |
| Reported epoch | best on test (optimistic) | best on test and last epoch | both reported |
| Frozen encoder | not in these papers; OPERA linear probe: one linear layer, lr 1e-4, L2 1e-5, 5 runs | head + projector trained, lr 1e-3, batch 8, 50 epochs, Patch-Mix on the frozen tokens | our design, not a paper protocol |
| HTS-AT, OPERA-CT, CLAP, HeAR with Patch-Mix | no paper does this | same recipe as AST | our adaptation |
| DASS | Lung-SRAD uses a dual time/frequency Patch-Mix CL with Gaussian blur | plain DASS with the 2-D Patch-Mix, no blur | deliberate, to compare with the other encoders |

Changed after this check: `encoder_dass.yaml` now sets batch size 16 and tau 0.2 as in Lung-SRAD.
Reproduction anchor already met: AST + Patch-Mix, 4 seeds, best-on-test Score 59.69 ± 0.51 against 59.46 ± 0.78 in the Patch-Mix CL ablation.
