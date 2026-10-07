# Stage 1 variants (proposal v8)

One json per variant, loaded by `baselines/patchmix_cl/main.py --config`. Each file is the Patch-Mix CL recipe of
`run_repro.sh` plus one flag. Run every variant with the same seed list: `bash scripts/run_stage1_variants.sh (repo root)` (SEEDS="0 1 2 3 4").

| id | flag | meaning |
|---|---|---|
| P0 | none | baseline, validation-selected epoch (`selection: cv`) |
| P1 | `a1_input` | A1 spectrum correction on the waveform, device from the file name |
| P2 | `bin_norm` | per-device, per-mel-bin standardisation |
| P3 | `rand_bin_gain` | random per-bin gain (6 dB SD), `random_bin_gain` of `src/shift/correction.py` |
| P4 | `freq_mixstyle` | Freq-MixStyle, p = 0.5 |
| P5 | dropped | microphone IR augmentation: no stethoscope or microphone IR set with a usable licence is in `data/` |
| P6 | `bin_affine` | learnable per-mel-bin affine layer before the encoder (`models/ast.py`) |
| P7 | `hc_fuse` | deep feature + the 20 frame-SDs of the MFCC: a stationary per-frequency gain adds a constant to every MFCC, so their SD is invariant (tested in `tests/test_handcrafted.py`) |
| P8 | `select_metric: worst_device` | epoch selected by the worst per-device validation Score |
| P9 | combination | two best single variants after screening: `--config configs/stage1_variants/P1_a1_input.json,configs/stage1_variants/P3_rand_bin_gain.json` (comma list, later files override); no fixed json, chosen by the registered rule |

Outputs per run, in `save/<model_name>/stage1_report.json`: epochs `last`, `cv_selected`, `test_best_optimistic` (the last
is chosen with test labels and must be labelled optimistic), each with Sp, Se, Score, HS, macro-F1 pooled and per device,
plus the 2-class view (`two_cls_*`) of the same 4-class predictions. `--two_cls_eval` is not a separately trained model: it
collapses the 4-class predictions to normal/abnormal.
