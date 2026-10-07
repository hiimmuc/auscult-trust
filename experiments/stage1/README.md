# Stage 1 experiments (proposal v9)

`base.yaml` is the Patch-Mix CL recipe. One arm file per variant; `--config base.yaml,<arm>.yaml[,sc_static.yaml]`, later
files win. The cell name is the stems joined by `+` without `base` (`P1_a1_input+sc_static`).

| arm | file | meaning |
|---|---|---|
| P0 | `P0_baseline.yaml` | Patch-Mix CL |
| P1 | `P1_a1_input.yaml` | + SC at train and test |
| P1+P3 | `P1P3_sc_gain.yaml` | SC + random per-bin gain (6 dB SD) |
| P4 | `P4_freq_mixstyle.yaml` | Freq-MixStyle p = 0.5 (competitor) |
| - | `sc_static.yaml` | `sc_mode: static`; the screen runs both modes per SC arm and keeps the better mean CV Score (tie: dynamic) |

v8 variants P2, P3, P6, P7, P8, P9 are not v9 candidates; their configs and code are in `archive/`.
Run with `bash scripts/run_stage1.sh` (see `docs/QUICKSTART.md`).
