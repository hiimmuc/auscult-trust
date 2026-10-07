# Training experiments

`base.yaml` is the Patch-Mix CL recipe (AST). One file per arm; `--config base.yaml,<arm>.yaml[,sc_static.yaml]`, later files win.
The cell name is the file stems joined by `+` without `base` (`sc+sc_static`).

| arm | file | meaning |
|---|---|---|
| baseline | `baseline.yaml` | Patch-Mix CL without any device-shift handling |
| sc | `sc.yaml` | spectrum correction (SC) of the waveform at train and test time; each device's spectrum is mapped to the mean training spectrum |
| sc_gain | `sc_gain.yaml` | SC plus a random smooth per-mel-bin gain (6 dB SD) at train time |
| freq_mixstyle | `freq_mixstyle.yaml` | Freq-MixStyle augmentation (p = 0.5), a competing method |
| (override) | `sc_static.yaml` | SC coefficients unclipped in 50-2000 Hz and unchanged outside; `screen` runs both bounds per SC arm and keeps the better CV Score (tie: dynamic) |

Encoders: `encoder_<name>.yaml` for `ast`, `htsat`, `opera_ct`, `clap`, `hear` (default `ast`); AST and HTS-AT run in both modes, OPERA-CT, CLAP and HeAR only with `freeze_encoder.yaml` (enforced). Method: `patchmix.yaml` (Patch-Mix only; default is Patch-Mix CL). Combine, e.g. `--config base.yaml,encoder_clap.yaml,freeze_encoder.yaml,sc.yaml`. Random gain (`sc_gain`) is not defined for HeAR (its image is a PCEN mel, not a log-mel).

Run everything with `bash scripts/run.sh` (the file header explains each step).
