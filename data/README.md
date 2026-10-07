# data/

Raw and derived data are not stored in this repo. They live in `../data` (the wrapper directory) or in `$AT_DATA`
(`src/paths.py:DATA`):

```
raw/icbhi/ICBHI_final_database  raw/kauh  raw/hf_lung_v1      immutable originals (never modify)
processed/patchmix_icbhi/       audio_test_data/ (symlinks to raw), official_split.txt: Patch-Mix CL input
splits/                         committed patient-disjoint split files
models/ast/audioset_10_10_0.4593.pth   AST AudioSet weights (Stage 1 init)
cache/                          embedding caches (Phase 0)
```

Tier rules (CLAUDE.md): tier 0 ICBHI + KAUH for training/selection/calibration; tier 1 phantom recordings evaluate-only;
tier 2 hospital data out of scope for v9.
