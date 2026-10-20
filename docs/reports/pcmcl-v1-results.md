# PC-MCL v1 results: BEATs PC-MCL and PAFA + PC-MCL on the ICBHI official split (20/10/2026)

Registration: `docs/updates/prereg-v9-amendment-5.md` (exploratory, no verdict). Run `pcmcl-v1`, machine D, seeds 0-4,
100 epochs, finished 20/10/2026 (10 of 10 units, 0 failed). Outputs and checkpoints copied to A and checked by checksum
(`outputs|checkpoints/train/pcmcl-v1/`). Comparators: grid-v1 `beats+ft+pafa` and `beats+ft+ce`, same seeds.
Columns as in `docs/reports/grid-v1-results.md` (all in %; Cov = LAC split-conformal coverage at 90%).

## Summary

- [Fact] Last epoch (primary): PC-MCL 60.9 ± 0.6, PAFA + PC-MCL 61.3 ± 1.0, against BEATs PAFA 61.7 ± 0.7. Paired by
  seed, both are below PAFA (-0.8 ± 0.9 and -0.4 ± 1.1). **Neither meets the Amendment 5 rule** (last-epoch Score above
  PAFA by more than one seed SD), so neither enters LODO or the SC arms.
- [Fact] Best epoch on test (optimistic): PC-MCL 64.3 ± 1.0 (published 65.37 ± 0.73, best epoch on test), the highest
  BEATs number so far (PAFA 63.2 ± 0.6, Patch-Mix CL 63.6 ± 0.8). Its best epochs are 4-6 of 100: PC-MCL peaks early and
  then loses about 3.4 Score by the last epoch.
- [Fact] PC-MCL gives the smallest conformal sets of the BEATs cells (1.95 at the last epoch vs 2.47 for PAFA) at the
  same pooled coverage, and keeps CE's high Se (52.6) with higher Sp (69.3 vs 65.7).
- [Interpretation] The published PC-MCL number relies on the test-selected epoch; with a fixed schedule PC-MCL is
  level with PAFA. A validated early epoch (grouped CV) could recover part of the gap, but the registered rule is the
  fixed last epoch.

## Last epoch

| Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beats+ft+pcmcl | 69.3 ± 1.7 | 52.6 ± 1.5 | 60.9 ± 0.6 | 59.7 ± 0.7 | 69.4 ± 1.5 | 69.3 ± 0.4 | 69.3 ± 0.4 | 57.2 ± 2.6 | 89.2 ± 0.1 | 86.6 ± 0.4 | 1.95 ± 0.02 |
| beats+ft+pafa_pcmcl | 74.0 ± 1.7 | 48.6 ± 0.7 | 61.3 ± 1.0 | 58.6 ± 0.8 | 65.3 ± 0.3 | 69.7 ± 0.9 | 69.4 ± 0.8 | 58.8 ± 1.0 | 89.8 ± 0.2 | 86.8 ± 0.2 | 2.07 ± 0.05 |
| beats+ft+pafa (grid) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| beats+ft+ce (grid) | 65.7 ± 3.7 | 52.7 ± 1.3 | 59.2 ± 1.5 | 58.4 ± 1.1 | 69.7 ± 1.4 | 67.7 ± 1.5 | 67.6 ± 1.5 | 50.4 ± 3.0 | 89.4 ± 0.4 | 86.6 ± 0.4 | 2.38 ± 0.06 |

Paired Score difference to beats+ft+pafa (same seed): pcmcl -0.8 ± 0.9 (per seed +0.3, -2.1, -1.3, -0.5, -0.5);
pafa_pcmcl -0.4 ± 1.1 (+1.3, -1.0, -0.5, -0.3, -1.8).

## Best epoch on test (optimistic, for comparison with papers only)

| Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beats+ft+pcmcl | 77.0 ± 2.8 | 51.7 ± 3.0 | 64.3 ± 1.0 | 61.8 ± 1.6 | 66.9 ± 3.6 | 71.9 ± 1.4 | 71.5 ± 1.6 | 58.8 ± 3.6 | 89.2 ± 0.2 | 85.5 ± 0.5 | 1.85 ± 0.03 |
| beats+ft+pafa_pcmcl | 80.2 ± 4.0 | 46.9 ± 2.3 | 63.6 ± 1.1 | 59.1 ± 1.1 | 61.7 ± 3.8 | 71.0 ± 0.8 | 69.6 ± 1.2 | 59.3 ± 2.6 | 89.4 ± 0.3 | 84.5 ± 1.7 | 2.02 ± 0.06 |
| beats+ft+pafa (grid) | 75.5 ± 2.9 | 51.0 ± 2.8 | 63.2 ± 0.6 | 60.8 ± 1.2 | 65.8 ± 3.2 | 70.6 ± 0.4 | 70.2 ± 0.6 | 60.0 ± 0.9 | 89.7 ± 0.2 | 86.0 ± 0.8 | 2.16 ± 0.28 |
| beats+ft+ce (grid) | 74.9 ± 4.3 | 49.4 ± 2.3 | 62.2 ± 1.1 | 59.4 ± 0.8 | 65.2 ± 2.8 | 70.1 ± 0.8 | 69.6 ± 0.4 | 53.7 ± 3.2 | 89.7 ± 0.2 | 86.2 ± 1.0 | 2.10 ± 0.21 |

Paired Score difference to beats+ft+pafa: pcmcl +1.1 ± 1.1 (best epochs 6, 4, 6, 4, 6); pafa_pcmcl +0.3 ± 0.7 (best
epochs 14, 55, 33, 13, 9).
