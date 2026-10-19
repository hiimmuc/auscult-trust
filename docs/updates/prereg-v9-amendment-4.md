# Amendment 4 to `prereg-v8.md` (proposal v9): LODO and SC across encoders and methods (LODO v1)

**Status: registered 19/10/2026, committed before the first training unit of run `lodo-v1` (the smoke run
`smoke-lodo-v1` is not a training unit of the run).**

Nothing here changes a registered rule of Amendments 1-3. Gate G, H1, H2, H3 and the positive controls keep their
registered arms (AST Patch-Mix CL baseline, AuscultTrust, Freq-MixStyle in `outputs/train/lodo-ast`), recipe, epochs and
statistics; those registered results stay reported. Everything below is **exploratory: reported with the same
statistics, no verdict**.

## 1. Why

The method x encoder grid (`docs/reports/grid-v1-results.md`, official split) found BEATs ahead of every other encoder
and the worst device to be Littmann 3200 in 343 of 350 units. The registered LODO (AST only) tested H2 on one fold
(Littmann 3200). This amendment asks whether the coverage effect of spectrum correction (SC) and the device-shift
robustness hold across encoders and methods, and selects the final AuscultTrust configuration from these runs.

[Fact] The cells below were chosen after seeing official-split test results; the held-out-device folds contain those
test cycles. [Fact] The final AuscultTrust configuration (encoder + method + SC) will be chosen after LODO and SC, so
its LODO numbers are selected on the held-out devices and are reported as such. The registered AuscultTrust (AST
Patch-Mix CL + SC) results stay the registered outcomes.

## 2. Arms

- No SC (machines A and B), 11 cells: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce,
  patchmix_cl, sg_scl, pafa, qlung. Recipe of each cell = its grid config (`scripts/grid.py cells`).
- Trained SC (machine C), 7 cells: AST patchmix_cl, pafa; HTS-AT patchmix_cl, sg_scl; BEATs patchmix_cl, pafa, sg_scl.
  Static bound first (primary: the bound chosen by CV in Stage 1), dynamic bound later. Official split (seeds 0-4,
  paired with the grid's no-SC units) and LODO.
- Inference-only arms on every LODO unit (`src.evaluation.lodo_ttsc`): test-time SC (held-out device corrected towards
  the source mean, static and dynamic bound), A2, TTA-EQ (8 views, seeds 1000-1007).

## 3. Protocol

- Folds: Meditron, LittC2SE, Litt3200, AKGC417L held out (`lodo_split`, 20% calibration patients per source device,
  seeded by the run seed; the 2 patients on both sides of the official split dropped).
- Seeds 0-4. Fixed epoch: the cell's last grid epoch (50; BEATs pafa 100), the same for every arm of a cell. Every
  epoch's test softmax is kept; only the fixed epoch is reported.
- Run `lodo-v1`; job lists from `scripts/grid.py lodo-jobs` (`logs/campaigns/lodo-v1-{A,B,C}.txt`).
- No 80/20 split stage.

## 4. Readings (no verdict)

- Per fold and macro (with and without the flagged AKGC417L fold): Sp, Se, Score, HS, 2-class view, per arm, mean ± SD.
- Method effects under shift, paired by seed: each method vs ce on the same encoder.
- H2 statistics as registered (`src.evaluation.lodo`: prior-matched coverage, Delta, null gate over 200 re-splits,
  two-level bootstrap), for each SC arm (trained or test-time) against the same cell without it. Set size and singleton
  rate are read next to |Delta|: H2 was only partly met on AST because SC enlarged the sets.
- SC on the official split: Score paired by seed against the grid unit of the same cell.
- Optional, after the above: ensembles fixed now (E1 = equal-weight mean softmax of the 11 no-SC cells per fold and
  seed; E2 the 5 BEATs cells; E3 the 3 AST cells; E4 the 3 HTS-AT cells), no weights, no member selection; then
  distillation of E1 into one BEATs model only if E1 beats the best single cell.

## 5. Not done

DASS and OPERA-CT in LODO; the 80/20 split step.
