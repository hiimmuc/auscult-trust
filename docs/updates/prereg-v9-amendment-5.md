# Amendment 5 to `prereg-v8.md` (proposal v9): PC-MCL and PAFA + PC-MCL on BEATs

**Status: registered 19/10/2026, before the first training unit of run `pcmcl-v1` (machine D; the user moved it ahead
of the end of LODO v1).** Exploratory: reported with the same statistics, no verdict.

## 1. Why

PC-MCL (`../repos/PC-MCL`; arXiv 2601.17080) reports BEATs 65.37 ± 0.73 Score on the official split, the best
published BEATs number, with high Se (51.71). It uses patient identity, like PAFA, which is relevant to device shift.
[Fact] The reference code keeps the best epoch on the test set, so the published number compares with the grid's
best-epoch column (BEATs patchmix_cl 63.6 ± 0.8, our PAFA port 63.2 ± 0.6 vs 64.84 published).

## 2. Arm

- Cell `beats+ft+pcmcl`: `base.yaml,encoder_beats.yaml,grid/pcmcl_beats.yaml`; method `pcmcl`
  (`src/training/patchmix_cl/pcmcl.py`).
  - Per epoch: every training cycle once with a 3-label target (normal, crackle, wheeze); 0.5 x n multi-cycle samples
    (2 or 3 short cycles concatenated, label = OR, the reference "additive" combinations); 0.3 x n patient-matching
    samples (same patient = 1; two normal-only patients = 0, the reference's hard negatives in practice).
  - Loss: BCE on the 3 labels + 0.1 x cross-entropy of the patient-matching head (paper weight).
  - Prediction: crackle and wheeze sigmoids as independent events give the 4-class distribution (argmax = the
    reference threshold rule at 0.5); conformal sets use it unchanged.
  - Settings not given by the paper or a script: PAFA's BEATs setting (batch 32, lr 5e-5, 100 epochs, no SpecAugment),
    10 s input (paper).
- Cell `beats+ft+pafa_pcmcl`: `grid/pafa_pcmcl_beats.yaml` = the above plus PAFA's BEATs losses (lambda_pcsl 50.0,
  lambda_gpal 0.0005) on the samples of one patient (single cycles and same-patient pairs; samples joining two
  patients have no patient and are left out of the PAFA loss).
- Not implemented: SC on concatenated cycles (needed only if PC-MCL enters the SC arms).

## 3. Protocol and readings

- Official split, seeds 0-4, fixed last epoch (100), run `pcmcl-v1`, machine D. Best epoch on test reported as optimistic.
- Paired by seed against `beats+ft+pafa` and `beats+ft+ce` of the grid: Sp, Se, Score, HS, 2-class, worst device,
  coverage.
- A cell enters LODO and SC only if its last-epoch Score beats BEATs PAFA (61.7 ± 0.7) by more than one seed SD.
- Cost (smoke on D, RTX 5090, two jobs at once): about 57 s per epoch, about 1.7 h per unit; 10 units, 2 at a time,
  about 9 h. Two epochs of the real schedule: test Score 59.9 (pcmcl) and 51.0 (pafa_pcmcl) after epochs 1 and 2.
