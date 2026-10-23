# Frozen PAFA/static-SC evaluation execution log

Date: 23 October 2026 (+07). Decision: `docs/decision_logs/2026-10-23-final-model.md`.

## Scope and fixed choices

- Baseline: BEATs + PAFA, no SC. AuscultTrust: same recipe trained/evaluated with static SC.
- Five paired seeds (0–4), epoch 100. Serving seed 0 fixed before this evaluation.
- Exploratory E1 and E1+E3: equal-weight mean softmax, no fitted weights, member epochs 100/50.
  Both no-SC and trained-static-SC variants. Classification `k rate` on the cycle benchmark is
  separate from this window/recording-level conformal evaluation; it is not silently transferred.
- KAUH: frozen weights, unlabelled per-filter spectra; A paired invariance and B2 source-filter
  calibration across five patient rotations. B1 applies ICBHI recording thresholds to KAUH.
- Screening: registered 8 s windows, mean aggregation primary; max aggregation is sensitivity analysis.
- H2 criteria are unchanged. Registered AST outcomes remain separate from exploratory final-model results.

## Process log

1. Verified five baseline and five AuscultTrust checkpoints locally. Retrieved missing extension
   report-epoch checkpoints from B/D/E; verified copied files against remote SHA-256 checksums.
2. Corrected KAUH analysis: 200 null re-splits, 1,000 bootstrap draws, identical patient resamples
   across each paired B2 contrast. Empty tested-pair sets produce an explicit not-tested result.
3. Started `scripts/final_evaluation.py`, logging every subprocess start/exit and exported artifact hash.
   Data hashes include ICBHI evaluation audio/annotations/split and KAUH raw inputs.
4. Started `scripts/final_ensembles.py` for frozen member inference, row-alignment checks, and
   equal-weight exports for E1/E1+E3. No parameters are selected from these new results.
5. Found and corrected an API preprocessing mismatch: annotated cycles must use the trained 5 s
   duration for PAFA; screening/windows remain 8 s. Added a focused regression check.
6. The checkpoint retrieval helper initially used a Python API unavailable on the host, then
   incorrectly expected a remote copy of an already-local checkpoint. Both retrieval issues were
   corrected; the append-only fetch log preserves the attempts. No model weights were changed.

## Artifacts and rerun commands

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_evaluation.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_ensembles.py
```

- Commands and timestamps: `logs/final-pafa-static/events.jsonl`.
- Console logs: `logs/final-pafa-static-run.log`, `logs/final-pafa-static-ensembles.log`.
- Retrieval provenance: `logs/final-pafa-static/fetch_members.log`.
- Input data manifest: `logs/final-pafa-static/input_hashes.json`.
- Freeze manifests: `outputs/train/final-pafa-static/<condition>/seed<s>/freeze.json`.
- Prediction exports and summaries: `outputs/{kauh,screening}/final-pafa-static/`.
- Extension summaries: the same roots under `extensions/{e1,e1_e3}/`.
- Candidate serving bundle: `app/bundle-final-pafa-static/`.

## Results

[Fact] The primary campaign completed at 15:25 (+07): ten frozen checkpoints and cycle/embedding exports,
ten ICBHI screening exports, fifteen KAUH exports (three conditions × five seeds), and the serving bundle.
Each KAUH seed contains 539 aligned windows from 333 filter recordings of 111 eligible patients.

### Primary KAUH positive controls

| Condition | Flip rate, mean ± seed SD | TV distance, mean ± seed SD |
| --- | --- | --- |
| Baseline | 0.4447 ± 0.0468 | 0.4372 ± 0.0433 |
| AuscultTrust | 0.2089 ± 0.0271 | 0.2062 ± 0.0247 |
| Baseline + test-time static SC | 0.2115 ± 0.0245 | 0.2104 ± 0.0212 |

[Fact] Paired patient/seed bootstrap (1,000 draws): baseline minus AuscultTrust flip rate
+0.236, 95% CI [+0.177, +0.305]; baseline minus test-time-only SC +0.233 [+0.187, +0.281].
KAUH-A passes for both contrasts.

[Fact] B2: 2 of 6 ordered filter pairs pass the 200-re-split null gate (Diaphragm→Bell,
Extended→Bell). On these pairs, mean absolute prior-matched coverage-error reduction is
+0.1130 [0.0724, 0.1532] for AuscultTrust and +0.1346 [0.0856, 0.1608] for test-time-only SC.
KAUH-B2 passes as a software-filter positive control. This is not H2 evidence on unseen hardware.

### Recording-level screening (primary mean aggregation)

Values below average the five seeds and 20 patient calibration/evaluation re-splits. They are
descriptive averages, not confidence intervals or 100 independent experiments.

| ICBHI evaluation half | Coverage | Mean set size (of 2) | Uncertain | Singleton abnormal/referral | Argmax Se | Argmax Sp |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 89.8% | 1.570 | 57.0% | 29.5% | 70.3% | 62.1% |
| AuscultTrust | 90.0% | 1.603 | 60.3% | 24.2% | 64.2% | 67.7% |

[Fact] Applying the ICBHI-calibrated recording threshold to KAUH (B1), AuscultTrust coverage
is 83.1% / 86.1% / 87.9% on Bell / Diaphragm / Extended. It does not preserve nominal 90%
coverage uniformly. Baseline values are 80.3% / 96.0% / 95.0%; the direction depends on filter.
The max-aggregation sensitivity analysis is retained in the raw summary, not chosen post hoc.

[Interpretation] SC improves paired filter consistency, but neither high pooled coverage nor
these controls establish reliable autonomous screening. The high uncertain fraction and lower
argmax sensitivity of the SC model must be reported beside coverage.

### Numerical and serving verification

- 21 focused tests passed (`tests.log`); one real-bundle API end-to-end test passed (`api-real.log`).
- Three real recordings: API cycle argmax matches the fresh cycle exports exactly; maximum
  probability difference 0.00323 across batch sizes. The initial 0.0001 probability tolerance
  failed; the explicit 0.005 numerical tolerance and observed values are preserved in the log.
- Across 27,560 exported cycles (ten units), one argmax differs from the saved mixed-precision
  training predictions (AuscultTrust seed 2); maximum probability difference across units 0.02044.
  Original benchmark results were not overwritten; see `export-parity.json`.
- The validated seed-0 bundle was activated in `app/bundle/` and the managed local API restarted.
  `/v1/health` is OK and `/v1/model` hashes match the selected manifest (`live-api-check.json`).
  The old bundle is retained in `app/bundle-backup-20261023-152025/`; no web rebuild was needed.
- Source hashes, patch snapshot and package/GPU metadata are in `logs/final-pafa-static/`.

### Completed exploratory extensions

[Fact] E1 and E1+E3 completed at 15:32 (+07), five seeds each, without SC and with trained static SC.
The following results use raw equal-weight softmax averaging, not the cycle-benchmark `k rate` rule.

| Group | KAUH flip rate without / with SC | Flip reduction, 95% CI | B2 tested pairs | B2 error reduction, 95% CI |
| --- | --- | --- | --- | --- |
| E1 | 0.318 / 0.188 | +0.130 [0.084, 0.177] | 2/6 | +0.0615 [0.0301, 0.0887] |
| E1+E3 | 0.304 / 0.190 | +0.115 [0.073, 0.159] | 0/6 | Not tested |

| ICBHI recording screening, mean aggregation | Coverage | Set size | Uncertain | Argmax Se / Sp |
| --- | --- | --- | --- | --- |
| E1, no SC | 90.0% | 1.536 | 53.6% | 68.8% / 67.9% |
| E1, static SC | 91.2% | 1.596 | 59.6% | 64.0% / 69.3% |
| E1+E3, no SC | 89.9% | 1.517 | 51.7% | 66.0% / 67.7% |
| E1+E3, static SC | 90.7% | 1.535 | 53.5% | 63.6% / 68.2% |

[Fact] E1+E3 with static SC has KAUH B1 coverage 90.7/88.9/89.4% on Bell/Diaphragm/Extended.
This does not establish nominal coverage uniformly, and these are descriptive averages over
repeated splits. E1+E3 B2 is not tested because its own no-SC ensemble has no gated discrepancy;
this differs from the primary PAFA baseline and is not a failed B2 improvement test.

[Interpretation] The extensions improve software-filter invariance. Their recording screening
results do not reproduce the cycle-benchmark `k rate` sensitivity gains; the tasks and decision
rules differ. Do not combine these numbers into a single claim of screening improvement.

### Completion checks

[Fact] Integrity checks passed for 75 KAUH and 70 screening prediction files (primary, members,
and ensemble exports): finite nonnegative probabilities summing to one. The serving split has
24 calibration patients and 23 evaluation patients, disjoint; all 12 demo recordings belong to
evaluation patients. Both campaign processes exited 0 with completion events and no failed
subprocess events. Full inventory: `logs/final-pafa-static/artifact-integrity.json`.

The final source audit detected an on-disk orchestration edit during execution: the local file-hash
helper was consolidated into `src.runs.sha256_file`, and the `hashlib` import needed for reference
hashing is present in the final script. The intermediate and final script snapshots/hashes are
both retained. Helper outputs were checked against direct SHA-256. Analysis, inference, and test
source hashes match the recorded snapshot. Completion proof: `logs/final-pafa-static/final-status.json`.

The software evaluation and serving update are complete. Outstanding work is phantom capture
and analysis, plus optional future `k rate` small-batch testing/distillation; no further model
selection is needed for the primary comparison.



## Remaining physical work

[Pending] Phantom hardware, capture, repeatability ceiling, H1, H3 and gated co-primary H2.
No phantom data was collected or evaluated in this execution. Freeze artifacts must precede capture.
