# Final model and evaluation decision (23 October 2026)

[Fact] The user selected the following configuration after reviewing report v5.0:

| Role | Configuration | Fixed epoch |
| --- | --- | --- |
| Project baseline | BEATs + PAFA, fine-tuned, no SC | 100 |
| AuscultTrust | BEATs + PAFA trained and evaluated with static SC | 100 |
| Extension E1 | Equal-weight BEATs PAFA, SG-SCL, Patch-Mix CL | 100 / 50 / 50 |
| Extension E1+E3 | E1 plus HTS-AT Patch-Mix CL and SG-SCL, equal weights over all five members | 100 / 50 / 50 / 50 / 50 |

Both no-SC and trained-static-SC variants of the extensions are retained for comparison. The
extensions remain exploratory; they do not replace the primary baseline/AuscultTrust comparison.
`k rate` is the additional classification rule defined in the ensemble report. Raw mean softmax
is used for the existing conformal analysis; the classification rule does not establish a
conformal result. Best-on-test epochs remain optimistic reference results only.

Static SC corrects the 50–2000 Hz common band, with the saved arithmetic reference. New-device
coefficients use unlabelled target audio. No further parameter selection on KAUH or phantom data.

## Execution and provenance

- Seeds 0–4, paired across conditions; serving seed 0 is fixed in advance, not chosen by performance.
- Main comparison: `grid-v1/beats+ft+pafa` vs `lodo-v1/beats+ft+pafa+sc+sc_static`.
- Isolation contrast: baseline weights with test-time static SC using the paired SC run's reference.
- Campaign: `final-pafa-static`; reproducible entry point: `scripts/final_evaluation.py`.
- Command events: `logs/final-pafa-static/events.jsonl`; console: `logs/final-pafa-static-run.log`.
- Input-data hashes: `logs/final-pafa-static/input_hashes.json`.
- Frozen manifests: `outputs/train/final-pafa-static/<condition>/seed<s>/freeze.json`.
- KAUH exports/analysis: `outputs/kauh/final-pafa-static/`.
- Recording-level exports/analysis: `outputs/screening/final-pafa-static/`.
- Candidate serving bundle: `app/bundle-final-pafa-static/`; validate before activation.

## Statistical interpretation

The registered AST study and H2 definition are unchanged. Model selection on the existing grid
and LODO results is disclosed. KAUH is a positive control using software filters, not independent
hardware validation. No claim of restored coverage follows from classification Score gains.

Before running KAUH analysis, its implementation was corrected to use 200 null re-splits and
1,000 bootstrap draws, with identical patient resamples in each paired B2 contrast. Earlier
reports are preserved. If no pair passes the null gate, B2 is reported as not tested.

[Pending] Phantom recordings and H1/H3/co-primary H2 require physical hardware and recordings.
Weights and the reference spectrum must remain frozen before the first phantom recording.
