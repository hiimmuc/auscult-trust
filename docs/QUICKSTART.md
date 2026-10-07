# Quick start: train and compare the classifier arms

```
export RUN_ID=$(date +%Y%m%d-%H%M%S)        # one campaign; export the same id again to resume after an interruption
bash scripts/run.sh screen                   # cross-validate baseline, sc, sc_gain, freq_mixstyle (3 folds; SC arms with both bounds)
bash scripts/run.sh summary                  # ranking, chosen epochs, AuscultTrust arm -> outputs/train/$RUN_ID/screen.json
bash scripts/run.sh final                    # retrain at the chosen epoch on all training patients; seeds 0-9 (baseline, AuscultTrust), 0-4 (others)
bash scripts/run.sh report                   # outputs/train/$RUN_ID/final.md: table, per-device scores, paired differences, non-inferiority check
```
One run: `PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.main --config experiments/train/base.yaml,experiments/train/sc.yaml --cv_folds 3 --cv_fold 0 --seed 0`
(repo root as working directory).

Freeze a model for the conformal evaluation: `PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.export_probs outputs/train/$RUN_ID/<cell>/seed0 --ckpt report_epoch_<E>.pth`.

Where things go: `outputs/<exp>/<run_id>/<cell>/<unit>/` (report, predictions, `sc_state.npz`, `train_args.json`) and
`checkpoints/<exp>/<run_id>/<cell>/<unit>/` (weights). A finished unit is skipped; a different config in an existing unit is refused.
