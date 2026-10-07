# Quick start: Stage 1 (v9)

```bash
export RUN_ID=$(date +%Y%m%d-%H%M%S)        # one campaign; re-export the same id to resume after an interruption

bash scripts/run_stage1.sh screen            # P0, P1, P1+P3, P4; SC arms with both sc_mode values; 3 folds each

bash scripts/run_stage1.sh summary           # ranking, CV epochs, AuscultTrust -> outputs/stage1/$RUN_ID/stage1_screen.json

bash scripts/run_stage1.sh final             # refit at the CV epoch; seeds 0-9 (P0, AuscultTrust), 0-4 (others)

bash scripts/run_stage1.sh report            # outputs/stage1/$RUN_ID/stage1_final.md with gate G
```

One unit: `python -m src.training.patchmix_cl.main --config experiments/stage1/base.yaml,experiments/stage1/P1_a1_input.yaml --cv_folds 3 --cv_fold 0 --seed 0`
(run with the training env, repo root as the working directory and `PYTHONPATH=.`; `PY=.venv-train/bin/python` for the script).

Freeze a model for Stage 2: `python -m src.training.patchmix_cl.export_probs outputs/stage1/$RUN_ID/<cell>/seed0 --ckpt report_epoch_<E>.pth`.

Where things go: `outputs/<exp>/<run_id>/<cell>/<unit>/` (reports, predictions, `sc_state.npz`, `train_args.json`) and
`checkpoints/<exp>/<run_id>/<cell>/<unit>/` (weights). A finished unit is skipped; a different config in an existing unit is refused.
