# checkpoints/ (gitignored)

Trained weights, one directory per run: `checkpoints/<exp>/<run_id>/<cell>/<unit>/`, mirroring `outputs/<exp>/<run_id>/...`
(reports, predictions, `sc_state.npz`, `train_args.json`). `run_id` is `YYYYMMDD-HHMMSS[_tag]`; `latest` points at the newest.

- Screening folds (`cv<k>`) keep no weights; only the report in `outputs/`.
- Fixed refits (`seed<s>`) keep `report_epoch_<E>.pth` (the registered CV epoch): the weights frozen for Stage 2.
- `last.pth` exists only while a unit runs (resume state) and is removed when it finishes.
- Freeze for Stage 2 with `python -m src.training.patchmix_cl.export_probs <outputs unit dir> --ckpt report_epoch_<E>.pth`,
  which writes `export/model.sha256`.
- Legacy v8 runs are under `stage1_v8/legacy-v8/` (`scripts/migrate_legacy_layout.py`).
