"""2x2 ablation (branch x phase) plus shuffled-phase control, per class, over seeds (H1, H2).

Usage: python -m src.ablate configs/tier0_icbhi_baseline.yaml
Phase cells need phase masks in the cache (TODO: HF_Lung detector). They are skipped otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from src.data.icbhi import cycle_table
from src.data.splits import load_split
from src.eval.metrics import mean_sd
from src.models.cache import load_cycle
from src.runlog import log_run
from src.train import run

CELLS = {"fm": (False, False, False), "fm+branch": (True, False, False),
         "fm+phase": (False, True, False), "fm+branch+phase": (True, True, False),
         "fm+branch+shuffled_phase": (True, True, True)}
CLASSES = ["normal", "crackle", "wheeze", "both"]


def has_phase(cfg):
    """Check whether the first train cycle has a cached phase mask.

    Args:
        cfg: Config dict.

    Returns:
        True if phase masks exist in the cache.
    """
    df = cycle_table(cfg["icbhi_root"])
    split, _ = load_split(cfg["split_file"])
    row = next(df[df.patient.isin(split["train"])].itertuples())
    return bool(load_cycle(Path(cfg["cache_root"]) / cfg["encoder"], row)[2].any())


def summarize(results):
    """Mean ± SD over seeds per cell and metric, and gain over the `fm` baseline.

    Args:
        results: Dict cell -> list of per-seed metric dicts from `src.train.run`.

    Returns:
        Dict cell -> {metric: (mean, sd)} with per-class recall/F1 as `recall_<class>`, `f1_<class>`,
        plus `real_gain` {metric: bool}: True when |gain| exceeds the baseline seed SD.
    """
    out = {}
    for cell, runs in results.items():
        m = {k: mean_sd([r[k] for r in runs]) for k in ("icbhi_score", "ece", "coverage", "set_size")}
        for i, c in enumerate(CLASSES):
            for k in ("recall", "f1"):
                m[f"{k}_{c}"] = mean_sd([r[k][i] for r in runs])
        out[cell] = m
    base = out["fm"]
    for cell, m in out.items():
        m["real_gain"] = {k: bool(abs(m[k][0] - base[k][0]) > base[k][1])
                          for k in m if k not in ("real_gain",)}
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    h = load_split(cfg["split_file"])[1]
    phase_ok = has_phase(cfg)
    results = {}
    for cell, (br, ph, shuf) in CELLS.items():
        if ph and not phase_ok:
            print(f"skip {cell}: no cached phase masks (TODO detector)")
            continue
        c = {**cfg, "use_branch": br, "use_phase": ph, "shuffle_phase": shuf}
        results[cell] = [run(c, s) for s in cfg["seeds"]]
        for s, m in zip(cfg["seeds"], results[cell]):
            log_run(f"ablation_{cell}", c, s, h, cfg["encoder"], m)
    summ = summarize(results)
    for cell, m in summ.items():
        print(cell, " ".join(f"{k}={m[k][0]:.3f}±{m[k][1]:.3f}"
                             for k in ("icbhi_score", "recall_crackle", "recall_wheeze", "f1_crackle", "f1_wheeze")))
        print("  gain > baseline SD:", {k: v for k, v in m["real_gain"].items() if v})
