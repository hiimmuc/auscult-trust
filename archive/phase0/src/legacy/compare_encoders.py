"""Compare frozen encoders under one protocol (plain FM, no branch, no phase).

Selection metric: grouped 5-fold CV over the train+val patients (logistic probe on mean-pooled
embeddings), then the MLP-head val Score. Official-test Score is reported but is NOT used to pick.

Usage: python -m src.legacy.compare_encoders configs/legacy/encoder_compare.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.icbhi import cycle_table
from src.data.splits import load_split, split_frames
from src.eval.metrics import class_report, icbhi_score, mean_sd
from src.runlog import ckpt_path, log_run, start_summary
from src.legacy.train import fit_eval, tensors


def grouped_cv(x, y, groups, k=5, pca=256, C=0.01):
    """Patient-grouped CV of a standardised, PCA-reduced logistic probe.

    Args:
        x: (n, d) features.
        y: (n,) labels.
        groups: (n,) patient ids.
        k: Folds.
        pca: PCA components.
        C: Inverse L2 strength.

    Returns:
        Dict metric -> per-fold list: `score`, `f1_crackle`, `f1_wheeze`.
    """
    out = {"score": [], "f1_crackle": [], "f1_wheeze": []}
    for tr, te in GroupKFold(k).split(x, y, groups):
        m = make_pipeline(StandardScaler(), PCA(min(pca, x.shape[1]), random_state=0),
                          LogisticRegression(C=C, max_iter=3000)).fit(x[tr], y[tr])
        p = m.predict(x[te])
        f1 = class_report(y[te], p)["f1"]
        out["score"].append(icbhi_score(y[te], p))
        out["f1_crackle"].append(f1[1])
        out["f1_wheeze"].append(f1[2])
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    start_summary(path.stem)
    split, h = load_split(cfg["split_file"])
    part = split_frames(cycle_table(cfg["icbhi_root"]), split)
    dev = pd.concat([part["train"], part["val"]])
    rows = []
    for enc in cfg["encoders"]:
        data = {k: tensors(Path(cfg["cache_root"]) / enc, v) for k, v in part.items()}
        x = np.concatenate([data[k][0]["emb"].float().mean(1).cpu().numpy() for k in ("train", "val")])
        cv = grouped_cv(x, dev.label.values, dev.patient.values)
        c = {**cfg, "encoder": enc, "use_branch": False, "use_phase": False}
        res = [fit_eval(c, s, data, ckpt=ckpt_path(path.stem, enc, s))[0] for s in cfg["seeds"]]
        for s, m in zip(cfg["seeds"], res):
            log_run(path.stem, enc, s, c, h, enc, m)
        rows.append({"encoder": enc, **{f"cv_{k}": "%.3f ± %.3f" % mean_sd(v) for k, v in cv.items()},
                     "mlp_val_score": "%.3f ± %.3f" % mean_sd([m["val_icbhi_score"] for m in res]),
                     "test_score": "%.3f ± %.3f" % mean_sd([m["icbhi_score"] for m in res]),
                     "test_f1_crackle": "%.3f ± %.3f" % mean_sd([m["f1"][1] for m in res]),
                     "test_f1_wheeze": "%.3f ± %.3f" % mean_sd([m["f1"][2] for m in res]),
                     "test_coverage": "%.3f ± %.3f" % mean_sd([m["coverage"] for m in res]),
                     "test_set_size": "%.2f ± %.2f" % mean_sd([m["set_size"] for m in res])})
        print(rows[-1], flush=True)
    print(pd.DataFrame(rows).set_index("encoder").T.to_string())
