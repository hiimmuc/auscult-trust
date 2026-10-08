"""Does a frozen encoder keep abnormal-sound information that the argmax decision throws away?

For every cell of a run, reads the saved test softmax of the last epoch (no epoch selection) and reports, mean +- SD
over seeds: one-vs-rest AUROC per class, AUROC of abnormal vs normal (score = 1 - p_normal), Sp and Se at the argmax,
and the 4 x 4 confusion matrix (rows = true class, % of the row). High abnormal AUROC with low Se means the decision is
biased towards normal; AUROC near 0.5 means the embedding lacks the information. Diagnosis only: nothing is tuned.

Usage: python -m src.evaluation.fm_diagnose --run repro-wave1 [--cells substring ...] [--epoch 50]
"""

import argparse

import numpy as np
from scipy.stats import rankdata
from src.paths import OUTPUTS

CLASSES = ["normal", "crackle", "wheeze", "both"]


def auroc(score, positive):
    """Rank-based AUROC (ties averaged)."""
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def diagnose(npz, epoch):
    """Metrics of one unit at `epoch` (1-based, `None` = last saved)."""
    z = np.load(npz)
    i = len(z["epochs"]) - 1 if epoch is None else int(np.where(z["epochs"] == epoch)[0][0])
    p, y = z["probs"][i].astype(np.float64), z["labels"]
    pred = p.argmax(1)
    out = {f"auroc_{c}": auroc(p[:, k], y == k) for k, c in enumerate(CLASSES)}
    out["auroc_abnormal"] = auroc(1 - p[:, 0], y != 0)
    out["sp"] = 100 * float((pred[y == 0] == 0).mean())
    out["se"] = 100 * float((pred[y > 0] == y[y > 0]).mean())
    out["pred_normal_share"] = 100 * float((pred == 0).mean())
    out["confusion"] = np.stack([np.bincount(pred[y == k], minlength=4) / (y == k).sum() * 100 for k in range(4)])
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--cells", nargs="*", default=[""])
    a.add_argument("--epoch", type=int, default=None)
    a = a.parse_args()
    root = OUTPUTS / "train" / a.run
    ms = lambda v: "{:.2f}±{:.2f}".format(np.mean(v), np.std(v))
    print("| cell | seeds | AUROC abnormal | AUROC normal | crackle | wheeze | both | Sp | Se | pred normal % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    conf = {}
    for cell in sorted(p for p in root.iterdir() if p.is_dir() and any(s in p.name for s in a.cells)):
        rs = [diagnose(f, a.epoch) for f in sorted(cell.glob("seed*/test_preds.npz"))]
        if not rs:
            continue
        g = lambda k: ms([r[k] for r in rs])
        print(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cell.name, len(rs), g("auroc_abnormal"), g("auroc_normal"), g("auroc_crackle"), g("auroc_wheeze"),
                g("auroc_both"), g("sp"), g("se"), g("pred_normal_share"),
            )
        )
        conf[cell.name] = np.mean([r["confusion"] for r in rs], 0)
    for name, c in conf.items():
        print("\n{} confusion (true row, predicted column, % of row; order {})".format(name, "/".join(CLASSES)))
        print("\n".join("  " + " ".join("{:5.1f}".format(v) for v in row) for row in c))


if __name__ == "__main__":
    main()
