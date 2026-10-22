"""Softmax-averaging ensembles of fine-tuned cells at the last epoch: official split and LODO v1, without SC and with
trained SC static (markdown on stdout).

    PYTHONPATH=. .venv-train/bin/python scripts/ensemble.py > ensemble_tables.md

Members of one ensemble share seed (and LODO fold); their cycle order is checked. Weightings: `equal` = mean softmax;
`fit` = member weights on the simplex minimising the NLL of the calibration patients of each LODO unit (source devices
only, never test cycles). The official split has no labelled held-out set, so it gets `fit*`: the mean of the
LODO-fitted weights of the same members (all folds and seeds). Reference = BEATs PAFA alone (same SC arm), paired by seed.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "scripts")
from grid import unit_metrics
from src.evaluation.lodo import coverage, load_unit
from src.processing.icbhi import DEVICES
from src.training.patchmix_cl.reporting import per_device_report

G, L = Path("outputs/train/grid-v1"), Path("outputs/train/lodo-v1")
E1 = ["beats+ft+pafa", "beats+ft+sg_scl", "beats+ft+patchmix_cl"]
E2 = ["ast+ft+patchmix_cl", "ast+ft+pafa"]
E3 = ["htsat+ft+patchmix_cl", "htsat+ft+sg_scl"]
ENS = {"E1 BEATs": E1, "E2 AST": E2, "E3 HTS-AT": E3, "E1+E2": E1 + E2, "E1+E3": E1 + E3, "E2+E3": E2 + E3,
       "E1+E2+E3": E1 + E2 + E3}
REF = "beats+ft+pafa"
SEEDS = range(5)
SC_ARMS = {"no SC": "", "trained SC static": "+sc+sc_static"}


def simplex(t):
    w = np.exp(t - t.max())
    return w / w.sum()


def fit_weights(ps, y):
    """Simplex weights of the K member softmaxes `ps` (K, N, C) minimising the NLL of their weighted mean."""
    nll = lambda t: -np.log(np.tensordot(simplex(t), ps, 1)[np.arange(len(y)), y] + 1e-12).mean()
    return simplex(minimize(nll, np.zeros(len(ps)), method="L-BFGS-B").x)


def same_cycles(units):
    a = units[0]
    assert all(np.array_equal(u["cal_y"], a["cal_y"]) and np.array_equal(u["cal_pat"], a["cal_pat"])
               and np.array_equal(u["test_y"], a["test_y"]) and np.array_equal(u["test_pat"], a["test_pat"]) for u in units)


def lodo(suf):
    """rows[(name, weighting)][seed][device] -> metrics; fitted[name] -> list of weight vectors."""
    U = {(c, d, s): load_unit(L / f"{c}{suf}+lodo_{d.lower()}" / f"seed{s}") for c in E1 + E2 + E3 for d in DEVICES for s in SEEDS}
    rows, fitted = {}, {}
    for name, m in [("ref", [REF])] + list(ENS.items()):
        for s in SEEDS:
            for d in DEVICES:
                us = [U[c, d, s] for c in m]
                same_cycles(us)
                cal, test = np.stack([u["cal_p"] for u in us]), np.stack([u["test_p"] for u in us])
                y, cy = us[0]["test_y"], us[0]["cal_y"]
                ws = {"equal": np.full(len(m), 1 / len(m))}
                if len(m) > 1:
                    ws["fit"] = fit_weights(cal, cy)
                    fitted.setdefault(name, []).append(ws["fit"])
                for wn, w in ws.items():
                    p, cp = np.tensordot(w, test, 1), np.tensordot(w, cal, 1)
                    r = per_device_report(y, p.argmax(1), np.zeros(len(y), int))["all"]
                    c = coverage(cp, cy, p, y)
                    rows.setdefault((name, wn), {}).setdefault(s, {})[d] = {
                        **{k: r[k] for k in ("score", "sp", "se")}, "absd": abs(c["delta"]), "size": c["size"]}
    return rows, fitted


def official(suf, fitted):
    """rows[(name, weighting)][seed] -> unit_metrics of the ensemble softmax."""
    path = lambda c, s: (G / c if not suf else L / f"{c}{suf}") / f"seed{s}" / "test_preds.npz"
    rows, tmp = {}, Path(tempfile.mkdtemp()) / "e.npz"
    for name, m in [("ref", [REF])] + list(ENS.items()):
        ws = {"equal": np.full(len(m), 1 / len(m))}
        if len(m) > 1:
            ws["fit*"] = np.mean(fitted[name], 0)
        for s in SEEDS:
            zs = [np.load(path(c, s)) for c in m]
            assert all(np.array_equal(z["labels"], zs[0]["labels"]) and np.array_equal(z["patient"], zs[0]["patient"]) for z in zs)
            for wn, w in ws.items():
                p = np.tensordot(w, np.stack([z["probs"][-1].astype(np.float64) for z in zs]), 1)
                np.savez(tmp, epochs=[0], preds=p.argmax(1)[None], probs=p[None], labels=zs[0]["labels"],
                         device=zs[0]["device"], patient=zs[0]["patient"])
                rows.setdefault((name, wn), {})[s] = unit_metrics(tmp, "last")
    return rows


ms = lambda v, p=1: f"{np.mean(v):.{p}f} ± {np.std(v, ddof=1):.{p}f}"
label = lambda k: "BEATs PAFA alone (reference)" if k[0] == "ref" else f"{k[0]} ({k[1]})"


def main():
    for arm, suf in SC_ARMS.items():
        lr, fitted = lodo(suf)
        orr = official(suf, fitted)
        print(f"## Official split, {arm} (last epoch, 5 seeds; d = paired vs BEATs PAFA {arm})\n")
        cols = [("sp", "Sp"), ("se", "Se"), ("score", "Score"), ("hs", "HS"), ("worst_dev_score", "Worst-dev Score"),
                ("cov", "Cov"), ("worst_dev_cov", "Worst-dev Cov"), ("set_size", "Set size")]
        print("| Ensemble | " + " | ".join(n for _, n in cols) + " | dScore | dWorst-dev Score |\n|" + "---|" * (len(cols) + 3))
        ref = orr["ref", "equal"]
        for k, v in orr.items():
            d = [v[s]["score"] - ref[s]["score"] for s in SEEDS]
            w = [v[s]["worst_dev_score"] - ref[s]["worst_dev_score"] for s in SEEDS]
            print(f"| {label(k)} | " + " | ".join(ms([v[s][c] for s in SEEDS], 2 if c == "set_size" else 1) for c, _ in cols)
                  + f" | {np.mean(d):+.1f} ± {np.std(d, ddof=1):.1f} | {np.mean(w):+.1f} ± {np.std(w, ddof=1):.1f} |")
        print(f"\n## LODO v1, {arm} (Score per held-out device, mean of 5 seeds; macro = mean over the 4 folds per seed)\n")
        print("| Ensemble | " + " | ".join(DEVICES) + " | Macro Score | dMacro vs ref | dLitt3200 vs ref | Litt3200 Sp / Se | Macro abs(Delta) | Macro set size |")
        print("|" + "---|" * (len(DEVICES) + 7))
        ref = lr["ref", "equal"]
        for k, v in lr.items():
            mac = [np.mean([v[s][d]["score"] for d in DEVICES]) for s in SEEDS]
            dm = [mac[s] - np.mean([ref[s][d]["score"] for d in DEVICES]) for s in SEEDS]
            dl = [v[s]["Litt3200"]["score"] - ref[s]["Litt3200"]["score"] for s in SEEDS]
            mean = lambda d, key: np.mean([v[s][d][key] for s in SEEDS])
            print(f"| {label(k)} | " + " | ".join(f"{mean(d, 'score'):.1f}" for d in DEVICES) + f" | {ms(mac)} | {np.mean(dm):+.1f} ± {np.std(dm, ddof=1):.1f}"
                  f" | {np.mean(dl):+.1f} ± {np.std(dl, ddof=1):.1f} | {mean('Litt3200', 'sp'):.1f} / {mean('Litt3200', 'se'):.1f}"
                  f" | {np.mean([mean(d, 'absd') for d in DEVICES]):.3f} | {np.mean([mean(d, 'size') for d in DEVICES]):.2f} |")
        print(f"\n## Fitted member weights, {arm} (mean ± SD over 4 folds x 5 seeds)\n")
        print("| Ensemble | Weights |\n|---|---|")
        for name, m in ENS.items():
            f = np.array(fitted[name])
            print(f"| {name} | " + ", ".join(f"{c} {f[:, i].mean():.2f} ± {f[:, i].std(ddof=1):.2f}" for i, c in enumerate(m)) + " |")
        print()


if __name__ == "__main__":
    main()
