"""Ensembles at the reference's specificity: raise Se while keeping Sp (markdown on stdout).

    PYTHONPATH=. .venv-train/bin/python scripts/ensemble_sesp.py > ensemble_sesp.md

Equal-weight ensembles of `scripts/ensemble.py`. Decision rule with one parameter k: the normal probability is multiplied
by k before the argmax (k = 1 is the plain argmax). k is set so that Sp equals the Sp of BEATs PAFA alone (same arm,
seed and fold):
- `k cal` (LODO): on the calibration patients (source devices), then applied to the held-out device;
- `k*` (official split): exp of the mean log k of the LODO folds (no test cycle used);
- `k rate` (label-free, transductive like SC): the fraction of cycles called normal on the unlabelled test cycles equals
  PAFA's fraction on the same cycles (PAFA anchors Sp without labels);
- `k test` [optimistic]: matched to PAFA's Sp on the labelled test cycles themselves.
Also [optimistic]: every member at its own best epoch on test (k = 1), as the literature column of the grid.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from ensemble import ENS, G, L, REF, SC_ARMS, SEEDS
from src.evaluation.lodo import load_unit
from src.processing.icbhi import DEVICES
from src.training.patchmix_cl.reporting import per_device_report

GROUPS = [("ref", [REF])] + list(ENS.items())


def ratio(p):
    """Smallest k at which a cycle is called normal: max abnormal probability / normal probability."""
    return p[:, 1:].max(1) / np.maximum(p[:, 0], 1e-12)


def k_for_sp(p, y, sp):
    """k that gives specificity `sp` (percent) on (p, y)."""
    return float(np.quantile(ratio(p)[y == 0], sp / 100))


def k_for_rate(p, rate):
    """k at which a fraction `rate` of the cycles is called normal."""
    return float(np.quantile(ratio(p), rate))


def predict(p, k=1.0):
    return np.where(ratio(p) <= k, 0, p[:, 1:].argmax(1) + 1)


def report(y, pred, dev):
    r = per_device_report(y, pred, dev)
    return {**{m: r["all"][m] for m in ("sp", "se", "score", "hs")}, "worst": min(v["score"] for n, v in r.items() if n != "all")}


def lodo(suf):
    """rows[(name, rule)][seed][device] -> report; logk[name] -> list of log k (all folds and seeds)."""
    U = {(c, d, s): load_unit(L / f"{c}{suf}+lodo_{d.lower()}" / f"seed{s}") for _, m in GROUPS for c in m for d in DEVICES for s in SEEDS}
    rows, logk = {}, {}
    for s in SEEDS:
        for d in DEVICES:
            ref = U[REF, d, s]
            target = per_device_report(ref["cal_y"], ref["cal_p"].argmax(1), np.zeros(len(ref["cal_y"]), int))["all"]["sp"]
            for name, m in GROUPS:
                cal = np.mean([U[c, d, s]["cal_p"] for c in m], 0)
                test = np.mean([U[c, d, s]["test_p"] for c in m], 0)
                y, z = ref["test_y"], np.zeros(len(ref["test_y"]), int)
                k = k_for_sp(cal, ref["cal_y"], target)
                logk.setdefault(name, []).append(np.log(k))
                ref_pred = predict(ref["test_p"])
                rules = (("k = 1", 1.0), ("k cal", k), ("k rate", k_for_rate(test, (ref_pred == 0).mean())),
                         ("k test [optimistic]", k_for_sp(test, y, 100 * (ref_pred[y == 0] == 0).mean())))
                for rule, kk in rules:
                    rows.setdefault((name, rule), {}).setdefault(s, {})[d] = report(y, predict(test, kk), z)
    return rows, logk


def official(suf, logk):
    """rows[(name, rule)][seed] -> report on the official test split."""
    path = lambda c, s: (G / c if not suf else L / f"{c}{suf}") / f"seed{s}" / "test_preds.npz"
    rows = {}
    for s in SEEDS:
        Z = {c: np.load(path(c, s)) for _, m in GROUPS for c in m}
        y, dev = Z[REF]["labels"], Z[REF]["device"]
        best = {c: z["probs"][int(np.argmax([per_device_report(y, q, dev)["all"]["score"] for q in z["preds"]]))] for c, z in Z.items()}
        target = report(y, predict(Z[REF]["probs"][-1]), dev)["sp"]
        for name, m in GROUPS:
            last = np.mean([Z[c]["probs"][-1].astype(np.float64) for c in m], 0)
            opt = np.mean([best[c].astype(np.float64) for c in m], 0)
            rate = (predict(Z[REF]["probs"][-1]) == 0).mean()
            for rule, p, k in (("last, k = 1", last, 1.0), ("last, k*", last, float(np.exp(np.mean(logk[name])))),
                               ("last, k rate", last, k_for_rate(last, rate)),
                               ("last, k test [optimistic]", last, k_for_sp(last, y, target)),
                               ("best epoch on test [optimistic]", opt, 1.0)):
                rows.setdefault((name, rule), {})[s] = report(y, predict(p, k), dev)
    return rows


def ms(v):
    return f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}"


def main():
    name = lambda k: f"BEATs PAFA alone ({k[1]})" if k[0] == "ref" else f"{k[0]} ({k[1]})"
    for arm, suf in SC_ARMS.items():
        lr, logk = lodo(suf)
        orr = official(suf, logk)
        print(f"## Official split, {arm} (5 seeds; d = paired vs BEATs PAFA alone with the same rule)\n")
        print("| Ensemble (rule) | Sp | Se | Score | HS | Worst-dev Score | dSp | dSe | dScore |\n|" + "---|" * 9)
        for k, v in orr.items():
            r = orr["ref", k[1]]
            d = lambda m: ms([v[s][m] - r[s][m] for s in SEEDS])
            print(f"| {name(k)} | " + " | ".join(ms([v[s][m] for s in SEEDS]) for m in ("sp", "se", "score", "hs", "worst"))
                  + f" | {d('sp')} | {d('se')} | {d('score')} |")
        print(f"\n## LODO v1, {arm} (macro over the 4 held-out devices per seed; d = paired vs BEATs PAFA alone, k = 1)\n")
        print("| Ensemble (rule) | Macro Sp | Macro Se | Macro Score | dSp | dSe | dScore | Litt3200 Sp / Se / Score | AKGC417L Sp / Se / Score |\n|" + "---|" * 9)
        ref = lr["ref", "k = 1"]
        for k, v in lr.items():
            mac = lambda x, m: [np.mean([x[s][d][m] for d in DEVICES]) for s in SEEDS]
            d = lambda m: ms(np.array(mac(v, m)) - np.array(mac(ref, m)))
            dev = lambda dv: " / ".join(f"{np.mean([v[s][dv][m] for s in SEEDS]):.1f}" for m in ("sp", "se", "score"))
            print(f"| {name(k)} | {ms(mac(v, 'sp'))} | {ms(mac(v, 'se'))} | {ms(mac(v, 'score'))} | {d('sp')} | {d('se')} | {d('score')}"
                  f" | {dev('Litt3200')} | {dev('AKGC417L')} |")
        print()


if __name__ == "__main__":
    main()
