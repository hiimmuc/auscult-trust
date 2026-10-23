"""Descriptive LODO v1 tables (markdown on stdout) from the LODO units, their inference arms and the trained-SC units.

    bash scripts/lodo_h2_all.sh   # null gate and H2 per cell -> outputs/train/lodo-v1/h2_<cell>.{md,json}
    PYTHONPATH=. .venv-train/bin/python scripts/lodo_tables.py > lodo_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from grid import unit_metrics
from src.evaluation.lodo import collect, coverage
from src.processing.icbhi import DEVICES

R = Path("outputs/train/lodo-v1")
CELLS = ["ast+ft+ce", "ast+ft+pafa", "ast+ft+patchmix_cl", "htsat+ft+ce", "htsat+ft+patchmix_cl", "htsat+ft+sg_scl",
         "beats+ft+ce", "beats+ft+patchmix_cl", "beats+ft+sg_scl", "beats+ft+pafa", "beats+ft+qlung"]
SC = ["ast+ft+patchmix_cl", "ast+ft+pafa", "htsat+ft+patchmix_cl", "htsat+ft+sg_scl", "beats+ft+patchmix_cl", "beats+ft+pafa",
      "beats+ft+sg_scl"]
ARMS = [("ttsc_static", "TT-SC static"), ("ttsc_dynamic", "TT-SC dynamic"), ("a2", "A2"), ("tta8", "TTA-EQ"),
        ("sc+sc_static", "trained SC static"), ("sc", "trained SC dynamic")]
EXTRA = {"beats+ft+pafa": [("beats+ft+pafa_pcmcl+sc+sc_static", "PAFA + PC-MCL, trained SC static"),
                           ("beats+ft+pafa_pcmcl+sc", "PAFA + PC-MCL, trained SC dynamic")]}
OFFICIAL = ([(c, f"{c}+sc+sc_static", Path("outputs/train/grid-v1") / c) for c in SC]
            + [(c, f"{c}+sc", Path("outputs/train/grid-v1") / c) for c in ["beats+ft+patchmix_cl", "beats+ft+pafa", "beats+ft+sg_scl"]]
            + [("beats+ft+pafa_pcmcl", f"beats+ft+pafa_pcmcl{s}", Path("outputs/train/pcmcl-v1/beats+ft+pafa_pcmcl"))
               for s in ("+sc+sc_static", "+sc")])
DEV = DEVICES
f1 = lambda x: f"{x:.1f}"


def per_seed(units, arm):
    """seed -> {device: dict(score, sp, se, hs, cov, absd, size)}"""
    out = {}
    for d in DEV:
        for s, u in units.get(d, {}).get(arm, {}).items():
            c = coverage(u["cal_p"], u["cal_y"], u["test_p"], u["test_y"])
            out.setdefault(s, {})[d] = {**{k: u["score"][k] for k in ("score", "sp", "se", "hs")},
                                        "cov": 100 * c["coverage"], "delta": c["delta"], "absd": abs(c["delta"]), "size": c["size"]}
    return out


def macro(ps, key, devs=DEV):
    return np.array([np.mean([ps[s][d][key] for d in devs]) for s in sorted(ps)])


ALL = CELLS + [f"{c}+{a}" for c in CELLS for a, _ in ARMS] + [a for v in EXTRA.values() for a, _ in v]
units = collect(R, ALL, DEV, range(5))
P = {a: per_seed(units, a) for a in ALL}
P = {a: v for a, v in P.items() if v}
noakg = [d for d in DEV if d != "AKGC417L"]

print("## T1 no-SC cells per held-out device (5 seeds, mean)\n")
print("| Cell | " + " | ".join(f"{d} Score" for d in DEV) + " | Macro Score | Macro w/o AKGC417L | " + " | ".join(f"{d} Cov / set" for d in DEV) + " |")
print("|" + "---|" * (2 * len(DEV) + 3))
for c in CELLS:
    ps = P[c]
    sc = [np.mean([ps[s][d]["score"] for s in ps]) for d in DEV]
    cv = [f"{np.mean([ps[s][d]['cov'] for s in ps]):.1f} / {np.mean([ps[s][d]['size'] for s in ps]):.2f}" for d in DEV]
    m, m2 = macro(ps, "score"), macro(ps, "score", noakg)
    print(f"| {c} | " + " | ".join(map(f1, sc)) + f" | {m.mean():.1f} ± {m.std(ddof=1):.1f} | {m2.mean():.1f} ± {m2.std(ddof=1):.1f} | " + " | ".join(cv) + " |")

print("\n## T1b Sp / Se / HS per held-out device (5 seeds, mean)\n")
print("| Cell | " + " | ".join(f"{d} Sp / Se / HS" for d in DEV) + " |\n|" + "---|" * (len(DEV) + 1))
for c in CELLS:
    ps = P[c]
    print(f"| {c} | " + " | ".join(" / ".join(f1(np.mean([ps[s][d][k] for s in ps])) for k in ("sp", "se", "hs")) for d in DEV) + " |")

print("\n## T2 inference and trained-SC arms vs the source cell (paired by seed; macro over the 4 folds; Litt3200 fold)\n")
print("| Cell | Arm | dScore macro | dScore Litt3200 | dCov Litt3200 | d|Delta| macro | dSet size macro | Set size Litt3200 (arm) |")
print("|---|---|---|---|---|---|---|---|")
for c in CELLS:
    b = P[c]
    for arm, name in [(f"{c}+{a}", n) for a, n in ARMS] + EXTRA.get(c, []):
        if len(P.get(arm, {})) < 5:  # arms run on some cells only
            continue
        x = P[arm]
        ss = sorted(set(b) & set(x))
        dm = np.array([np.mean([x[s][d]["score"] - b[s][d]["score"] for d in DEV]) for s in ss])
        dl = np.array([x[s]["Litt3200"]["score"] - b[s]["Litt3200"]["score"] for s in ss])
        dc = np.mean([x[s]["Litt3200"]["cov"] - b[s]["Litt3200"]["cov"] for s in ss])
        da = np.mean([np.mean([x[s][d]["absd"] - b[s][d]["absd"] for d in DEV]) for s in ss])
        dz = np.mean([np.mean([x[s][d]["size"] - b[s][d]["size"] for d in DEV]) for s in ss])
        zl = np.mean([x[s]["Litt3200"]["size"] for s in ss])
        print(f"| {c} | {name} | {dm.mean():+.1f} ± {dm.std(ddof=1):.1f} | {dl.mean():+.1f} ± {dl.std(ddof=1):.1f} | {dc:+.1f} | {da:+.3f} | {dz:+.2f} | {zl:.2f} |")

print("\n## T3 method vs CE on the same encoder (no SC, paired by seed, macro Score over 4 folds)\n")
print("| Encoder | Method | dScore macro | dScore w/o AKGC417L | dScore Litt3200 |\n|---|---|---|---|---|")
for c in CELLS:
    enc = c.split("+")[0]
    if c.endswith("+ce") or f"{enc}+ft+ce" not in P:
        continue
    b, x = P[f"{enc}+ft+ce"], P[c]
    ss = sorted(set(b) & set(x))
    d1 = np.array([np.mean([x[s][d]["score"] - b[s][d]["score"] for d in DEV]) for s in ss])
    d2 = np.array([np.mean([x[s][d]["score"] - b[s][d]["score"] for d in noakg]) for s in ss])
    d3 = np.array([x[s]["Litt3200"]["score"] - b[s]["Litt3200"]["score"] for s in ss])
    print(f"| {enc} | {c.split('+')[-1]} | {d1.mean():+.1f} ± {d1.std(ddof=1):.1f} | {d2.mean():+.1f} ± {d2.std(ddof=1):.1f} | {d3.mean():+.1f} ± {d3.std(ddof=1):.1f} |")

print("\n## T4 trained SC on the official split vs the no-SC unit of the same cell and seed (last epoch)\n")
cols = [("sp", "Sp"), ("se", "Se"), ("score", "Score"), ("hs", "HS"), ("two_cls_score", "2c Score"), ("worst_dev_score", "Worst-dev Score"),
        ("cov", "Cov"), ("worst_dev_cov", "Worst-dev Cov"), ("set_size", "Set size")]
print("| Cell | " + " | ".join(n for _, n in cols) + " | dScore vs no SC | dWorst-dev Score |\n|" + "---|" * (len(cols) + 3))
for _, c, base in OFFICIAL:
    r = [unit_metrics(R / c / f"seed{s}" / "test_preds.npz", "last") for s in range(5)]
    g = [unit_metrics(base / f"seed{s}" / "test_preds.npz", "last") for s in range(5)]
    cell = lambda k: "{:.{p}f} ± {:.{p}f}".format(np.mean([x[k] for x in r]), np.std([x[k] for x in r], ddof=1), p=2 if k == "set_size" else 1)
    d = np.array([a["score"] - b["score"] for a, b in zip(r, g)])
    w = np.array([a["worst_dev_score"] - b["worst_dev_score"] for a, b in zip(r, g)])
    print(f"| {c} | " + " | ".join(cell(k) for k, _ in cols) + f" | {d.mean():+.1f} ± {d.std(ddof=1):.1f} | {w.mean():+.1f} ± {w.std(ddof=1):.1f} |")
