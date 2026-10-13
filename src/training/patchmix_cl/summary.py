"""Cross-validation screening table and final table of a training campaign.

Reads one campaign, `outputs/train/<run_id>/<cell>/<unit>/report.json` (default run: `latest`):
    cell `baseline`, `sc`, `sc+sc_static`, ...   units `cv<k>` (screening fold) or `seed<s>` (final refit)

    python -m src.training.patchmix_cl.summary screen [--run RUN_ID] [--folds 3]
        -> screening table; writes <run>/screen.json (CV epoch and sc_mode per arm, selected `auscult_trust` arm, plan)
    python -m src.training.patchmix_cl.summary gate --run screen-ast --base_cell baseline_e14 --base_epoch 14
        -> paired ΔScore of --arm_cell against another refit cell (sensitivity analysis of the non-inferiority check)
    python -m src.training.patchmix_cl.summary final [--run RUN_ID] [--boot 1000]
        -> final table (primary = epoch chosen by CV; extra column = best epoch on test, optimistic), paired differences
           against the baseline, and the non-inferiority check of the selected arm against the baseline
           (95% t-interval lower bound of the Score difference above -1.5 points); writes <run>/final.md

Spectrum-correction (SC) arms are screened with both `sc_mode` values (`<arm>` = dynamic, `<arm>+sc_static` = static); the
higher mean CV Score is kept, ties go to dynamic. AuscultTrust = the better of `sc` / `sc_gain` by mean CV Score (tie: `sc`).
"""

import argparse
import json
import re

import numpy as np
from scipy import stats
from src.paths import OUTPUTS

from .config import EXP
from .reporting import cv_epoch, mean_cv_curve, per_device_report

BASE, SC_ARMS = "baseline", ("sc", "sc_gain")
STATIC = "+sc_static"
NON_INFERIORITY_MARGIN = (
    -1.5
)  # Score points; about half the published gap between plain AST fine-tuning and Patch-Mix CL
BENCHMARK_SEEDS = range(5)


def run_dir(run):
    return OUTPUTS / EXP / run


def load_units(root, unit_pattern):
    """{(cell, index): report dict} of finished units whose name matches `unit_pattern` (one group: the index)."""
    out = {}
    for f in root.glob("*/*/report.json"):
        m = re.fullmatch(unit_pattern, f.parent.name)
        if m:
            out[(f.parent.parent.name, int(m.group(1)))] = json.loads(f.read_text())
    return out


def screen(a):
    root = run_dir(a.run)
    units = load_units(root, r"cv(\d+)")
    cells, rows, missing = sorted({c for c, _ in units}), {}, []
    for c in cells:
        folds = [units.get((c, k)) for k in range(a.folds)]
        if any(f is None for f in folds):
            missing.append(c)
            continue
        curves = [[h["val_score"] for h in f["history"]] for f in folds]
        ep = cv_epoch(curves)
        rows[c] = {
            "cell": c,
            "epoch": ep,
            "cv_score": float(mean_cv_curve(curves)[ep - 1]),
            "fold_scores": [float(x[ep - 1]) for x in curves],
            **{k: float(np.mean([f["history"][ep - 1]["val"][k] for f in folds])) for k in ("sp", "se", "hs")},
            "sc_mode": "static" if c.endswith(STATIC) else "dynamic",
        }
    arms = {}
    for c, r in rows.items():  # per arm keep the better sc_mode; ties go to dynamic
        arm = c.removesuffix(STATIC)
        best = arms.get(arm)
        if (
            best is None
            or r["cv_score"] > best["cv_score"]
            or (r["cv_score"] == best["cv_score"] and r["sc_mode"] == "dynamic")
        ):
            arms[arm] = r
    for arm, r in arms.items():
        r["sc_mode"] = r["sc_mode"] if arm in SC_ARMS else None
    ranking = sorted(arms, key=lambda x: -arms[x]["cv_score"])
    sc = [x for x in SC_ARMS if x in arms]
    trust = (
        max(sc, key=lambda x: (arms[x]["cv_score"], x == SC_ARMS[0])) if sc else None
    )  # tie: sc
    print("| rank | arm | cell | sc_mode | CV epoch | Sp | Se | mean CV Score | HS | fold Scores |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(ranking, 1):
        r = arms[x]
        print(
            "| {} | {} | {} | {} | {} | {:.1f} | {:.1f} | {:.2f} | {:.1f} | {} |".format(
                i,
                x,
                r["cell"],
                r["sc_mode"] or "-",
                r["epoch"],
                r["sp"],
                r["se"],
                r["cv_score"],
                r["hs"],
                ", ".join("{:.1f}".format(s) for s in r["fold_scores"]),
            )
        )
    if missing:
        print("\nnot finished (all {} folds needed): {}".format(a.folds, ", ".join(missing)))
    (root / "screen.json").write_text(
        json.dumps({"arms": arms, "ranking": ranking, "auscult_trust": trust}, indent=1)
    )
    print("\nAuscultTrust = {} (better mean CV Score of sc / sc_gain, tie: sc)".format(trust))


def _score_counts(y, pred, patients):
    """Per-patient counts for the ICBHI Score: (normal n, normal correct, abnormal n, abnormal correct, abn. any-abnormal)."""
    ids = np.unique(patients)
    idx = np.searchsorted(ids, patients)
    n, a = y == 0, y != 0
    c = np.zeros((len(ids), 5))
    np.add.at(c[:, 0], idx, n)
    np.add.at(c[:, 1], idx, n & (pred == 0))
    np.add.at(c[:, 2], idx, a)
    np.add.at(c[:, 3], idx, a & (pred == y))
    np.add.at(c[:, 4], idx, a & (pred > 0))
    return c


def _score_from_counts(c, two_cls=False):
    """ICBHI Score (%) from (summed) counts; works on (..., 5) arrays."""
    sp = c[..., 1] / np.maximum(c[..., 0], 1)
    se = (c[..., 4] if two_cls else c[..., 3]) / np.maximum(c[..., 2], 1)
    return 100 * (sp + se) / 2


def _t_ci(d, level=0.95):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float("nan"), float("nan")
    h = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def _ms(x):
    x = np.asarray(x, float)
    return "{:.2f} ± {:.2f}".format(x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def _collect(root, runs, arms):
    """Per arm: seeds, per-seed test reports at the registered epoch, test-best score, predictions."""
    res = {}
    for arm, r in arms.items():
        ep, per = r["epoch"], {}
        for s in sorted(s for (c, s) in runs if c == r["cell"]):
            rep = runs[(r["cell"], s)]
            z = np.load(root / r["cell"] / "seed{}".format(s) / "test_preds.npz")
            pred = z["preds"][list(z["epochs"]).index(ep)]
            per[s] = {
                "test": rep["fixed"][str(ep)]["test"],
                "best": rep["test_best_optimistic"]["test"]["all"]["score"],
                "pred": pred,
                "labels": z["labels"],
                "patient": z["patient"],
                "device": z["device"],
            }
        res[arm] = {"epoch": ep, "cell": r["cell"], "per": per}
    return res


def both_sides_patients(split_file=None):
    """Patients whose recordings appear in both the train and the test list of `official_split.txt`."""
    from .config import build_parser

    split_file = split_file or "{}/official_split.txt".format(build_parser().parse_args([]).data_folder)
    sides = {}
    for line in open(split_file).read().splitlines():
        stem, side = line.strip().split("\t")
        sides.setdefault(stem.split("_")[0], set()).add(side)
    return {p for p, v in sides.items() if len(v) > 1}


def _dropped_counts(res, both):
    p = next(iter(next(iter(res.values()))["per"].values()))
    return int(np.isin(p["patient"], list(both)).sum()), int(len(p["patient"]))


def final(a):
    root = run_dir(a.run)
    scr = json.loads((root / "screen.json").read_text())
    res = _collect(root, load_units(root, r"seed(\d+)"), scr["arms"])
    trust = scr["auscult_trust"]
    L = [
        "## Final results (official 60/40 split, 4-class, refit on all training patients)\n",
        "Primary = epoch chosen by 3-fold CV screening (no test selection), seeds 0-4. The last column is the best epoch on test, as "
        "in the Patch-Mix CL and SG-SCL papers: **optimistic, for comparison only**.\n",
        "| arm | cell | CV epoch | seeds | Sp | Se | Score | HS | macro-F1 | 2-cls Score | best-on-test Score (optimistic) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, r in res.items():
        per = {s: p for s, p in r["per"].items() if s in BENCHMARK_SEEDS}
        tests = [p["test"]["all"] for p in per.values()]
        get = lambda k: [t[k] for t in tests]
        L.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                arm + (" (AuscultTrust)" if arm == trust else ""),
                r["cell"],
                r["epoch"],
                len(tests),
                _ms(get("sp")),
                _ms(get("se")),
                _ms(get("score")),
                _ms(get("hs")),
                _ms(get("macro_f1")),
                _ms(get("two_cls_score")),
                _ms([p["best"] for p in per.values()]),
            )
        )
    L.append(
        "\nPublished anchors, same split, 4-class: AST fine-tuning 59.55 ± 0.88; Patch-Mix CL 62.37 ± 0.61; SG-SCL 61.71 ± 1.61 "
        "(all best epoch on test)."
    )

    both = both_sides_patients()
    L += [
        "\n### Official test without the patients on both sides of the official split\n",
        "Patients {} have recordings in both the official train and test lists; their test cycles are dropped here ({} of {} "
        "cycles). Same seeds 0-4 and epochs as above. `Both` = recall of the crackle-and-wheeze class.\n".format(
            ", ".join(sorted(both)) or "none", *_dropped_counts(res, both)
        ),
        "| arm | seeds | Sp | Se | Score | HS | Both recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, r in res.items():
        reps = []
        for s, p in r["per"].items():
            if s not in BENCHMARK_SEEDS:
                continue
            keep = ~np.isin(p["patient"], list(both))
            reps.append(per_device_report(p["labels"][keep], p["pred"][keep], p["device"][keep])["all"])
        L.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                arm, len(reps), *(_ms([x[k] for x in reps]) for k in ("sp", "se", "score", "hs")),
                _ms([x["recall"][3] for x in reps]),
            )
        )
    print("dropped cycles (patients on both sides): {} of {}".format(*_dropped_counts(res, both)))

    devs = ["Meditron", "Litt3200", "AKGC417L"]
    L += ["\n### Per device (primary epoch, seeds 0-4), mean ± SD\n"]
    for metric, title in (("sp", "Sp"), ("se", "Se"), ("score", "Score"), ("hs", "HS")):
        L += ["**{}**\n".format(title), "| arm | " + " | ".join(devs) + " |", "|---|" + "---|" * len(devs)]
        for arm, r in res.items():
            cells = [
                _ms(
                    [
                        p["test"][d][metric]
                        for s, p in r["per"].items()
                        if s in BENCHMARK_SEEDS and d in p["test"]
                    ]
                    or [np.nan]
                )
                for d in devs
            ]
            L.append("| {} | {} |".format(arm, " | ".join(cells)))
        L.append("")

    if BASE in res:
        L += [
            "\n### Paired difference against the baseline (same seeds)\n",
            "| arm | seeds | ΔScore mean | 95% t-CI (paired seeds) | 95% patient bootstrap CI |",
            "|---|---|---|---|---|",
        ]
        base, rng = res[BASE]["per"], np.random.default_rng(0)
        check = None
        for arm, r in res.items():
            if arm == BASE:
                continue
            seeds = sorted(set(r["per"]) & set(base))
            m, lo, hi = _t_ci(
                [
                    r["per"][s]["test"]["all"]["score"] - base[s]["test"]["all"]["score"]
                    for s in seeds
                ]
            )
            y, pat = r["per"][seeds[0]]["labels"], r["per"][seeds[0]]["patient"]
            cv = np.stack([_score_counts(y, r["per"][s]["pred"], pat) for s in seeds])
            cb = np.stack([_score_counts(y, base[s]["pred"], pat) for s in seeds])
            w = rng.multinomial(
                cv.shape[1], np.ones(cv.shape[1]) / cv.shape[1], size=a.boot
            )  # (boot, patients)
            diff = (
                _score_from_counts(np.einsum("bp,spk->bsk", w, cv))
                - _score_from_counts(np.einsum("bp,spk->bsk", w, cb))
            ).mean(1)
            q = np.percentile(diff, [2.5, 97.5])
            L.append(
                "| {} | {} | {:+.2f} | [{:.2f}, {:.2f}] | [{:.2f}, {:.2f}] |".format(
                    arm, len(seeds), m, lo, hi, *q
                )
            )
            if arm == trust:
                check = (len(seeds), m, lo)
        if check:
            n, m, lo = check
            verdict = (
                "pass"
                if np.isfinite(lo) and lo > NON_INFERIORITY_MARGIN
                else "FAIL (use the baseline weights with test-time SC)"
            )
            L.append(
                "\nNon-inferiority ({} vs baseline, {} seeds): ΔScore {:+.2f}, lower bound {:.2f} vs margin {:.1f} -> **{}**.".format(
                    trust, n, m, lo, NON_INFERIORITY_MARGIN, verdict
                )
            )
    txt = "\n".join(L)
    print(txt)
    (root / "final.md").write_text(txt + "\n")


def gate(a):
    """Paired Score difference of two refit cells at their epochs (sensitivity analysis of gate G against another baseline)."""
    root = run_dir(a.run)
    units = load_units(root, r"seed(\d+)")
    arms = {"base": {"cell": a.base_cell, "epoch": a.base_epoch}, "arm": {"cell": a.arm_cell, "epoch": a.arm_epoch}}
    res = _collect(root, units, arms)
    b, r = res["base"]["per"], res["arm"]["per"]
    seeds = sorted(set(b) & set(r))
    d = [r[s]["test"]["all"]["score"] - b[s]["test"]["all"]["score"] for s in seeds]
    m, lo, hi = _t_ci(d)
    y, pat = b[seeds[0]]["labels"], b[seeds[0]]["patient"]
    cv = np.stack([_score_counts(y, r[s]["pred"], pat) for s in seeds])
    cb = np.stack([_score_counts(y, b[s]["pred"], pat) for s in seeds])
    w = np.random.default_rng(0).multinomial(cv.shape[1], np.ones(cv.shape[1]) / cv.shape[1], size=a.boot)
    diff = (_score_from_counts(np.einsum("bp,spk->bsk", w, cv)) - _score_from_counts(np.einsum("bp,spk->bsk", w, cb))).mean(1)
    q = np.percentile(diff, [2.5, 97.5])
    txt = (
        "| comparison | seeds | base Score | arm Score | ΔScore | 95% t-CI | 95% patient bootstrap CI | lower bound > {:.1f} |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| {} @{} vs {} @{} | {} | {} | {} | {:+.2f} | [{:.2f}, {:.2f}] | [{:.2f}, {:.2f}] | {} |".format(
            NON_INFERIORITY_MARGIN, a.arm_cell, a.arm_epoch, a.base_cell, a.base_epoch, len(seeds),
            _ms([b[s]["test"]["all"]["score"] for s in seeds]), _ms([r[s]["test"]["all"]["score"] for s in seeds]),
            m, lo, hi, *q, "yes" if lo > NON_INFERIORITY_MARGIN else "no",
        )
    )
    print(txt)
    (root / "gate_{}_vs_{}.md".format(a.arm_cell, a.base_cell)).write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "final", "gate"])
    ap.add_argument("--base_cell", default="baseline_e14")
    ap.add_argument("--base_epoch", type=int, default=14)
    ap.add_argument("--arm_cell", default="sc+sc_static")
    ap.add_argument("--arm_epoch", type=int, default=14)
    ap.add_argument("--run", default="latest", help="run_id under outputs/train/")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    {"screen": screen, "final": final, "gate": gate}[a.mode](a)


if __name__ == "__main__":
    main()
