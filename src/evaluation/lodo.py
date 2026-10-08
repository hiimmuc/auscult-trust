"""Conformal coverage under leave-one-device-out (LODO): baseline vs SC arms.

Reads the units written by `--holdout_device` runs (`outputs/train/<run>/<arm cell>+lodo_<device>/seed<s>/`):
`cal_preds.npz` (softmax of the calibration patients, source devices) and `test_preds.npz` (the held-out device).

    python -m src.evaluation.lodo --run lodo-ast --base baseline --arms sc+sc_static freq_mixstyle

Per fold and arm: LAC split conformal at `alpha`, coverage on the held-out device, prior-matched coverage (class mix of
the calibration data, evaluation only), Delta = (1 - alpha) - prior-matched coverage, set size, singleton and empty rate.
Statistic H2: D = mean over tested folds of |Delta_base| - |Delta_arm| (positive: the arm covers closer to nominal).
A fold is tested only if |Delta_base| exceeds the 95th percentile of |Delta| over exchangeable re-splits of the
calibration patients (a source-device-only null, same procedure, no device shift). Confidence interval: two-level bootstrap
(seeds, then calibration and test patients), recalibrating in every draw, paired across arms.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.conformal import lac_scores, predict_sets, split_threshold
from src.evaluation.shift import prior_matched_coverage, set_stats
from src.processing.icbhi import DEVICES

ALPHA = 0.1


def load_unit(unit_dir):
    """Calibration and test softmax of one LODO unit at its report epoch."""
    unit_dir = Path(unit_dir)
    cal, test = np.load(unit_dir / "cal_preds.npz"), np.load(unit_dir / "test_preds.npz")
    epoch = int(cal["epochs"][0])
    i = list(test["epochs"]).index(epoch)
    report = json.loads((unit_dir / "report.json").read_text())
    return {
        "epoch": epoch,
        "cal_p": cal["probs"][0].astype(np.float32),
        "cal_y": cal["labels"],
        "cal_pat": cal["patient"],
        "test_p": test["probs"][i].astype(np.float32),
        "test_y": test["labels"],
        "test_pat": test["patient"],
        "score": report["fixed"][str(epoch)]["test"]["all"],
    }


def coverage(cal_p, cal_y, test_p, test_y, alpha=ALPHA):
    """Split-conformal result of one calibration/test pair.

    Returns:
        Dict `coverage`, `delta` (nominal minus prior-matched coverage; positive = under-coverage), and set statistics.
    """
    thr = split_threshold(lac_scores(cal_p, cal_y), alpha)
    sets = predict_sets(test_p, thr)
    prior = np.bincount(cal_y, minlength=cal_p.shape[1]) / len(cal_y)
    hit = sets[np.arange(len(test_y)), test_y]
    return {
        "coverage": float(hit.mean()),
        "delta": float((1 - alpha) - prior_matched_coverage(sets, test_y, prior)),
        **set_stats(sets),
    }


def _rows_by_patient(patients):
    ids, inv = np.unique(patients, return_inverse=True)
    return [np.flatnonzero(inv == k) for k in range(len(ids))]


def _resample_rows(groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[k] for k in pick])


def null_delta_p95(u, n_splits=200, alpha=ALPHA, seed=0):
    """95th percentile of |Delta| when the calibration patients are split at random (no device shift)."""
    groups = _rows_by_patient(u["cal_pat"])
    n_test = max(1, min(len(set(u["test_pat"])), len(groups) // 2))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        order = rng.permutation(len(groups))
        te = np.concatenate([groups[k] for k in order[:n_test]])
        ca = np.concatenate([groups[k] for k in order[n_test:]])
        out.append(
            abs(coverage(u["cal_p"][ca], u["cal_y"][ca], u["cal_p"][te], u["cal_y"][te], alpha)["delta"])
        )
    return float(np.percentile(out, 95))


def collect(run_dir, arm_cells, devices, seeds):
    """Load every unit: units[device][arm][seed] -> dict (missing units are skipped)."""
    units = {}
    for dev in devices:
        for arm in arm_cells:
            for s in seeds:
                d = Path(run_dir) / f"{arm}+lodo_{dev.lower()}" / f"seed{s}"
                if (d / "report.json").exists() and (d / "cal_preds.npz").exists():
                    units.setdefault(dev, {}).setdefault(arm, {})[s] = load_unit(d)
    return units


def fold_table(units, arms, alpha=ALPHA):
    """Per fold and arm: mean over seeds of Score, coverage, |Delta| and set statistics."""
    rows = []
    for dev, by_arm in units.items():
        for arm in arms:
            us = by_arm.get(arm, {})
            if not us:
                continue
            cs = [coverage(u["cal_p"], u["cal_y"], u["test_p"], u["test_y"], alpha) for u in us.values()]
            rows.append(
                {
                    "device": dev,
                    "arm": arm,
                    "seeds": len(us),
                    "n_test": int(len(next(iter(us.values()))["test_y"])),
                    **{k: float(np.mean([u["score"][k] for u in us.values()])) for k in ("sp", "se", "score", "hs")},
                    "coverage": float(np.mean([c["coverage"] for c in cs])),
                    "delta": float(np.mean([c["delta"] for c in cs])),
                    "abs_delta": float(np.mean([abs(c["delta"]) for c in cs])),
                    "size": float(np.mean([c["size"] for c in cs])),
                    "singleton": float(np.mean([c["singleton"] for c in cs])),
                    "empty": float(np.mean([c["empty"] for c in cs])),
                }
            )
    return rows


def tested_folds(units, base, alpha=ALPHA, n_splits=200):
    """Folds whose baseline |Delta| exceeds the null 95th percentile; returns {device: (abs_delta, null_p95)}."""
    out = {}
    for dev, by_arm in units.items():
        us = by_arm.get(base, {})
        if not us:
            continue
        d = np.mean([abs(coverage(u["cal_p"], u["cal_y"], u["test_p"], u["test_y"], alpha)["delta"]) for u in us.values()])
        p95 = np.mean([null_delta_p95(u, n_splits, alpha, seed=s) for s, u in us.items()])
        out[dev] = (float(d), float(p95))
    return out


def h2_statistic(units, base, arm, folds, alpha=ALPHA, n_boot=1000, seed=0):
    """D = mean over `folds` of |Delta_base| - |Delta_arm|, point estimate and two-level bootstrap interval."""
    def one(draw_rng):
        diffs = []
        for dev in folds:
            ub, ua = units[dev][base], units[dev][arm]
            common = sorted(set(ub) & set(ua))
            if not common:
                continue
            seeds = common if draw_rng is None else draw_rng.choice(common, len(common))
            vals = []
            for s in seeds:
                b, a = ub[s], ua[s]  # same seed = same calibration and test patients in both arms
                cg, tg = _rows_by_patient(b["cal_pat"]), _rows_by_patient(b["test_pat"])
                if draw_rng is None:
                    ci, ti = np.arange(len(b["cal_y"])), np.arange(len(b["test_y"]))
                else:
                    ci, ti = _resample_rows(cg, draw_rng), _resample_rows(tg, draw_rng)
                db = abs(coverage(b["cal_p"][ci], b["cal_y"][ci], b["test_p"][ti], b["test_y"][ti], alpha)["delta"])
                da = abs(coverage(a["cal_p"][ci], a["cal_y"][ci], a["test_p"][ti], a["test_y"][ti], alpha)["delta"])
                vals.append(db - da)
            diffs.append(np.mean(vals))
        return float(np.mean(diffs)) if diffs else float("nan")

    rng = np.random.default_rng(seed)
    point = one(None)
    boots = np.array([one(rng) for _ in range(n_boot)])
    return {"D": point, "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))], "folds": list(folds)}


def markdown(rows, gate, stats, base):
    L = ["| Held-out device | Arm | Seeds | Test cycles | Sp | Se | Score | HS | Coverage | Delta (prior-matched) | Set size | Singleton | Empty |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            "| {device} | {arm} | {seeds} | {n_test} | {sp:.1f} | {se:.1f} | {score:.1f} | {hs:.1f} | {coverage:.3f} | {delta:+.3f} | {size:.2f} | {singleton:.2f} | {empty:.3f} |".format(**r)
        )
    L += ["", "Null gate (baseline |Delta| vs 95th percentile of |Delta| under exchangeable re-splits of calibration patients):", ""]
    L += ["| Device | baseline abs Delta | null p95 | tested |", "|---|---|---|---|"]
    for dev, (d, p) in gate.items():
        L.append(f"| {dev} | {d:.3f} | {p:.3f} | {'yes' if d > p else 'no'} |")
    L += ["", f"H2 statistic D = mean over tested folds of |Delta_{base}| - |Delta_arm| (95% two-level bootstrap interval):", ""]
    for arm, s in stats.items():
        L.append(f"- {arm}: D = {s['D']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] over folds {', '.join(s['folds']) or 'none'}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id under outputs/train/")
    ap.add_argument("--base", default="baseline")
    ap.add_argument("--arms", nargs="+", default=["sc+sc_static", "freq_mixstyle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    root = Path("outputs/train") / a.run
    units = collect(root, [a.base] + a.arms, DEVICES, a.seeds)
    rows = fold_table(units, [a.base] + a.arms)
    gate = tested_folds(units, a.base)
    folds = [d for d, (x, p) in gate.items() if x > p]
    stats = {}
    for arm in a.arms:
        both = [d for d in folds if arm in units.get(d, {})]
        stats[arm] = h2_statistic(units, a.base, arm, both, n_boot=a.boot)
        no_akg = [d for d in both if d != "AKGC417L"]  # class-mix dominated fold, reported separately
        stats[arm + " (without AKGC417L)"] = h2_statistic(units, a.base, arm, no_akg, n_boot=a.boot)
    md = markdown(rows, gate, stats, a.base)
    (root / "lodo_summary.md").write_text(md)
    (root / "lodo_summary.json").write_text(json.dumps({"rows": rows, "gate": gate, "h2": stats}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
