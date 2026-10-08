"""KAUH positive controls with frozen models: A (paired invariance across the three software filters) and B2 (conformal
coverage when calibrating on one filter and testing on another).

    python -m src.evaluation.kauh --dir outputs/kauh/<run> --base p0 --arms p0_sc auscult

`<dir>/<condition>/seed<s>.npz` come from `src.evaluation.kauh_infer` (window softmax). Conditions: the baseline weights without
test-time SC (`p0`), with it (`p0_sc`, the isolation contrast) and the AuscultTrust weights with SC (`auscult`).

A: the same 8 s window rendered with the Bell, Diaphragm and Extended filter must give the same prediction; flip rate (window
argmax differs, 4-class) and total variation distance, averaged over the three filter pairs.
B2: recording-level screening view (normal vs abnormal, p(abnormal) = mean over windows of 1 - p(normal)), LAC split conformal.
Patients are split into 5 rotation groups; fold i tests on the target-filter recordings of group i and calibrates on the
source-filter recordings of the other four groups; six ordered (source, target) filter pairs. Delta = nominal minus prior-matched
coverage; a pair is tested only if the baseline |Delta| exceeds the 95th percentile of |Delta| under exchangeable re-splits of
the calibration patients; D = mean over tested pairs of |Delta_base| - |Delta_arm| with a patient/seed bootstrap interval.
All of this is a positive control: SC should help because the three filters are near-linear renderings of one signal.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.evaluation.lodo import ALPHA, _resample_rows, _rows_by_patient, coverage, null_delta_p95
from src.processing.kauh import FILTERS, ORDERED_PAIRS, file_table, rotation_partitions

KAUH_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "kauh"


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def recordings(d):
    """Recording-level screening view: one row per recording and filter."""
    stems, inv = np.unique(d["stem"], return_inverse=True)
    p_abn = np.array([(1 - d["probs"][inv == k][:, 0]).mean() for k in range(len(stems))])
    first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(stems))])
    return {
        "p": np.stack([1 - p_abn, p_abn], 1),
        "y": (d["label"][first] > 0).astype(int),
        "patient": d["patient"][first],
        "filter": d["filter"][first],
    }


def invariance(d, patients=None):
    """Mean flip rate and TV distance over the three filter pairs; `patients` restricts or resamples rows (with repeats)."""
    key = {}
    for i, (p, w, f) in enumerate(zip(d["patient"], d["window"], d["filter"])):
        key.setdefault((p, int(w)), {})[f] = i
    keys = [k for k, v in key.items() if len(v) == len(FILTERS)]
    if patients is not None:
        by_p = {}
        for k in keys:
            by_p.setdefault(k[0], []).append(k)
        keys = [k for p in patients for k in by_p.get(p, [])]
    flips, tvs = [], []
    for f1, f2 in itertools.combinations(FILTERS.values(), 2):
        a = np.array([key[k][f1] for k in keys])
        b = np.array([key[k][f2] for k in keys])
        flips.append((d["probs"][a].argmax(1) != d["probs"][b].argmax(1)).mean())
        tvs.append(0.5 * np.abs(d["probs"][a] - d["probs"][b]).sum(1).mean())
    return float(np.mean(flips)), float(np.mean(tvs))


def rotation_folds(seed=0):
    """Patient groups of the rotation: list of (test patients, calibration patients = all other patients)."""
    df = file_table(KAUH_ROOT)
    parts = rotation_partitions(df[df.label >= 0], n_folds=5, seed=seed)
    allp = sorted({p for part in parts for k in ("train", "cal", "test") for p in part[k]})
    return [(set(p["test"]), set(allp) - set(p["test"])) for p in parts]


def pair_fold(rec, src, tgt, test_pat, cal_pat, cal_rows=None, test_rows=None):
    """Calibration and test arrays of one ordered pair in one fold (rows may be bootstrap-resampled)."""
    is_src, is_tgt = rec["filter"] == src, rec["filter"] == tgt
    c = np.flatnonzero(is_src & np.isin(rec["patient"], list(cal_pat)))
    t = np.flatnonzero(is_tgt & np.isin(rec["patient"], list(test_pat)))
    if cal_rows is not None:
        c, t = c[cal_rows(len(c))], t[test_rows(len(t))]
    return {
        "cal_p": rec["p"][c], "cal_y": rec["y"][c], "cal_pat": rec["patient"][c],
        "test_p": rec["p"][t], "test_y": rec["y"][t], "test_pat": rec["patient"][t],
    }


def b2_delta(recs, folds, src, tgt, rng=None):
    """Mean over folds of |Delta| for one ordered pair (one model/seed); `rng` resamples calibration and test patients."""
    out = []
    for test_pat, cal_pat in folds:
        u = pair_fold(recs, src, tgt, test_pat, cal_pat)
        if rng is not None:
            ci = _resample_rows(_rows_by_patient(u["cal_pat"]), rng)
            ti = _resample_rows(_rows_by_patient(u["test_pat"]), rng)
            u = {"cal_p": u["cal_p"][ci], "cal_y": u["cal_y"][ci], "cal_pat": u["cal_pat"][ci],
                 "test_p": u["test_p"][ti], "test_y": u["test_y"][ti], "test_pat": u["test_pat"][ti]}
        out.append(abs(coverage(u["cal_p"], u["cal_y"], u["test_p"], u["test_y"])["delta"]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default="p0")
    ap.add_argument("--arms", nargs="+", default=["p0_sc", "auscult"])
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    root = Path(a.dir)
    conds = [a.base] + a.arms
    data = {c: {int(p.stem[4:]): load(p) for p in sorted((root / c).glob("seed*.npz"))} for c in conds}
    recs = {c: {s: recordings(d) for s, d in data[c].items()} for c in conds}
    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    folds = rotation_folds()
    L = ["## A: paired invariance across the three filters (windows)", "",
         "| Condition | Flip rate | TV distance |", "|---|---|---|"]
    A = {}
    for c in conds:
        v = np.array([invariance(data[c][s]) for s in seeds])
        A[c] = v
        L.append(f"| {c} | {v[:, 0].mean():.3f} ± {v[:, 0].std(ddof=1) if len(v) > 1 else 0:.3f} | {v[:, 1].mean():.3f} ± {v[:, 1].std(ddof=1) if len(v) > 1 else 0:.3f} |")
    rng = np.random.default_rng(0)
    pats = sorted(set(data[a.base][seeds[0]]["patient"]))
    L += ["", "Flip-rate reduction against the baseline (positive: arm is more invariant), 95% bootstrap interval over patients and seeds:", ""]
    for arm in a.arms:
        diffs = []
        for _ in range(a.boot):
            ps = list(rng.choice(pats, len(pats)))
            ss = rng.choice(seeds, len(seeds))
            diffs.append(np.mean([invariance(data[a.base][s], ps)[0] - invariance(data[arm][s], ps)[0] for s in ss]))
        point = A[a.base][:, 0].mean() - A[arm][:, 0].mean()
        L.append(f"- {a.base} - {arm}: {point:+.3f} [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]")
    # B2
    L += ["", "## B2: conformal coverage, calibrate on one filter, test on another (recording level, normal vs abnormal)", "",
          "| Pair (source -> target) | Condition | mean abs Delta | null p95 | tested |", "|---|---|---|---|---|"]
    null, abs_d = {}, {}
    for src, tgt in ORDERED_PAIRS:
        for c in conds:
            abs_d[src, tgt, c] = np.mean([b2_delta(recs[c][s], folds, src, tgt) for s in seeds])
        nulls = []
        for s in seeds:
            for test_pat, cal_pat in folds:
                u = pair_fold(recs[a.base][s], src, src, test_pat, cal_pat)  # source-filter data only
                nulls.append(null_delta_p95({**u, "test_pat": u["test_pat"]}, n_splits=100, seed=s))
        null[src, tgt] = float(np.mean(nulls))
        for c in conds:
            L.append(f"| {src} -> {tgt} | {c} | {abs_d[src, tgt, c]:.3f} | {null[src, tgt]:.3f} | {'yes' if abs_d[src, tgt, a.base] > null[src, tgt] else 'no'} |")
    tested = [(s_, t_) for s_, t_ in ORDERED_PAIRS if abs_d[s_, t_, a.base] > null[s_, t_]]
    L += ["", f"Tested pairs (baseline |Delta| above the null p95): {len(tested)} of {len(ORDERED_PAIRS)}.", ""]
    results = {"A": {c: A[c].mean(0).tolist() for c in conds}, "tested_pairs": [list(p) for p in tested]}
    for arm in a.arms:
        def stat(draw):
            vals = []
            for src, tgt in tested:
                ss = seeds if draw is None else draw.choice(seeds, len(seeds))
                vals.append(np.mean([b2_delta(recs[a.base][s], folds, src, tgt, draw) - b2_delta(recs[arm][s], folds, src, tgt, draw) for s in ss]))
            return float(np.mean(vals)) if vals else float("nan")

        point = stat(None)
        boots = np.array([stat(np.random.default_rng(1000 + i)) for i in range(max(50, a.boot // 5))])
        L.append(f"- D({a.base} vs {arm}) = {point:+.4f} [{np.nanpercentile(boots, 2.5):+.4f}, {np.nanpercentile(boots, 97.5):+.4f}] over {len(tested)} tested pairs")
        results[f"D_{arm}"] = [point, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    md = "\n".join(L) + "\n"
    (root / "kauh_summary.md").write_text(md)
    (root / "kauh_summary.json").write_text(json.dumps(results, indent=1))
    print(md)


if __name__ == "__main__":
    main()
