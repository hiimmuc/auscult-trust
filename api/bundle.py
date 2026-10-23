"""Build the serving bundle: frozen pipelines, conformal thresholds and the demo samples.

    PYTHONPATH=. .venv-train/bin/python -m api.bundle [--out app/bundle]

The bundle is the only thing the API reads, so the model, the API and the web app can change independently.

Pipelines: `auscult` = AuscultTrust weights + test-time SC, `baseline` = the same recipe without SC. Each comes from one
trained unit (`outputs/train/<run>/<cell>/seed<s>`, its `train_args.json`, its report epoch checkpoint and, with SC, its
`sc_state.npz`).

Conformal (LAC, alpha 0.1), calibrated on split 0 of `calibration_eval_splits` over the official-test patients (without the
patients on both official sides):
- `cycle4`: 4-class cycle threshold, from the unit's saved test softmax at the report epoch.
- `cycle2`: normal vs abnormal on the same cycles, p(abnormal) = 1 - p(normal).
- `screen`: recording-level screening threshold, from the unit's screening export (8 s windows, mean of 1 - p(normal)).

Demo samples: 3 recordings per 4-class label, only from evaluation-half patients (never trained on, never calibrated on).
Ranked by the share of cycles with that label; among recordings where it is the majority label, spread over devices, then
patients; then filled by rank. No ICBHI patient was recorded on three
devices, and the multi-device patients are on the training side or on both sides, so the samples are not same-patient sets.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src import runs
from src.evaluation.conformal import lac_scores, split_threshold
from src.evaluation.lodo_ttsc import load_args, report_epoch
from src.evaluation.screening import ALPHA, icbhi_recordings
from src.processing.icbhi import CLASSES, DEVICES, cycle_table
from src.processing.splits import calibration_eval_splits
from src.training.patchmix_cl.summary import both_sides_patients

# (unit, screening export from `src.evaluation.screening infer`, label); BEATs + PAFA with and without static SC.
# The AST units (screen-ast/sc+sc_static, screen-ast/baseline) fit too once their weights are on disk.
PIPELINES = {
    "auscult": ("outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static/seed0", "outputs/screening/final-pafa-static/auscult/seed0.npz",
                "AuscultTrust (PAFA + static SC)"),
    "baseline": ("outputs/train/grid-v1/beats+ft+pafa/seed0", "outputs/screening/final-pafa-static/baseline/seed0.npz", "Baseline (BEATs + PAFA, no SC)"),
}
PER_CLASS = 3


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def test_cycles(data_folder):
    """Official-test cycles without both-sides patients, in the order of the saved test softmax."""
    split = dict(line.split("\t") for line in open(Path(data_folder) / "official_split.txt").read().splitlines())
    t = cycle_table(Path(data_folder) / "audio_test_data")
    t["patient"] = t.stem.str.split("_").str[0]
    return t[t.stem.map(split) == "test"].reset_index(drop=True), both_sides_patients(Path(data_folder) / "official_split.txt")


def pipeline_entry(name, unit, screen_npz, label, split, out):
    unit = Path(unit)
    args = load_args(unit)
    epoch = report_epoch(unit)
    ckpt = runs.twin_dir(unit, "checkpoints") / "report_epoch_{}.pth".format(epoch)
    sha = runs.sha256_file(ckpt)

    z = np.load(unit / "test_preds.npz")
    probs = z["probs"][list(z["epochs"]).index(epoch)].astype(np.float64)
    cal = np.isin(z["patient"], split["calibration"])
    y4 = z["labels"]
    p2 = np.stack([probs[:, 0], 1 - probs[:, 0]], 1)
    thr4 = float(split_threshold(lac_scores(probs[cal], y4[cal]), ALPHA))
    thr2 = float(split_threshold(lac_scores(p2[cal], (y4[cal] > 0).astype(int)), ALPHA))

    sc = np.load(screen_npz, allow_pickle=True)
    assert str(sc["ckpt_sha256"]) == sha, "screening export {} is not from {}".format(screen_npz, ckpt)
    rec = icbhi_recordings(dict(sc), "mean")
    rcal = np.isin(rec["patient"], split["calibration"])
    thr_screen = float(split_threshold(lac_scores(rec["p"][rcal], rec["y"][rcal]), ALPHA))

    entry = {
        "label": label,
        "unit": str(unit),
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha,
        "epoch": epoch,
        "encoder": args.encoder,
        "method": args.method,
        "spectrum_correction": bool(args.spectrum_correction),
        "thresholds": {"cycle4": thr4, "cycle2": thr2, "screen": thr_screen},
        "calibration": {"cycles": int(cal.sum()), "recordings": int(rcal.sum()), "patients": len(split["calibration"])},
    }
    if args.spectrum_correction:
        st = dict(np.load(unit / "sc_state.npz"))
        np.savez(out / "sc_{}.npz".format(name), **st)
        entry["sc"] = {"mode": str(st["sc_mode"]), "reference": args.sc_reference, "limit_freq_low": args.sc_limit_freq_low,
                       "limit_freq_high": args.sc_limit_freq_high, "limit_freq_diff": args.sc_limit_freq_diff,
                       "s_ref_sha256": sha256_bytes(st["reference"].tobytes()), "state": "sc_{}.npz".format(name)}
    return entry, args


def pick_samples(cycles, split, both):
    ev = cycles[cycles.patient.isin(split["evaluation"]) & ~cycles.patient.isin(both)]
    rec = ev.groupby("stem").agg(device=("device", "first"), patient=("patient", "first"), n=("label", "size"))
    out = []
    for c, name in enumerate(CLASSES):
        rec["share"] = ev.groupby("stem").label.apply(lambda x: (x == c).mean())
        rec["count"] = ev.groupby("stem").label.apply(lambda x: (x == c).sum())
        ranked = rec[rec["count"] > 0].sort_values(["share", "count"], ascending=False)
        chosen = []
        majority = ranked[ranked.share >= 0.5]
        # spread over devices, then patients, among recordings where the label is the majority; then fill by rank
        for pool, key in ((majority, "device"), (majority, "patient"), (ranked, None)):
            for stem, r in pool.iterrows():
                if len(chosen) == PER_CLASS:
                    break
                if stem in [s for s, _ in chosen]:
                    continue
                if key and r[key] in {x[key] for _, x in chosen}:
                    continue
                chosen.append((stem, r))
        out += [(name, stem, r) for stem, r in chosen]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/bundle")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    data_folder = load_args(PIPELINES["auscult"][0]).data_folder
    cycles, both = test_cycles(data_folder)
    pc = cycles[~cycles.patient.isin(both)][["patient", "device"]].drop_duplicates()
    split = calibration_eval_splits(pc.assign(device=pc.device.map(DEVICES.index)), n_splits=20, seed=0)[0]

    pipelines, train_args = {}, None
    for name, (unit, screen, label) in PIPELINES.items():
        pipelines[name], args = pipeline_entry(name, unit, screen, label, split, out)
        train_args = train_args or args

    samples = []
    for cls, stem, r in pick_samples(cycles, split, both):
        src = Path(data_folder) / "audio_test_data" / (stem + ".wav")
        shutil.copyfile(src, out / "samples" / (stem + ".wav"))
        cyc = cycles[cycles.stem == stem]
        samples.append({
            "id": stem, "class": cls, "file": "samples/{}.wav".format(stem), "patient": "icbhi_" + r.patient,
            "device": r.device, "chest_location": stem.split("_")[2], "acquisition": stem.split("_")[3],
            "class_share": round(float(r.share), 3), "class_cycles": int(r["count"]), "cycles_total": int(r.n),
            "cycles": [{"start": float(s), "end": float(e), "label": CLASSES[int(l)]} for s, e, l in zip(cyc.start, cyc.end, cyc.label)],
            "sha256": runs.sha256_file(src),
        })

    manifest = {
        "classes": CLASSES,
        "classes2": ["normal", "abnormal"],
        "devices": DEVICES,
        "alpha": ALPHA,
        "preprocessing": {"sample_rate": train_args.sample_rate, "segment_sec": train_args.desired_length,
                          "pad": train_args.pad_types, "fade_sec": 1 / 16, "features": "kaldi fbank, 128 mel, 10 ms hop",
                          "screen_window_sec": 8.0, "screen_aggregation": "mean of 1 - p(normal)"},
        "calibration_split": {"name": "calibration_eval_splits[0], official-test patients", "calibration": split["calibration"],
                              "evaluation": split["evaluation"]},
        "pipelines": pipelines,
        "samples": samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(out / "manifest.json", len(samples), "samples")
    print(pd.DataFrame([(s["class"], s["id"], s["device"], s["class_share"]) for s in samples]).to_string())
    for n, p in pipelines.items():
        print(n, p["thresholds"], p["calibration"])


if __name__ == "__main__":
    main()
