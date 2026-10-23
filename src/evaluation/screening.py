"""Recording-level screening view: ICBHI 8 s windows, recording p(abnormal), 2-class split conformal and triage, and the
ICBHI-calibrated threshold applied to every KAUH filter (deployment check).

    python -m src.evaluation.screening infer outputs/train/screen-ast/<cell>/seed<s> --ckpt report_epoch_<E>.pth \
        --out outputs/screening/<condition>/seed<s>.npz [--sc]
    python -m src.evaluation.screening analyse --pairs p0:p0 p0:p0_sc auscult:auscult

ICBHI windows: every official-test recording (without the 2 patients on both official sides) is cut into
max(1, floor(duration / 8 s)) windows evenly spread, as for KAUH; waveform loading as in training (soundfile, torchaudio
resampling to 16 kHz, linear fade, repeat padding). Recording label: abnormal if any annotated cycle has a crackle or a
wheeze. With `--sc` (models trained with SC) every window is corrected with the saved test coefficient of its device
(`sc_state.npz`, estimated from the unlabelled official-test cycles, as in the registered test pipeline).

Recording p(abnormal) = mean over windows of 1 - p(normal) (`max` as a sensitivity view). E0: the ICBHI test patients are
split 20 times into calibration and evaluation halves stratified by device. LAC split conformal (alpha 0.1) on the
calibration recordings. Applied to (a) the ICBHI evaluation half and (b) every KAUH filter (`outputs/kauh/final/<cond>`).
Triage: {normal} no referral, {abnormal} refer, {both} uncertain, {} re-record. Reported: coverage, prior-matched Delta,
set size, referral / uncertain / re-record rates, screening Se (abnormal in the set) and Sp (set = {normal}), and argmax
Se / Sp at p(abnormal) > 0.5.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torchaudio import transforms as T

from src.evaluation.conformal import lac_scores, predict_sets, split_threshold
from src.evaluation.kauh import load as load_kauh
from src.evaluation.kauh import recordings as kauh_recordings
from src.evaluation.kauh_infer import predict
from src.evaluation.lodo_ttsc import load_args
from src.evaluation.shift import prior_matched_coverage
from src.processing.correction import apply_spectrum_correction
from src.processing.icbhi import DEVICES
from src.processing.splits import calibration_eval_splits
from src.training.patchmix_cl.cycles import FADE_RATIO, cut_pad, read_annotations
from src.training.patchmix_cl.summary import both_sides_patients

SR, WIN = 16000, 8.0
ALPHA = 0.1


def icbhi_windows(args):
    """Waveforms and table (stem, patient, device, label, window) of the official-test recordings' 8 s windows."""
    wav_dir = os.path.join(args.data_folder, "audio_test_data")
    official = dict(line.strip().split("\t") for line in open(os.path.join(args.data_folder, "official_split.txt")).read().splitlines())
    ann = read_annotations(wav_dir)
    both = both_sides_patients()
    waves, rows = [], []
    for stem in sorted(s for s, side in official.items() if side == "test" and s.split("_")[0] not in both):
        x, sr = sf.read(os.path.join(wav_dir, stem + ".wav"), dtype="float32", always_2d=True)
        x = torch.from_numpy(x.T.copy())
        if sr != SR:
            x = T.Resample(sr, SR)(x)
        fade = int(SR / FADE_RATIO)
        x = T.Fade(fade_in_len=fade, fade_out_len=fade, fade_shape="linear")(x)
        dur = x.shape[1] / SR
        a = ann[stem]
        label = int(((a["Crackles"] > 0) | (a["Wheezes"] > 0)).any())
        for w, s in enumerate(np.linspace(0, max(dur - WIN, 0), max(1, int(dur // WIN)))):
            i = int(round(s * SR))
            waves.append(cut_pad(x[:, i : i + int(WIN * SR)], int(WIN), SR, "repeat")[0].numpy())
            rows.append({"stem": stem, "patient": stem.split("_")[0], "device": DEVICES.index(stem.split("_")[-1]),
                         "label": label, "window": w})
    return waves, pd.DataFrame(rows)


def infer(a):
    from src.artifacts import prediction_inputs, record_prediction, atomic_npz
    from src.runs import twin_dir
    unit = Path(a.unit_dir)
    context = dict(unit=unit, checkpoint=twin_dir(unit) / a.ckpt, dataset="screening", sc=a.sc)
    inputs = prediction_inputs(**context)
    args = load_args(unit)
    waves, win = icbhi_windows(args)
    if a.sc:
        st = np.load(unit / "sc_state.npz")
        coef = {d: st["test_coefficients_{}".format(d)] for d in win.device.unique()}
        waves = [apply_spectrum_correction(w, coef[d]).astype(np.float32) for w, d in zip(waves, win.device)]
    from src import runs

    ckpt = runs.twin_dir(unit, "checkpoints") / a.ckpt
    probs = predict(waves, args, ckpt)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(out, probs=probs, **{k: win[k].to_numpy() for k in win.columns}, sc=np.array(a.sc),
                        ckpt_sha256=np.array(runs.sha256_file(ckpt)))
    record_prediction(out, inputs, **context)
    print(out, probs.shape)


def icbhi_recordings(d, agg="mean"):
    stems, inv = np.unique(d["stem"], return_inverse=True)
    f = np.mean if agg == "mean" else np.max
    p_abn = np.array([f(1 - d["probs"][inv == k][:, 0]) for k in range(len(stems))])
    first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(stems))])
    return {"p": np.stack([1 - p_abn, p_abn], 1), "y": d["label"][first], "patient": d["patient"][first],
            "device": d["device"][first]}


def kauh_recs(d, agg="mean"):
    if agg == "mean":
        return kauh_recordings(d)
    stems, inv = np.unique(d["stem"], return_inverse=True)
    p_abn = np.array([(1 - d["probs"][inv == k][:, 0]).max() for k in range(len(stems))])
    first = np.array([np.flatnonzero(inv == k)[0] for k in range(len(stems))])
    return {"p": np.stack([1 - p_abn, p_abn], 1), "y": (d["label"][first] > 0).astype(int), "patient": d["patient"][first],
            "filter": d["filter"][first]}


def triage(thr, p, y, prior):
    sets = predict_sets(p, thr)
    n = sets.sum(1)
    ab, no = y == 1, y == 0
    return {
        "coverage": float(sets[np.arange(len(y)), y].mean()),
        "delta": float((1 - ALPHA) - prior_matched_coverage(sets, y, prior)),
        "size": float(n.mean()),
        "refer": float((sets[:, 1] & ~sets[:, 0]).mean()),
        "no_referral": float((sets[:, 0] & ~sets[:, 1]).mean()),
        "uncertain": float((n == 2).mean()),
        "rerecord": float((n == 0).mean()),
        "screen_se": float(sets[ab, 1].mean()) if ab.any() else float("nan"),
        "screen_sp": float((sets[no, 0] & ~sets[no, 1]).mean()) if no.any() else float("nan"),
        "argmax_se": float((p[ab, 1] > 0.5).mean()) if ab.any() else float("nan"),
        "argmax_sp": float((p[no, 1] <= 0.5).mean()) if no.any() else float("nan"),
        "abnormal_share": float(ab.mean()),
    }


def analyse(a):
    root, kroot = Path(a.dir), Path(a.kauh_dir)
    keys = ("coverage", "delta", "size", "refer", "uncertain", "rerecord", "screen_se", "screen_sp", "argmax_se", "argmax_sp", "abnormal_share")
    L = ["| ICBHI calibration | Applied to | Aggregation | Seeds | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
    res = {}
    for pair in a.pairs:
        cal_cond, kauh_cond = pair.split(":")
        for agg in ("mean", "max"):
            out = {}
            seeds = sorted(int(p.stem[4:]) for p in (root / cal_cond).glob("seed*.npz"))
            for s in seeds:
                ic = icbhi_recordings(dict(np.load(root / cal_cond / "seed{}.npz".format(s), allow_pickle=True)), agg)
                kf = kroot / kauh_cond / "seed{}.npz".format(s)
                kr = kauh_recs(load_kauh(kf), agg) if kf.exists() else None
                df = pd.DataFrame({"patient": ic["patient"], "device": ic["device"]})
                for sp in calibration_eval_splits(df, n_splits=20, seed=0):
                    ci, ei = np.isin(ic["patient"], sp["calibration"]), np.isin(ic["patient"], sp["evaluation"])
                    thr = split_threshold(lac_scores(ic["p"][ci], ic["y"][ci]), ALPHA)
                    prior = np.bincount(ic["y"][ci], minlength=2) / ci.sum()
                    out.setdefault("ICBHI evaluation half", []).append(triage(thr, ic["p"][ei], ic["y"][ei], prior))
                    if kr is not None:
                        for f in sorted(set(kr["filter"])):
                            m = kr["filter"] == f
                            out.setdefault("KAUH {} ({})".format(f, kauh_cond), []).append(triage(thr, kr["p"][m], kr["y"][m], prior))
            for target, rows in out.items():
                mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
                res["{}|{}|{}".format(cal_cond, target, agg)] = mean
                L.append("| {} | {} | {} | {} | ".format(cal_cond, target, agg, len(seeds)) + " | ".join("{:.3f}".format(mean[k]) for k in keys) + " |")
    md = "\n".join(L) + "\n"
    (root / "screening_summary.md").write_text(md)
    (root / "screening_summary.json").write_text(json.dumps(res, indent=1))
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "analyse"])
    ap.add_argument("unit_dir", nargs="?")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--sc", action="store_true")
    ap.add_argument("--dir", default="outputs/screening")
    ap.add_argument("--kauh_dir", default="outputs/kauh/final")
    ap.add_argument("--pairs", nargs="+", default=["p0:p0", "p0:p0_sc", "auscult:auscult"],
                    help="<ICBHI condition>:<KAUH condition>; the ICBHI calibration of the weights, applied to that KAUH condition")
    a = ap.parse_args()
    infer(a) if a.mode == "infer" else analyse(a)


if __name__ == "__main__":
    main()
