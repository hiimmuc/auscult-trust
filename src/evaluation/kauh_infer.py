"""Frozen-model inference on KAUH 8 s windows, with optional test-time spectrum correction (SC).

    python -m src.evaluation.kauh_infer outputs/train/<run>/<cell>/seed<s> --ckpt report_epoch_<E>.pth \
        --out outputs/kauh/<run>/<name>/seed<s>.npz [--sc --sc_state outputs/train/<run>/<sc cell>/seed<s>/sc_state.npz]

No KAUH data is used for training or selection. With `--sc`, the reference spectrum `s_ref` comes from the saved SC state of an
SC training run (identical across seeds: it is the mean of the ICBHI training devices' spectra), the coefficient of each KAUH
filter from the mean spectrum of that filter's own unlabelled windows, bounded with the `sc_mode` of that run. Windows keep
the recording-level label (an [Assumption] noted in `src.processing.kauh.windows`); KAUH audio is 4 kHz, resampled to 16 kHz.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src import runs
from src.paths import DATA
from src.processing.audio import SR, crop_pad, load_wav
from src.processing.correction import (
    apply_spectrum_correction,
    mean_spectrum,
    spectrum_coefficients,
)
from src.processing.kauh import file_table, windows
from src.training.patchmix_cl.config import build_parser
from src.training.patchmix_cl.encoders import get_encoder
from src.training.patchmix_cl.modeling import build_model

KAUH_ROOT = DATA / "raw" / "kauh"
WINDOW_SEC = 8.0


def window_waves(win):
    """16 kHz waveform (8 s, zero-padded) of every window row of `kauh.windows`."""
    cache, out = {}, []
    for r in win.itertuples(index=False):
        if r.wav not in cache:
            cache = {r.wav: load_wav(r.wav, SR)}  # rows are grouped by recording: keep one at a time
        x = cache[r.wav]
        i = int(round(r.start * SR))
        out.append(crop_pad(x[i : i + int(WINDOW_SEC * SR)], int(WINDOW_SEC * SR)))
    return out


def test_time_sc(waves, filters, sc_state, args):
    """Spectrum-correct every window towards the saved reference, one coefficient curve per KAUH filter."""
    ref = sc_state["reference"]
    spectra = {
        f: mean_spectrum([w for w, g in zip(waves, filters) if g == f]) for f in sorted(set(filters))
    }
    coef = spectrum_coefficients(
        spectra,
        reference=args.sc_reference,
        sc_mode=args.sc_mode,
        limit_freq_low=args.sc_limit_freq_low,
        limit_freq_high=args.sc_limit_freq_high,
        limit_freq_diff=args.sc_limit_freq_diff,
        reference_spectra={"ref": ref},
    )
    return [apply_spectrum_correction(w, coef[g]).astype(np.float32) for w, g in zip(waves, filters)]


def load_frozen(args, ckpt_path):
    model, classifier, _, _, _ = build_model(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck["model"] is not None:
        model.load_state_dict(ck["model"])
    classifier.load_state_dict(ck["classifier"])
    return model.eval(), classifier.eval()


def images_of(waves, args):
    enc = get_encoder(args.encoder)
    return np.stack([enc.preprocess(torch.from_numpy(w)[None]) for w in waves])[..., 0]  # (n, time, freq)


@torch.no_grad()
def predict_images(images, model, classifier, curves=None):
    """Softmax on (n, time, freq) images; `curves` (n, freq) are added to every frame (TTA-EQ view)."""
    probs = []
    for i in range(0, len(images), 32):
        x = torch.from_numpy(images[i : i + 32]).float()[:, None]
        if curves is not None:
            x = x + torch.from_numpy(curves[i : i + 32])[:, None, None, :]
        o = classifier(model(x.cuda()))  # fp32, as the earlier KAUH conditions
        probs.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(probs)


def predict(waves, args, ckpt_path):
    """Softmax of the frozen model on every waveform."""
    model, classifier = load_frozen(args, ckpt_path)
    return predict_images(images_of(waves, args), model, classifier)


def source_stats(unit, args):
    """Per-freq-bin mean and SD of the training images of the unit's recipe (A2 source statistics), cached on disk."""
    cell = Path(unit).parent.name
    f = Path("outputs/kauh/a2_stats") / (cell + ".npz")
    if f.exists():
        z = np.load(f)
        return z["mean"], z["sd"]
    from src.training.patchmix_cl.dataset import ICBHIDataset

    tr = vars(argparse.Namespace(**vars(args))).copy()
    a = argparse.Namespace(**{**tr, "holdout_device": None, "cv_folds": 0})
    ds = ICBHIDataset(True, None, a, print_flag=False)
    acc, acc2, n = 0.0, 0.0, 0
    for i in range(0, len(ds.images), 256):
        x = np.asarray(ds.images[i : i + 256], dtype=np.float64)[..., 0]
        acc, acc2, n = acc + x.sum((0, 1)), acc2 + (x ** 2).sum((0, 1)), n + x.shape[0] * x.shape[1]
    mean = acc / n
    sd = np.sqrt(acc2 / n - mean ** 2) + 1e-6
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, mean=mean, sd=sd)
    return mean, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit_dir", help="outputs/<exp>/<run_id>/<cell>/<unit> of the trained model (holds train_args.json)")
    ap.add_argument("--ckpt", required=True, help="weight file in the unit's checkpoints dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sc", action="store_true", help="test-time spectrum correction")
    ap.add_argument("--sc_state", help="sc_state.npz of an SC training run (reference spectrum); default: the unit's own")
    ap.add_argument("--sc_args", help="train_args.json of the SC run (coefficient bound); default: the unit's own")
    ap.add_argument("--a2", action="store_true", help="A2: per-filter image moments mapped onto the training images'")
    ap.add_argument("--tta_eq", type=int, default=0, help="TTA-EQ views (6 dB SD per-bin gain, seeds 1000 + k)")
    a = ap.parse_args()
    unit = Path(a.unit_dir)
    args = build_parser().parse_args([])
    vars(args).update(json.load(open(unit / "train_args.json")))
    sc_args = argparse.Namespace(**json.load(open(a.sc_args or unit / "train_args.json")))
    win = windows(file_table(KAUH_ROOT), WINDOW_SEC).reset_index(drop=True)
    waves = window_waves(win)
    if a.sc:
        state = dict(np.load(a.sc_state or unit / "sc_state.npz"))
        waves = test_time_sc(waves, list(win["filter"]), state, sc_args)
    ckpt = runs.twin_dir(a.unit_dir, "checkpoints") / a.ckpt
    model, classifier = load_frozen(args, ckpt)
    images = images_of(waves, args)
    if a.a2:
        from src.processing.correction import isa_match

        src = source_stats(unit, args)
        filters = win["filter"].to_numpy()
        for f in np.unique(filters):
            m = filters == f
            flat = images[m].reshape(-1, images.shape[-1]).astype(np.float64)
            images[m] = isa_match(images[m], (flat.mean(0), flat.std(0) + 1e-6), src).astype(np.float32)
    view_probs = None
    if a.tta_eq:
        from src.evaluation.lodo_ttsc import TTA_DB, TTA_SEED0, gain_curves

        sd = TTA_DB * get_encoder(args.encoder).db_scale
        view_probs = np.stack([predict_images(images, model, classifier, gain_curves(len(images), images.shape[-1], sd, TTA_SEED0 + k))
                               for k in range(a.tta_eq)])
        probs = view_probs.mean(0)
    else:
        probs = predict_images(images, model, classifier)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        probs=probs,
        label=win["label"].to_numpy(),
        patient=win["patient"].to_numpy(),
        stem=win["stem"].to_numpy(),
        filter=win["filter"].to_numpy(),
        window=win["window"].to_numpy(),
        sc=np.array(a.sc),
        a2=np.array(a.a2),
        tta_eq=np.array(a.tta_eq),
        **({"view_probs": view_probs.astype(np.float16)} if view_probs is not None else {}),
        ckpt_sha256=np.array(runs.sha256_file(ckpt)),
    )
    print(out, probs.shape, "sc" if a.sc else "no sc", "a2" if a.a2 else "", "tta{}".format(a.tta_eq) if a.tta_eq else "")


if __name__ == "__main__":
    main()
