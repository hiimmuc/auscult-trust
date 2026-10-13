"""Spectrum-correction (SC) diagnostics on ICBHI: coefficient curves, sampling noise of a device spectrum, class-mix drift.

    PYTHONPATH=. .venv-train/bin/python scripts/sc_diagnostics.py [--boot 200]

(a) Coefficient in dB against frequency per device, for the official refit and every LODO fold of the SC arm (from the saved
    `sc_state.npz`), with the static band edges (50 and 2000 Hz) marked -> `outputs/sc_diag/coefficients_*.png`.
(b) Sampling noise: per device, N patients drawn without replacement (N in 1, 2, 3, 5, 10, all), device spectrum = mean of
    their cycles' clip spectra, coefficient against the fixed reference (mean of the device spectra over all patients);
    SD in dB per bin over the draws, summarised as the median over 50-2000 Hz. sigma_N10 = median over devices at N = 10.
(c) Class-mix drift: the device spectrum recomputed with the cycles reweighted to another class mix (all normal, all
    abnormal, and the class mix of every other device), RMS dB change of the coefficient over 50-2000 Hz.
Between-device distance: RMS dB difference of the coefficient curves of two devices over 50-2000 Hz, every pair.
Uses all ICBHI cycles (both official sides, labels only for (c)); never a model.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.processing.correction import clip_spectra_torch, reference_spectrum  # noqa: E402
from src.processing.icbhi import DEVICES  # noqa: E402
from src.training.patchmix_cl.config import build_parser  # noqa: E402
from src.training.patchmix_cl.dataset import ICBHIDataset  # noqa: E402

OUT = Path("outputs/sc_diag")
FREQS = np.fft.rfftfreq(1024, 1 / 16000)
BAND = (FREQS >= 50) & (FREQS <= 2000)
NS = (1, 2, 3, 5, 10, "all")


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def rms_band(a_db, b_db):
    return float(np.sqrt(np.mean((a_db[BAND] - b_db[BAND]) ** 2)))


def load_cycles():
    args = build_parser().parse_args([])
    vars(args).update(desired_length=8, pad_types="repeat", n_cls=4, holdout_device="Meditron", lodo_cal_frac=0.2)
    ds = ICBHIDataset(True, None, args, audio_only=True, print_flag=False)
    waves = torch.stack([a[0] for a in ds.audio]).double()
    spec = torch.cat([clip_spectra_torch(waves[i : i + 256]) for i in range(0, len(waves), 256)]).numpy()
    return spec, np.asarray(ds.labels), np.asarray(ds.devices), np.asarray(ds.patients)


def plot_states(paths, title, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in paths:
        st = np.load(p)
        tag = p.parent.parent.name.split("+lodo_")[-1] if "+lodo_" in p.parent.parent.name else "refit"
        for k in sorted(x for x in st.files if x.startswith("train_coefficients_") or x.startswith("test_coefficients_")):
            d = int(k.rsplit("_", 1)[1])
            ax.semilogx(FREQS[1:], db(st[k])[1:], lw=1, label="{} {} {}".format(tag, k.split("_")[0], DEVICES[d]))
    for f in (50, 2000):
        ax.axvline(f, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Hz", ylabel="coefficient (dB)", title=title)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--run", default="screen-ast")
    ap.add_argument("--lodo_run", default="lodo-ast")
    ap.add_argument("--sc_cell", default="sc+sc_static")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # (a) saved coefficient curves
    refit = sorted(Path("outputs/train", a.run, a.sc_cell).glob("seed0/sc_state.npz"))
    if refit:
        plot_states(refit, "SC coefficients, official refit ({})".format(a.sc_cell), OUT / "coefficients_refit.png")
    meditron = {}
    for dev in DEVICES:
        ps = sorted(Path("outputs/train", a.lodo_run, "{}+lodo_{}".format(a.sc_cell, dev.lower())).glob("seed0/sc_state.npz"))
        if ps:
            plot_states(ps, "SC coefficients, LODO fold {} ({})".format(dev, a.sc_cell), OUT / "coefficients_lodo_{}.png".format(dev.lower()))
            st = np.load(ps[0])
            meditron[dev] = {DEVICES[int(k.rsplit("_", 1)[1])]: {
                "band_mean_db": float(db(st[k])[BAND].mean()), "band_min_db": float(db(st[k])[BAND].min()),
                "band_max_db": float(db(st[k])[BAND].max())} for k in st.files if k.startswith("train_coefficients_")}

    # (b), (c) from the raw cycles
    spec, labels, devices, patients = load_cycles()
    present = sorted(set(devices.tolist()))
    dev_spec = {d: spec[devices == d].mean(0) for d in present}
    ref = reference_spectrum(dev_spec, "arithmetic")
    coef_db = {d: db(ref / dev_spec[d]) for d in present}
    between = {"{}-{}".format(DEVICES[x], DEVICES[y]): rms_band(coef_db[x], coef_db[y]) for x, y in itertools.combinations(present, 2)}

    rng = np.random.default_rng(0)
    sampling = {}
    for d in present:
        rows = np.flatnonzero(devices == d)
        pats = np.unique(patients[rows])
        sampling[DEVICES[d]] = {"patients": int(len(pats))}
        for n in NS:
            k = len(pats) if n == "all" else n
            if k > len(pats) or (n != "all" and k == len(pats)):
                continue  # N equal to every patient is the full estimate, no sampling noise
            curves = []
            for _ in range(a.boot if k < len(pats) else 1):
                pick = rng.choice(pats, k, replace=False)
                s = spec[rows[np.isin(patients[rows], pick)]].mean(0)
                curves.append(db(ref / s))
            curves = np.array(curves)
            sd = curves.std(0) if len(curves) > 1 else np.zeros(curves.shape[1])
            sampling[DEVICES[d]][str(n)] = {
                "median_sd_db_band": float(np.median(sd[BAND])),
                "rms_to_full_db": float(np.mean([rms_band(c, coef_db[d]) for c in curves])),
            }
    sigma_n10 = float(np.median([v["10"]["median_sd_db_band"] for v in sampling.values() if "10" in v]))

    priors = {DEVICES[d]: np.bincount(labels[devices == d], minlength=4) / (devices == d).sum() for d in present}
    mixes = {"all_normal": np.array([1.0, 0, 0, 0]), "all_abnormal": np.array([0, 1, 1, 1]) / 3.0}
    mixes.update({"prior_of_" + k: v for k, v in priors.items()})
    classmix = {}
    for d in present:
        rows = np.flatnonzero(devices == d)
        own = priors[DEVICES[d]]
        classmix[DEVICES[d]] = {}
        for name, mix in mixes.items():
            w = np.where(own[labels[rows]] > 0, mix[labels[rows]] / np.maximum(own[labels[rows]], 1e-12), 0.0)
            if w.sum() == 0:
                continue
            s = (spec[rows] * w[:, None]).sum(0) / w.sum()
            classmix[DEVICES[d]][name] = rms_band(db(ref / s), coef_db[d])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in present:
        ax.semilogx(FREQS[1:], coef_db[d][1:], label=DEVICES[d])
    for f in (50, 2000):
        ax.axvline(f, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Hz", ylabel="coefficient (dB)", title="SC coefficient per device, all ICBHI cycles, arithmetic reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coefficients_all_cycles.png", dpi=130)
    plt.close(fig)

    summary = {"sigma_N10_db": sigma_n10, "sampling": sampling, "class_mix_drift_rms_db": classmix,
               "between_device_rms_db": between, "lodo_fold_coefficients": meditron, "band_hz": [50, 2000], "boot": a.boot}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"sigma_N10_db": sigma_n10, "between_device_rms_db": between}, indent=1))
    for dev, v in sampling.items():
        print(dev, v["patients"], {n: round(v[str(n)]["median_sd_db_band"], 2) for n in NS if str(n) in v})
    for dev, v in classmix.items():
        print(dev, {k: round(x, 2) for k, x in v.items()})


if __name__ == "__main__":
    main()
