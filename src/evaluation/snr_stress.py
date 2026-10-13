"""Noise stress on the official test set: stationary ESC-50 noise mixed into the test cycles before SC, frozen refit models.

    python -m src.evaluation.snr_stress infer outputs/train/screen-ast/<cell>/seed<s> [...] --out outputs/snr/<name>
    python -m src.evaluation.snr_stress analyse --out outputs/snr --arms baseline sc

Noise passes through the stethoscope in reality, so it is added to the waveform before spectrum correction (SC). Noise
classes: rain, wind, sea_waves, engine, vacuum_cleaner, washing_machine, airplane, train (stationary; impulsive classes such
as crackling_fire, clock_tick, door_wood_knock and footsteps are excluded because they can look like crackles). Each cycle gets
one noise clip (seed = cycle index), tiled to 8 s at 16 kHz, scaled to the SNR by the power of the whole cycle.
SC models: the training reference s_ref and bound come from the unit's `sc_state.npz` / `train_args.json`; the test coefficient
of every device is re-estimated from the (noisy) unlabelled test clips, as in the clean pipeline. The coefficient change
against the clean estimate is stored.

Analysis (E0): official-test patients, without the 2 patients on both official sides, split 20 times into calibration and
evaluation halves stratified by device; calibration half clean, evaluation half noisy. LAC split conformal at alpha 0.1:
coverage, prior-matched Delta, set size; Score of the evaluation half. Mean over splits and seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from src.evaluation.lodo import coverage
from src.evaluation.lodo_ttsc import Pipeline, coef_db_distance, load_args
from src.paths import DATA
from src.processing.correction import clip_spectra_torch, spectrum_coefficients
from src.processing.icbhi import DEVICES
from src.processing.splits import calibration_eval_splits
from src.training.patchmix_cl.dataset import ICBHIDataset
from src.training.patchmix_cl.reporting import per_device_report
from src.training.patchmix_cl.summary import both_sides_patients

ESC50 = DATA / "raw" / "esc50" / "ESC-50-master"
NOISE_CLASSES = ("rain", "wind", "sea_waves", "engine", "vacuum_cleaner", "washing_machine", "airplane", "train")
SNRS = (30, 20, 10, 5)
SR, SAMPLES = 16000, 16000 * 8


def noise_bank():
    meta = pd.read_csv(ESC50 / "meta" / "esc50.csv")
    meta = meta[meta.category.isin(NOISE_CLASSES)].sort_values("filename").reset_index(drop=True)
    clips = []
    for f in meta.filename:
        x, sr = sf.read(ESC50 / "audio" / f, dtype="float32")
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        if np.mean(x ** 2) < 1e-8:  # a few ESC-50 clips are near silent
            continue
        x = np.tile(x, int(np.ceil(SAMPLES / len(x))))[:SAMPLES]
        clips.append(x.astype(np.float32))
    return np.stack(clips)


def add_noise(waves, bank, snr_db):
    """waves (n, samples) torch; one noise clip per cycle, chosen and offset by the cycle index."""
    out = waves.clone()
    for i in range(len(waves)):
        rng = np.random.default_rng(i)
        n = np.roll(bank[rng.integers(len(bank))], rng.integers(SAMPLES))
        ps, pn = float((waves[i] ** 2).mean()), float((n ** 2).mean())
        out[i] = waves[i] + torch.from_numpy(n) * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return out


class OfficialPipeline(Pipeline):
    """A refit unit (official split) on the raw official test cycles."""

    def __init__(self, unit_dir):
        self.unit = Path(unit_dir)
        self.args = load_args(unit_dir)
        rep = json.loads((self.unit / "report.json").read_text())
        self.epoch = int(sorted(rep["fixed"], key=int)[0])
        ds = ICBHIDataset(False, None, self.args, audio_only=True, print_flag=False)
        self.labels, self.devices, self.patients = np.asarray(ds.labels), np.asarray(ds.devices), np.asarray(ds.patients)
        self.waves = torch.stack([a[0] for a in ds.audio])
        from src.training.patchmix_cl.encoders import get_encoder
        from src.training.patchmix_cl.modeling import build_model
        from src import runs

        self.encoder = get_encoder(self.args.encoder)
        self.ckpt = runs.twin_dir(self.unit, "checkpoints") / "report_epoch_{}.pth".format(self.epoch)
        self.model, self.classifier, _, _, _ = build_model(self.args)
        ck = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        if ck["model"] is not None:
            self.model.load_state_dict(ck["model"])
        self.classifier.load_state_dict(ck["classifier"])
        self.model.eval(), self.classifier.eval()
        self.memo = {}
        self.rows = np.arange(len(self.waves))

    def sc_coefficients(self, waves):
        """Test coefficients against the saved training reference, from the unlabelled clips `waves`."""
        a = self.args
        state = np.load(self.unit / "sc_state.npz")
        spec = torch.cat([clip_spectra_torch(waves[i : i + 256].cuda().double()) for i in range(0, len(waves), 256)]).cpu().numpy()
        spectra = {d: spec[self.devices == d].mean(0) for d in np.unique(self.devices)}
        return spectrum_coefficients(spectra, reference=a.sc_reference, sc_mode=a.sc_mode, limit_freq_low=a.sc_limit_freq_low,
                                     limit_freq_high=a.sc_limit_freq_high, limit_freq_diff=a.sc_limit_freq_diff,
                                     reference_spectra={"ref": state["reference"]})

    def run(self, waves):
        coef, apply = None, ()
        if self.args.spectrum_correction:
            coef = self.sc_coefficients(waves)
            apply = set(coef)
        saved = self.waves
        self.waves = waves
        try:
            probs = self.infer(self.images(self.rows, coef, apply))[0]
        finally:
            self.waves = saved
        return probs, coef


def infer(a):
    bank = noise_bank()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for u in a.unit_dirs:
        f = out / (Path(u).name + ".npz")
        if f.exists():
            print("done:", f)
            continue
        p = OfficialPipeline(u)
        res = {"labels": p.labels, "device": p.devices, "patient": p.patients}
        clean, coef0 = p.run(p.waves)
        res["probs_inf"] = clean
        rep = per_device_report(p.labels, clean.argmax(1), p.devices)["all"]
        print(u, "clean Score {:.2f}".format(rep["score"]))
        for snr in SNRS:
            probs, coef = p.run(add_noise(p.waves, bank, snr))
            res["probs_{}".format(snr)] = probs
            if coef is not None:
                res["coef_rms_db_{}".format(snr)] = np.array([coef_db_distance(coef[d], coef0[d]) for d in sorted(coef)])
            print("  SNR {} dB: Score {:.2f}".format(snr, per_device_report(p.labels, probs.argmax(1), p.devices)["all"]["score"]))
        res["coef_devices"] = np.array(sorted(coef0)) if coef0 is not None else np.array([])
        np.savez_compressed(f, **res)
        del p
        torch.cuda.empty_cache()


def analyse(a):
    root = Path(a.out)
    both = both_sides_patients()
    L = ["| Arm | SNR (dB) | Seeds | Score (eval half) | Coverage | Delta (prior-matched) | Set size | SC coefficient change, RMS dB (50-2000 Hz) |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    for arm in a.arms:
        files = sorted((root / arm).glob("seed*.npz"))
        if not files:
            continue
        for snr in ("inf",) + SNRS:
            rows = []
            for f in files:
                z = np.load(f)
                keep = ~np.isin(z["patient"], list(both))
                df = pd.DataFrame({"patient": z["patient"][keep], "device": z["device"][keep]})
                clean, noisy = z["probs_inf"][keep], z["probs_{}".format(snr)][keep]
                y, pat = z["labels"][keep], z["patient"][keep]
                for sp in calibration_eval_splits(df, n_splits=20, seed=0):
                    ci, ei = np.isin(pat, sp["calibration"]), np.isin(pat, sp["evaluation"])
                    cov = coverage(clean[ci], y[ci], noisy[ei], y[ei])
                    sc = per_device_report(y[ei], noisy[ei].argmax(1), z["device"][keep][ei])["all"]["score"]
                    rows.append({"score": sc, **cov})
                coef = float(z["coef_rms_db_{}".format(snr)].mean()) if "coef_rms_db_{}".format(snr) in z.files else None
            m = lambda k: float(np.mean([r[k] for r in rows]))
            res["{}|{}".format(arm, snr)] = {k: m(k) for k in ("score", "coverage", "delta", "size")}
            res["{}|{}".format(arm, snr)]["coef_rms_db"] = coef
            L.append("| {} | {} | {} | {:.2f} | {:.3f} | {:+.3f} | {:.2f} | {} |".format(
                arm, snr, len(files), m("score"), m("coverage"), m("delta"), m("size"), "-" if coef is None else "{:.2f}".format(coef)))
    md = "\n".join(L) + "\n"
    (root / "snr_summary.md").write_text(md)
    (root / "snr_summary.json").write_text(json.dumps(res, indent=1))
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "analyse"])
    ap.add_argument("unit_dirs", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline", "sc"])
    a = ap.parse_args()
    infer(a) if a.mode == "infer" else analyse(a)


if __name__ == "__main__":
    main()
