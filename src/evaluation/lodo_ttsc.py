"""Inference-only arms on a trained leave-one-device-out (LODO) unit: test-time SC, A2, TTA-EQ, and the target-sample sweep.

    python -m src.evaluation.lodo_ttsc outputs/train/lodo-ast/baseline+lodo_litt3200/seed0 --sc target --cell baseline_ttsc
    python -m src.evaluation.lodo_ttsc outputs/train/lodo-ast/sc+sc_static+lodo_litt3200/seed0 --sc all --tta_eq 8 --cell sc_tta8
    python -m src.evaluation.lodo_ttsc <AuscultTrust unit> --sc all --target_n 1,2,3,5,10,all --target_rep 50

The frozen weights of the unit (`report_epoch_<E>.pth`) are applied to the same calibration and test cycles as in training
(same `lodo_split` seed). Everything runs batched on the GPU; `src.processing.correction` has the torch SC (checked against
the librosa version in the tests).

SC (`--sc`): `none`; `target` = only the held-out device is corrected towards the mean spectrum of the source devices (the
calibration cycles stay raw; primary variant for weights trained without SC); `all` = every device is corrected, as during
SC training. Coefficient bound: `--sc_mode` (default: the unit's own when it was trained with SC, else static).
A2 (`--a2`): per-mel-bin mean and SD of the unlabelled held-out images mapped onto the training images' statistics.
TTA-EQ (`--tta_eq K`): mean softmax over K views, view k adds a smooth random per-bin gain (6 dB SD, `random_bin_gain`
distribution, seed 1000 + k, one curve per cycle) to every image; calibration and test use the same views.

Outputs, readable by `src.evaluation.lodo` as an arm: `outputs/train/<run>/<cell>+lodo_<dev>/seed<s>/` with `cal_preds.npz`,
`test_preds.npz`, `report.json` (`fixed[E].test`), `emb.npz` (pooled features, for the device probe) and, with TTA-EQ,
`view_probs.npz`. The target-sample sweep (`--target_n`) writes `outputs/train/<run>/target_n/<unit cell>/seed<s>.json`
instead: per N and repetition, the held-out coefficient is estimated from N held-out patients and the model is evaluated on
the other held-out patients (estimation patients never enter the evaluation), next to the full-estimate model on the same
patients.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src import runs
from src.evaluation.lodo import coverage
from src.processing.correction import (
    apply_spectrum_correction_torch,
    clip_spectra_torch,
    isa_match,
    spectrum_coefficients,
)
from src.processing.icbhi import DEVICES
from src.training.patchmix_cl.config import build_parser
from src.training.patchmix_cl.dataset import ICBHIDataset
from src.training.patchmix_cl.encoders import get_encoder
from src.training.patchmix_cl.modeling import build_model
from src.training.patchmix_cl.reporting import per_device_report

TTA_SEED0, TTA_DB, TTA_KNOTS = 1000, 6.0, 6
BATCH = 64


def load_args(unit_dir):
    args = build_parser().parse_args([])
    vars(args).update(json.load(open(Path(unit_dir) / "train_args.json")))
    return args


def report_epoch(unit_dir):
    rep = json.loads((Path(unit_dir) / "report.json").read_text())
    return int(sorted(rep["fixed"], key=int)[0])


class Pipeline:
    """Raw LODO cycles of one unit, the frozen model and the batched image/inference steps."""

    def __init__(self, unit_dir, num_workers=0):
        self.unit = Path(unit_dir)
        self.args = load_args(unit_dir)
        self.epoch = report_epoch(unit_dir)
        ds = ICBHIDataset(True, None, self.args, audio_only=True, print_flag=False)
        self.labels = np.asarray(ds.labels)
        self.devices = np.asarray(ds.devices)
        self.patients = np.asarray(ds.patients)
        self.tr, self.ca, self.te = ds.lodo
        self.held = DEVICES.index(self.args.holdout_device)
        self.waves = torch.stack([a[0] for a in ds.audio])  # (n, samples), CPU
        self.encoder = get_encoder(self.args.encoder)
        used = np.concatenate([self.tr, self.ca, self.te])
        self.used = np.sort(used)
        self.clip_spec = np.zeros((len(self.waves), 513))
        for i in range(0, len(self.used), 256):
            j = self.used[i : i + 256]
            self.clip_spec[j] = clip_spectra_torch(self.waves[j].cuda().double()).cpu().numpy()
        self.ckpt = runs.twin_dir(self.unit, "checkpoints") / "report_epoch_{}.pth".format(self.epoch)
        self.model, self.classifier, _, _, _ = build_model(self.args)
        ck = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        if ck["model"] is not None:
            self.model.load_state_dict(ck["model"])
        self.classifier.load_state_dict(ck["classifier"])
        self.model.eval(), self.classifier.eval()

    def spectra(self, held_rows=None):
        """Device mean spectra over the used cycles; `held_rows` restricts the held-out device's estimate to those cycles."""
        out = {}
        for d in sorted(set(self.devices[self.used].tolist())):
            rows = self.used[self.devices[self.used] == d]
            if d == self.held and held_rows is not None:
                rows = held_rows
            out[d] = self.clip_spec[rows].mean(0)
        return out

    def coefficients(self, sc_mode, held_rows=None):
        a, spectra = self.args, self.spectra(held_rows)
        ref = {d: s for d, s in spectra.items() if d != self.held}
        return spectrum_coefficients(
            spectra,
            reference=a.sc_reference,
            sc_mode=sc_mode,
            limit_freq_low=a.sc_limit_freq_low,
            limit_freq_high=a.sc_limit_freq_high,
            limit_freq_diff=a.sc_limit_freq_diff,
            reference_spectra=ref,
        )

    def images(self, rows, coef=None, apply_devices=()):
        """Encoder images (n, 1, time, freq) of `rows`; cycles of `apply_devices` are spectrum-corrected with `coef`."""
        out = []
        for i in range(0, len(rows), BATCH):
            j = rows[i : i + BATCH]
            w = self.waves[j].cuda()
            fix = np.isin(self.devices[j], list(apply_devices))
            if fix.any():
                c = torch.from_numpy(np.stack([coef[d] for d in self.devices[j][fix]])).cuda().float()
                w[torch.from_numpy(fix).cuda()] = apply_spectrum_correction_torch(w[torch.from_numpy(fix).cuda()], c)
            out += [torch.from_numpy(self.encoder.preprocess(x[None])) for x in w]
        return torch.stack(out).permute(0, 3, 1, 2)  # (n, 1, time, freq) on CPU

    @torch.no_grad()
    def infer(self, images, curves=None):
        """Softmax and pooled features; `curves` (n, freq) are added to every frame of the matching image (TTA-EQ view)."""
        probs, feats = [], []
        for i in range(0, len(images), BATCH):
            x = images[i : i + BATCH].cuda()
            if curves is not None:
                x = x + torch.from_numpy(curves[i : i + BATCH]).cuda()[:, None, None, :].to(x.dtype)
            with torch.cuda.amp.autocast():
                f = self.model(x)
                o = self.classifier(f)
            probs.append(torch.softmax(o.float(), 1).cpu().numpy())
            feats.append(f.float().cpu().numpy())
        return np.concatenate(probs), np.concatenate(feats)


def gain_curves(n_total, n_freq, sd, seed, knots=TTA_KNOTS):
    """One smooth random per-bin gain curve per cycle index (same draw as `random_bin_gain`), shape (n_total, n_freq)."""
    rng = np.random.default_rng(seed)
    grid, kn = np.linspace(0, 1, n_freq), np.linspace(0, 1, knots)
    return np.stack([np.interp(grid, kn, rng.normal(0, sd, knots)) for _ in range(n_total)]).astype(np.float32)


def a2_images(images_test, images_source):
    """A2 on (n, 1, time, freq) images: per-freq-bin moments of the test images mapped onto the source images'."""
    def stats(x):
        flat = x[:, 0].reshape(-1, x.shape[-1]).double()
        return flat.mean(0).numpy(), flat.std(0).numpy() + 1e-6

    t, s = stats(images_test), stats(images_source)
    out = isa_match(images_test[:, 0].numpy(), t, s)
    return torch.from_numpy(out.astype(np.float32))[:, None]


def resolve_sc_mode(p, sc_mode):
    return sc_mode or (p.args.sc_mode if p.args.spectrum_correction else "static")


def run_arm(p, a):
    """One inference arm: calibration and test softmax in the format of a LODO training unit."""
    sc_mode = resolve_sc_mode(p, a.sc_mode)
    coef = None if a.sc == "none" else p.coefficients(sc_mode)
    apply = () if a.sc == "none" else ({p.held} if a.sc == "target" else set(coef))
    cal_img = p.images(p.ca, coef, apply)
    test_img = p.images(p.te, coef, apply)
    if a.a2:
        test_img = a2_images(test_img, p.images(p.tr, coef, apply))
    if a.tta_eq:
        n_freq = test_img.shape[-1]
        sd = TTA_DB * p.encoder.db_scale
        views = {"cal": [], "test": []}
        for k in range(a.tta_eq):
            curves = gain_curves(len(p.waves), n_freq, sd, TTA_SEED0 + k)
            views["cal"].append(p.infer(cal_img, curves[p.ca])[0])
            views["test"].append(p.infer(test_img, curves[p.te])[0])
        cal_p, test_p = (np.mean(views[k], 0) for k in ("cal", "test"))
        cal_f = test_f = None
    else:
        (cal_p, cal_f), (test_p, test_f) = p.infer(cal_img), p.infer(test_img)
    return {"cal_p": cal_p, "test_p": test_p, "cal_f": cal_f, "test_f": test_f, "coef": coef, "sc_mode": sc_mode,
            "views": views if a.tta_eq else None}


def write_arm(p, a, res):
    dev = p.args.holdout_device.lower()
    out = Path("outputs/train") / a.run / "{}+lodo_{}".format(a.cell, dev) / p.unit.name
    out.mkdir(parents=True, exist_ok=True)
    E = p.epoch
    test_pred = res["test_p"].argmax(1)
    rep = per_device_report(p.labels[p.te], test_pred, p.devices[p.te], p.args.n_cls)
    np.savez_compressed(out / "test_preds.npz", epochs=np.array([E]), preds=test_pred[None].astype(np.int8),
                        probs=res["test_p"][None].astype(np.float16), labels=p.labels[p.te], device=p.devices[p.te],
                        patient=p.patients[p.te])
    np.savez_compressed(out / "cal_preds.npz", epochs=np.array([E]), probs=res["cal_p"][None].astype(np.float16),
                        labels=p.labels[p.ca], device=p.devices[p.ca], patient=p.patients[p.ca])
    if res["cal_f"] is not None:
        np.savez_compressed(out / "emb.npz", cal=res["cal_f"].astype(np.float16), test=res["test_f"].astype(np.float16))
    if res["views"] is not None:
        np.savez_compressed(out / "view_probs.npz", cal=np.stack(res["views"]["cal"]).astype(np.float16),
                            test=np.stack(res["views"]["test"]).astype(np.float16))
    if res["coef"] is not None:
        np.savez_compressed(out / "sc_coefficients.npz", **{"coefficients_{}".format(d): c for d, c in res["coef"].items()})
    meta = {"source_unit": str(p.unit), "ckpt": p.ckpt.name, "ckpt_sha256": runs.sha256_file(p.ckpt), "sc": a.sc,
            "sc_mode": res["sc_mode"] if a.sc != "none" else None, "a2": a.a2, "tta_eq": a.tta_eq,
            "code_commit": runs.code_commit() if hasattr(runs, "code_commit") else None}
    (out / "infer_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "report.json").write_text(json.dumps({"cell": a.cell, "unit": p.unit.name, "selection": "fixed",
                                                 "fixed": {str(E): {"epoch": E, "test": rep}}, "infer": meta}, indent=1))
    print(out, "Score {:.2f} (Sp {:.2f}, Se {:.2f})".format(rep["all"]["score"], rep["all"]["sp"], rep["all"]["se"]))


def coef_db_distance(c1, c2, sr=16000, n_fft=1024, lo=50.0, hi=2000.0):
    """RMS difference in dB between two coefficient curves over the band [lo, hi] Hz."""
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (f >= lo) & (f <= hi)
    return float(np.sqrt(np.mean((20 * np.log10(c1[band] / c2[band])) ** 2)))


def target_n_sweep(p, a):
    """Held-out coefficient from N held-out patients; evaluation on the other held-out patients, R repetitions."""
    sc_mode = resolve_sc_mode(p, a.sc_mode)
    apply = set(DEVICES.index(d) for d in DEVICES) if a.sc == "all" else {p.held}
    full = p.coefficients(sc_mode)
    cal_p = p.infer(p.images(p.ca, full, apply))[0]
    full_test = p.infer(p.images(p.te, full, apply))[0]
    pats = np.unique(p.patients[p.te])
    rows = []
    for tok in a.target_n.split(","):
        n = len(pats) - 5 if tok == "all" else int(tok)
        if n < 1 or n > len(pats) - 5:
            print("skip N={} ({} held-out patients)".format(tok, len(pats)))
            continue
        for r in range(a.target_rep):
            rng = np.random.default_rng(10_000 * n + r)
            est = rng.choice(pats, n, replace=False)
            in_est = np.isin(p.patients[p.te], est)
            coef = p.coefficients(sc_mode, held_rows=p.te[in_est])
            ev = p.te[~in_est]
            test_p = p.infer(p.images(ev, coef, apply))[0]
            ref_p = full_test[~in_est]
            y = p.labels[ev]
            row = {"N": int(n), "N_token": tok, "rep": r, "eval_patients": int(len(pats) - n), "eval_cycles": int(len(ev)),
                   "coef_db_rms_to_full": coef_db_distance(coef[p.held], full[p.held])}
            for name, pr in (("n", test_p), ("full", ref_p)):
                rep = per_device_report(y, pr.argmax(1), p.devices[ev], p.args.n_cls)["all"]
                cov = coverage(cal_p, p.labels[p.ca], pr, y)
                row[name] = {"score": rep["score"], "sp": rep["sp"], "se": rep["se"], **cov}
            rows.append(row)
            if tok == "all":
                break  # "all" leaves 5 evaluation patients; one draw per seed is enough to show the ceiling
        print("N={}: |Delta| n {:.3f} vs full {:.3f}, coef RMS {:.2f} dB".format(
            tok, np.mean([abs(x["n"]["delta"]) for x in rows if x["N_token"] == tok]),
            np.mean([abs(x["full"]["delta"]) for x in rows if x["N_token"] == tok]),
            np.mean([x["coef_db_rms_to_full"] for x in rows if x["N_token"] == tok])))
    out = Path("outputs/train") / a.run / "target_n" / p.unit.parent.name / (p.unit.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"unit": str(p.unit), "sc": a.sc, "sc_mode": sc_mode, "rows": rows}, indent=1))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit_dirs", nargs="+", help="LODO units outputs/train/<run>/<cell>+lodo_<dev>/seed<s>")
    ap.add_argument("--cell", help="output cell name (arm name in lodo.py)")
    ap.add_argument("--run", default="lodo-ast")
    ap.add_argument("--sc", choices=["none", "target", "all"], default="none")
    ap.add_argument("--sc_mode", choices=["dynamic", "static"], default=None)
    ap.add_argument("--a2", action="store_true")
    ap.add_argument("--tta_eq", type=int, default=0, help="number of TTA-EQ views (0: off)")
    ap.add_argument("--target_n", default=None, help="comma list, e.g. 1,2,3,5,10,all: target-sample sweep instead of an arm")
    ap.add_argument("--target_rep", type=int, default=50)
    a = ap.parse_args()
    assert a.target_n or a.cell, "--cell is needed for an arm"
    for u in a.unit_dirs:
        if a.cell and not a.target_n:
            dev = load_args(u).holdout_device.lower()
            if (Path("outputs/train") / a.run / "{}+lodo_{}".format(a.cell, dev) / Path(u).name / "report.json").exists():
                print("done:", a.cell, dev, Path(u).name)
                continue
        p = Pipeline(u)
        if a.target_n:
            target_n_sweep(p, a)
        else:
            write_arm(p, a, run_arm(p, a))
        del p
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
