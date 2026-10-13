"""ICBHI cycle dataset (official 60/40 split) and the training data loaders."""

import copy
import os
import random
from collections import defaultdict

import numpy as np
import torch
from src.paths import DATA
from src.processing.correction import (
    apply_spectrum_correction,
    mean_spectrum,
    reference_spectrum,
    spectrum_coefficients,
)
from src.processing.icbhi import DEVICES
from src.processing.splits import cv_split, lodo_split, patient_val_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .augment import SpecAugment, random_bin_gain_image
from .cycles import individual_cycles, read_annotations
from .encoders import get_encoder

CACHE = (
    DATA / "cache" / "train_images"
)  # one array per encoder and split, shared by all jobs and seeds


class ICBHIDataset(Dataset):
    """Cycles of the official train or test split. Items: (fbank image, label, device id).

    With `args.spectrum_correction` the waveform of every cycle goes through spectrum correction (SC) before the STFT: train
    clips use their own (source) device spectra, test clips their own (target) spectra, both against the train
    reference (`sc_reference` = `device_spectra` of the train dataset). The SC state is kept in `self.sc`.
    """

    def __init__(self, train_flag, transform, args, sc_reference=None, print_flag=True, audio_only=False):
        self.train_flag, self.transform, self.args = train_flag, transform, args
        lodo = bool(args.holdout_device)  # leave-one-device-out: one dataset over all cycles, split by index later
        self.split = "all" if lodo else "train" if train_flag else "test"
        wav_dir = os.path.join(args.data_folder, "audio_test_data")
        official = dict(
            line.strip().split("\t")
            for line in open(os.path.join(args.data_folder, "official_split.txt"))
            .read()
            .splitlines()
        )
        stems = sorted(
            {f.split(".")[0] for f in os.listdir(wav_dir) if f.endswith((".wav", ".txt"))}
        )
        if lodo:  # drop the patients that the official split puts on both sides
            sides = defaultdict(set)
            for f in stems:
                sides[f.split("_")[0]].add(official.get(f))
            self.dropped = {p for p, s in sides.items() if len(s) > 1 or None in s}
            self.filenames = [f for f in stems if f.split("_")[0] not in self.dropped]
        else:
            self.filenames = [f for f in stems if official.get(f) == self.split]
        annotations = read_annotations(wav_dir)

        self.encoder = get_encoder(args.encoder)
        cached = None if args.spectrum_correction or audio_only else self._load_cache()
        if cached:
            self.images, self.labels, self.devices, self.patients = cached
            self.sc = None
            self._lodo_indices()
        else:
            audio, labels, devices, patients = [], [], [], []
            for f in self.filenames:
                for wave, label in individual_cycles(annotations[f], wav_dir, f, args):
                    audio.append(wave)
                    labels.append(label)
                    devices.append(DEVICES.index(f.split("_")[-1]))
                    patients.append(f.split("_")[0])
            self.labels, self.devices, self.patients = labels, devices, np.array(patients)
            if audio_only:  # raw cycles for inference-time pipelines; no spectrograms, no SC
                self.audio, self.sc, self.images = audio, None, None
                self._lodo_indices()
                return
            self._lodo_indices()
            self.sc = (
                self._spectrum_correct(audio, sc_reference) if args.spectrum_correction else None
            )
            if self.sc:
                audio = self.sc.pop("audio")
            if args.spectrum_correction:
                self.images = [self.encoder.preprocess(a) for a in audio]
            else:
                self.images = self._build_cache(audio)
        assert self.images[0].shape == (*self.encoder.image_shape, 1), self.images[0].shape
        if print_flag:
            counts = np.bincount(self.labels, minlength=args.n_cls)
            print(
                "[{} dataset] {} cycles, {}".format(
                    self.split,
                    len(self.labels),
                    ", ".join(
                        "{} {} ({:.1f}%)".format(c, n, 100 * n / len(self.labels))
                        for c, n in zip(args.cls_list, counts)
                    ),
                )
            )

    def _lodo_indices(self):
        """In leave-one-device-out mode, the (train, calibration, test) cycle indices of this run's seed."""
        self.lodo = None
        if self.args.holdout_device:
            self.lodo = lodo_split(
                self.patients,
                np.asarray(self.devices),
                DEVICES.index(self.args.holdout_device),
                self.args.lodo_cal_frac,
                self.args.seed,
            )

    def _cache_paths(self):
        a = self.args
        stem = "{}_{}_{}cls_{}s_{}_{}hz_{}mel".format(
            a.encoder, self.split, a.n_cls, a.desired_length, a.pad_types, a.sample_rate, a.n_mels
        )
        return CACHE / (stem + ".npy"), CACHE / (stem + ".meta.npz")

    def _load_cache(self):
        """Spectrograms of this encoder and split memory-mapped from disk (shared by every job), or None."""
        images, meta = self._cache_paths()
        if not (images.exists() and meta.exists()):
            return None
        m = np.load(meta)
        return (
            np.load(images, mmap_mode="r"),
            m["labels"].tolist(),
            m["devices"].tolist(),
            m["patients"],
        )

    def _build_cache(self, audio):
        """Write the spectrogram of every cycle straight into a .npy file (no second copy in RAM), return it memory-mapped."""
        images, meta = self._cache_paths()
        images.parent.mkdir(parents=True, exist_ok=True)
        tmp, tmp_meta = (
            p.with_name("{}.{}.tmp{}".format(p.stem, os.getpid(), p.suffix))
            for p in (images, meta)
        )
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(len(audio), *self.encoder.image_shape, 1)
        )
        for i, a in enumerate(audio):
            out[i] = self.encoder.preprocess(a)
        out.flush()
        del out
        np.savez(tmp_meta, labels=self.labels, devices=self.devices, patients=self.patients)
        tmp_meta.replace(meta)
        tmp.replace(images)  # the image file last: its presence means both files are complete
        return np.load(images, mmap_mode="r")

    def _spectrum_correct(self, audio, sc_reference):
        """SC of every cycle. Returns dict `device_spectra`, `reference` (s_ref), `coefficients`, corrected `audio`."""
        used = None if self.lodo is None else set(np.concatenate(self.lodo).tolist())
        by_dev = defaultdict(list)
        for i, (a, d) in enumerate(zip(audio, self.devices)):
            if used is None or i in used:  # LODO: only cycles that are in train, calibration or test
                by_dev[d].append(a.numpy()[0])
        spectra = {d: mean_spectrum(w) for d, w in by_dev.items()}
        if self.lodo is not None:  # reference = mean of the source devices; the held-out device is corrected towards it
            held = DEVICES.index(self.args.holdout_device)
            ref_spectra = {d: s for d, s in spectra.items() if d != held}
        else:
            ref_spectra = spectra if self.train_flag else sc_reference
        a = self.args
        coef = spectrum_coefficients(
            spectra,
            reference=a.sc_reference,
            sc_mode=a.sc_mode,
            limit_freq_low=a.sc_limit_freq_low,
            limit_freq_high=a.sc_limit_freq_high,
            limit_freq_diff=a.sc_limit_freq_diff,
            reference_spectra=ref_spectra,
        )
        if not self.train_flag and self.lodo is None and getattr(a, "sc_test_profile", "target") == "source":
            # inductive: each test device uses the coefficient estimated on the training clips of the same device
            coef = spectrum_coefficients(
                sc_reference,
                reference=a.sc_reference,
                sc_mode=a.sc_mode,
                limit_freq_low=a.sc_limit_freq_low,
                limit_freq_high=a.sc_limit_freq_high,
                limit_freq_diff=a.sc_limit_freq_diff,
                reference_spectra=sc_reference,
            )
        only = getattr(a, "sc_only_device", None)
        apply = {d for d in coef if only is None or d == DEVICES.index(only)}
        corrected = [
            torch.from_numpy(apply_spectrum_correction(w.numpy()[0], coef[d]).astype(np.float32))[None]
            if d in apply
            else w  # devices outside --sc_only_device stay raw (not even an STFT round trip)
            for w, d in zip(audio, self.devices)
        ]
        return {
            "device_spectra": spectra,
            "reference": reference_spectrum(ref_spectra, a.sc_reference),
            "coefficients": coef,
            "audio": corrected,
        }

    def __getitem__(self, index):
        image = np.array(self.images[index])  # a copy: the cached array is read-only
        if self.train_flag and self.args.random_gain_db > 0:  # P3
            image = random_bin_gain_image(
                image,
                self.args.random_gain_db,
                self.encoder.db_scale,
                np.random.default_rng(random.getrandbits(32)),
            )
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], self.devices[index]

    def __len__(self):
        return len(self.labels)


class _Cycles:
    """Labels, devices, patients and SC state of a subset of cycles (what the report and export code reads)."""

    def __init__(self, ds, idx):
        self.labels = [ds.labels[i] for i in idx]
        self.devices = [ds.devices[i] for i in idx]
        self.patients = ds.patients[idx]
        self.sc = ds.sc


def _lodo_loaders(args):
    """Leave-one-device-out loaders: training patients, calibration patients (evaluated at the report epochs) and the
    held-out device as test set.

    Returns:
        Tuple (train_loader, None, test_loader, dataset, test cycles, calibration loader, calibration cycles).
    """
    train_tf = transforms.Compose([transforms.ToTensor(), SpecAugment()])
    eval_tf = transforms.ToTensor()
    ds = ICBHIDataset(True, train_tf, args)
    tr_idx, ca_idx, te_idx = ds.lodo
    eval_ds = copy.copy(ds)
    eval_ds.transform, eval_ds.train_flag = eval_tf, False
    print(
        "LODO {}: patients train/cal/test {}/{}/{}, cycles {}/{}/{}, dropped (both official sides) {}".format(
            args.holdout_device,
            *(len(set(ds.patients[i])) for i in (tr_idx, ca_idx, te_idx)),
            len(tr_idx),
            len(ca_idx),
            len(te_idx),
            sorted(ds.dropped),
        )
    )

    def make(d, **kw):
        return DataLoader(
            d, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, **kw
        )

    return (
        make(Subset(ds, tr_idx), shuffle=True, drop_last=True),
        None,
        make(Subset(eval_ds, te_idx), shuffle=False),
        ds,
        _Cycles(ds, te_idx),
        make(Subset(eval_ds, ca_idx), shuffle=False),
        _Cycles(ds, ca_idx),
    )


def build_loaders(args):
    """Train, validation (None for a fixed refit) and test loaders, plus calibration loader and cycles (leave-one-device-out only).

    Returns:
        Tuple (train_loader, val_loader, test_loader, train_dataset, test_dataset, calibration_loader, calibration_cycles);
        the last two are None unless `args.holdout_device` is set.
    """
    if args.holdout_device:
        return _lodo_loaders(args)
    train_tf = transforms.Compose([transforms.ToTensor(), SpecAugment()])
    eval_tf = transforms.ToTensor()
    train_dataset = ICBHIDataset(True, train_tf, args)
    test_dataset = ICBHIDataset(
        False, eval_tf, args, sc_reference=train_dataset.sc and train_dataset.sc["device_spectra"]
    )

    # Patient-level validation split of the training set. Shallow copy with the eval transform: shares the cached
    # spectrograms, no SpecAugment.
    if (
        args.cv_folds
    ):  # screening fold: same device-stratified patient folds for every variant and seed
        tr_idx, va_idx = cv_split(
            train_dataset.patients, np.asarray(train_dataset.devices), args.cv_folds, args.cv_fold
        )
    else:
        tr_idx, va_idx = patient_val_split(train_dataset.patients, args.val_frac, args.seed)
    val_dataset = copy.copy(train_dataset)
    val_dataset.transform, val_dataset.train_flag = eval_tf, False
    val_set = Subset(val_dataset, va_idx)
    train_set = Subset(train_dataset, tr_idx) if args.selection == "cv" else train_dataset
    print(
        "train/val patients: {}/{}  cycles {}/{}".format(
            len(set(train_dataset.patients[tr_idx])),
            len(set(train_dataset.patients[va_idx])),
            len(tr_idx),
            len(va_idx),
        )
    )

    def make(ds, **kw):
        return DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, **kw
        )

    train_loader = make(train_set, shuffle=True, drop_last=True)
    val_loader = (
        make(val_set, shuffle=False) if args.selection == "cv" else None
    )  # fixed refit: val patients are in the training set
    return (
        train_loader,
        val_loader,
        make(test_dataset, shuffle=False),
        train_dataset,
        test_dataset,
        None,
        None,
    )
