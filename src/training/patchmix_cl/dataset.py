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
from src.processing.splits import cv_split, patient_val_split
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

    def __init__(self, train_flag, transform, args, sc_reference=None, print_flag=True):
        self.train_flag, self.transform, self.args = train_flag, transform, args
        self.split = "train" if train_flag else "test"
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
        self.filenames = [f for f in stems if official.get(f) == self.split]
        annotations = read_annotations(wav_dir)

        self.encoder = get_encoder(args.encoder)
        cached = None if args.spectrum_correction else self._load_cache()
        if cached:
            self.images, self.labels, self.devices, self.patients = cached
            self.sc = None
        else:
            audio, labels, devices, patients = [], [], [], []
            for f in self.filenames:
                for wave, label in individual_cycles(annotations[f], wav_dir, f, args):
                    audio.append(wave)
                    labels.append(label)
                    devices.append(DEVICES.index(f.split("_")[-1]))
                    patients.append(f.split("_")[0])
            self.labels, self.devices, self.patients = labels, devices, np.array(patients)
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
        by_dev = defaultdict(list)
        for a, d in zip(audio, self.devices):
            by_dev[d].append(a.numpy()[0])
        spectra = {d: mean_spectrum(w) for d, w in by_dev.items()}
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
        corrected = [
            torch.from_numpy(apply_spectrum_correction(w.numpy()[0], coef[d]).astype(np.float32))[
                None
            ]
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


def build_loaders(args):
    """Train, validation (None for a fixed refit) and test loaders.

    Returns:
        Tuple (train_loader, val_loader, test_loader, train_dataset, test_dataset).
    """
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
    return train_loader, val_loader, make(test_dataset, shuffle=False), train_dataset, test_dataset
