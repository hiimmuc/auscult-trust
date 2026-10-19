"""PC-MCL (patient-consistent multi-cycle learning) as in `../repos/PC-MCL` (`util/dataset_pcmcl.py`, `main.py`).

Training items, per epoch: every training cycle once (3-label target normal, crackle, wheeze), plus `pcmcl_mix` x n
multi-cycle samples (2 or 3 short cycles concatenated, label = element-wise OR of the parts, the combinations of the
reference "additive" strategy) and `pcmcl_ssl` x n patient-matching samples (two cycles of one patient = 1, of two
patients with the same pathology = 0; the reference takes the first pathology group with two patients, in practice
normal-only patients). Concatenations are cut/padded to `desired_length` like single cycles.
Loss: BCE on the 3 labels + `pcmcl_aux` x cross-entropy of the patient-matching head (on its samples only).
Prediction: crackle and wheeze sigmoids as independent events give the 4-class distribution, whose argmax is the
reference rule (threshold 0.5 on each label).
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .cycles import cut_pad, individual_cycles, read_annotations

# (parts, 3-label result) of the reference "additive" strategy; 4-class ids: 0 normal, 1 crackle, 2 wheeze, 3 both
COMBINATIONS = [((0, 1), (1, 1, 0)), ((0, 2), (1, 0, 1)), ((1, 2), (0, 1, 1)),
                ((0, 0), (1, 0, 0)), ((1, 1), (0, 1, 0)), ((2, 2), (0, 0, 1)),
                ((0, 3), (1, 1, 1)), ((3, 1), (0, 1, 1)), ((3, 2), (0, 1, 1)), ((3, 3), (0, 1, 1)),
                ((0, 1, 2), (1, 1, 1))]


def multilabel(label):
    """4-class id -> [normal, crackle, wheeze]."""
    return [int(label == 0), int(label in (1, 3)), int(label in (2, 3))]


def four_class_probs(logits):
    """(n, 3) logits -> (n, 4) probabilities of normal, crackle, wheeze, both (crackle and wheeze independent)."""
    s = torch.sigmoid(logits.float())
    c, w = s[:, 1], s[:, 2]
    return torch.stack([(1 - c) * (1 - w), c * (1 - w), (1 - c) * w, c * w], 1)


def four_class(target):
    """(n, 3) multi-hot targets -> 4-class id (normal unless crackle or wheeze)."""
    return (target[:, 1] > 0.5).long() + 2 * (target[:, 2] > 0.5).long()


class PCMCLTrain(Dataset):
    """Training items of PC-MCL over the cycles `idx` of `ds` (an `ICBHIDataset` of the training split).

    Items: (image, 3-label target, device, patient, patient-matching label or -1). Images of multi-cycle samples are
    computed on the fly, so the loader runs in the main process (`num_workers=0`; the encoder front end uses the GPU).
    """

    def __init__(self, ds, idx, args):
        assert not args.spectrum_correction, "pcmcl: SC on concatenated cycles is not implemented"
        self.ds, self.idx, self.args = ds, np.asarray(idx), args
        wav_dir = f"{args.data_folder}/audio_test_data"
        ann = read_annotations(wav_dir)
        raw = [w for f in ds.filenames for w, _ in individual_cycles(ann[f], wav_dir, f, args, raw=True)]
        assert len(raw) == len(ds.labels), (len(raw), len(ds.labels))
        self.raw = {i: raw[i] for i in self.idx}
        labels = np.asarray(ds.labels)
        half = args.desired_length * args.sample_rate // 2
        self.short = {k: [i for i in self.idx if labels[i] == k and self.raw[i].shape[-1] < half] for k in range(4)}
        self.by_patient = {}
        for i in self.idx:
            self.by_patient.setdefault(ds.patients[i], []).append(i)
        self.n = len(self.idx)
        self.n_mix, self.n_ssl = int(self.n * args.pcmcl_mix), int(self.n * args.pcmcl_ssl)

    def __len__(self):
        return self.n + self.n_mix + self.n_ssl

    def _image(self, parts):
        a = self.args
        wave = cut_pad(torch.cat([self.raw[i] for i in parts], -1), a.desired_length, a.sample_rate, a.pad_types)
        image = self.ds.encoder.preprocess(wave)
        return self.ds.transform(image) if self.ds.transform is not None else image

    def _item(self, parts, target, ssl):
        pats = {int(self.ds.patients[i]) for i in parts}
        return (self._image(parts), torch.tensor(target, dtype=torch.float32), self.ds.devices[parts[0]],
                pats.pop() if len(pats) == 1 else -1, float(ssl))  # patient -1: parts of several patients

    def _mixed(self):
        classes, target = random.choice(COMBINATIONS)
        if len(set(classes)) == 1 and len(self.short[classes[0]]) >= 2:
            return self._item(random.sample(self.short[classes[0]], 2), target, -1)
        if len(set(classes)) == len(classes) and all(self.short[k] for k in classes):
            return self._item([random.choice(self.short[k]) for k in classes], target, -1)
        return self._single(random.randrange(self.n))  # not enough short cycles: a plain cycle, as the reference

    def _or(self, parts):
        return [max(multilabel(self.ds.labels[i])[j] for i in parts) for j in range(3)]

    def _patient_pair(self):
        if random.random() < 0.5:  # same patient (>= 3 cycles if any): 70% a non-consecutive pair, else a consecutive one
            pids = [p for p, c in self.by_patient.items() if len(c) >= 3] or [p for p, c in self.by_patient.items() if len(c) >= 2]
            cyc = self.by_patient[random.choice(pids)]
            pairs = [(i, j) for i in range(len(cyc)) for j in range(i + 2, len(cyc))]
            if pairs and random.random() < 0.7:
                i, j = random.choice(pairs)
            else:
                i = random.randrange(len(cyc) - 1)
                j = i + 1
            parts = [cyc[i], cyc[j]]
            return self._item(parts, self._or(parts), 1)
        # hard negative, as the reference: the first patient group (in its order) with two patients, cycles of its pathology
        has = {p: [any(multilabel(self.ds.labels[i])[k] for i in c) for k in range(3)] for p, c in self.by_patient.items()}
        order = [("normal_only", (1, 0, 0)), ("crackle_only", (0, 1, 0)), ("wheeze_only", (0, 0, 1)),
                 ("crackle_wheeze", (0, 1, 1)), ("normal_crackle", (1, 1, 0)), ("normal_wheeze", (1, 0, 1)), ("all_types", (1, 1, 1))]
        for name, flags in order:
            pids = [p for p, h in has.items() if tuple(map(int, h)) == flags]
            if len(pids) < 2:
                continue
            p1, p2 = random.sample(pids, 2)
            keep = {"normal_only": lambda m: m == [1, 0, 0], "crackle_only": lambda m: m == [0, 1, 0],
                    "wheeze_only": lambda m: m == [0, 0, 1], "crackle_wheeze": lambda m: m[1] and m[2]}.get(name, lambda m: True)
            c1, c2 = ([i for i in self.by_patient[p] if keep(multilabel(self.ds.labels[i]))] for p in (p1, p2))
            if c1 and c2:
                parts = [random.choice(c1), random.choice(c2)]
                return self._item(parts, self._or(parts), 0)
        p1, p2 = random.sample(list(self.by_patient), 2)  # fallback: any two patients
        parts = [random.choice(self.by_patient[p1]), random.choice(self.by_patient[p2])]
        return self._item(parts, self._or(parts), 0)

    def _single(self, k):
        i = self.idx[k]
        image, label, device, patient, _ = self.ds[i]
        return image, torch.tensor(multilabel(label), dtype=torch.float32), device, patient, -1.0

    def __getitem__(self, k):
        if k < self.n:
            return self._single(k)
        return self._mixed() if k < self.n + self.n_mix else self._patient_pair()
