"""Patient-disjoint splits, the E1 device-held-out protocol, and versioned split files.

Group key is always the patient id. For KAUH the loader must give all three filter
renderings of one patient the same patient id, so they land in the same split.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def split_patients(patients, fracs=(0.7, 0.15, 0.15), seed=0, names=("train", "val", "test")):
    """Randomly assign unique patients to named splits.

    Args:
        patients: Iterable of patient ids (duplicates allowed).
        fracs: Fraction of patients per split. Must sum to 1.
        seed: RNG seed.
        names: Split names, same length as `fracs`.

    Returns:
        Dict name -> sorted list of patient ids.
    """
    ids = sorted(set(patients))
    np.random.default_rng(seed).shuffle(ids)
    cuts = np.round(np.cumsum(fracs)[:-1] * len(ids)).astype(int)
    return {n: sorted(part) for n, part in zip(names, np.split(np.array(ids, dtype=object), cuts))}


def split_frames(df, split):
    """Cut a cycle table into split parts.

    Args:
        df: Cycle table with `patient` and `stem` columns.
        split: Dict name -> patient ids. Optional key `drop` lists recording stems to exclude.

    Returns:
        Dict name -> DataFrame (index reset), without `drop`.
    """
    df = df[~df.stem.isin(split.get("drop", []))]
    return {k: df[df.patient.isin(v)].reset_index(drop=True) for k, v in split.items() if k != "drop"}


def assert_disjoint(split):
    """Raise if any patient appears in more than one split.

    Args:
        split: Dict name -> list of patient ids.
    """
    seen = {}
    for name, ids in split.items():
        if name == "drop":
            continue
        for p in ids:
            assert p not in seen, f"patient {p} in both {seen[p]} and {name}"
            seen[p] = name


def save_split(split, path):
    """Write a split file with a content hash. Refuses to overwrite.

    Args:
        split: Dict name -> list of patient ids.
        path: Output json path.

    Returns:
        The sha256 hash stored in the file.
    """
    assert_disjoint(split)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. Split files are versioned, write a new name.")
    h = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hash": h, "split": split}, indent=1))
    return h


def load_split(path):
    """Read a split file and verify its hash.

    Args:
        path: Json path written by `save_split`.

    Returns:
        Tuple (split dict, hash).
    """
    d = json.loads(Path(path).read_text())
    h = hashlib.sha256(json.dumps(d["split"], sort_keys=True).encode()).hexdigest()
    assert h == d["hash"], f"{path} was modified"
    return d["split"], h


def patient_device_crosstab(df):
    """Patients x devices cycle counts. A patient with two non-zero columns breaks device-held-out disjointness (gate G0)."""
    return df.pivot_table(index="patient", columns="device", values="stem", aggfunc="size", fill_value=0)


def device_holdout_official(df, device):
    """E1 protocol: hold out one device, keep the official patient-disjoint split, flag seen vs unseen content.

    Head training uses the official train cycles of the other devices; calibration uses the official test
    cycles of the other devices; the held-out device's official test cycles are `test_unseen` (not in OPERA
    pretraining) and its official train cycles are `test_seen` (in OPERA pretraining). Patients that also
    recorded on the held-out device are dropped from the other parts, so patients stay disjoint.

    Args:
        df: Cycle table with `patient`, `device`, `official` ("train"/"test") columns.
        device: Device name to hold out.

    Returns:
        Dict of DataFrames `train`, `cal`, `test_unseen`, `test_seen` (index reset), plus `n_dropped` (int, cycles
        of shared patients removed from `train`/`cal`).
    """
    held = df.device == device
    shared = set(df.loc[held, "patient"])
    other = df[~held]
    dropped = other.patient.isin(shared)
    other = other[~dropped]

    def part(d, split):
        return d[d.official == split].reset_index(drop=True)

    out = {"train": part(other, "train"), "cal": part(other, "test"),
           "test_unseen": part(df[held], "test"), "test_seen": part(df[held], "train")}
    assert not set(out["train"].patient) & set(out["cal"].patient)
    assert not set(out["train"].patient) & set(out["test_unseen"].patient) and not set(out["cal"].patient) & set(out["test_unseen"].patient)
    return {**out, "n_dropped": int(dropped.sum())}


def calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0):
    """Repeatedly split the official test patients of the non-held-out devices into calibration and evaluation groups.

    Stratified by device so every device's patients are divided in roughly `fracs` proportion in every split. The
    cut point uses stochastic rounding (seed-determined) rather than a fixed round(), so a device whose patient
    count does not divide evenly by `fracs` is not biased toward the same group size in every one of the
    `n_splits` re-splits; a device with only one patient alternates which group it lands in across splits.

    Args:
        df: DataFrame with `patient`, `device` columns (e.g. `device_holdout_official(...)["cal"]`).
        n_splits: Number of independent re-splits.
        fracs: (calibration, evaluation) patient fractions, applied per device.
        seed: Base seed; split i uses seed `seed + i`.

    Returns:
        List of `n_splits` dicts `{"calibration": [...], "evaluation": [...]}`, sorted patient id lists.
    """
    by_device = {d: sorted(set(g.patient)) for d, g in df.groupby("device")}
    out = []
    for i in range(n_splits):
        rng = np.random.default_rng(seed + i)
        cal, ev = [], []
        for ids in by_device.values():
            ids = list(ids)
            rng.shuffle(ids)
            n = len(ids)
            target = n * fracs[0]
            cut = int(target) + (1 if rng.random() < target - int(target) else 0)  # stochastic rounding
            if n > 1:
                cut = min(max(cut, 1), n - 1)
            cal.extend(ids[:cut])
            ev.extend(ids[cut:])
        out.append({"calibration": sorted(cal), "evaluation": sorted(ev)})
    return out


def with_official(df, split):
    """Add the `official` column ("train" = train+val patients, "test") from a split file; drops `split["drop"]` stems.

    Use when the cycle table was built without the official split file (`cycle_table` leaves `official` empty).
    """
    df = df[~df.stem.isin(split.get("drop", []))].copy()
    side = {p: ("test" if k == "test" else "train") for k, v in split.items() if k != "drop" for p in v}
    df["official"] = df.patient.map(side)
    return df[df.official.notna()].reset_index(drop=True)


def patient_val_split(patients, frac=0.2, seed=0):
    """Hold out whole patients of the training set as a validation set.

    Args:
        patients: (n,) patient id per training cycle.
        frac: Fraction of training patients for validation.
        seed: RNG seed (fixed per run so that all variants of one seed use the same split).

    Returns:
        Tuple (train_idx, val_idx), disjoint by patient.
    """
    patients = np.asarray(patients)
    ids = sorted(set(patients.tolist()))
    rng = np.random.default_rng(10_000 + seed)
    rng.shuffle(ids)
    val = set(ids[:max(1, int(round(frac * len(ids))))])
    is_val = np.array([p in val for p in patients.tolist()])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def patient_folds(patients, devices, n_folds=3, seed=12345):
    """Assign every training patient to one of `n_folds` grouped-CV folds, stratified by the patient's main device.

    The assignment depends only on the patient list and `seed`, not on the run seed, so every variant and every seed
    sees the same folds (paired screening). Devices with few patients (Litt3200 in the official train split) are spread
    over the folds instead of landing in one.

    Args:
        patients: (n,) patient id per training cycle.
        devices: (n,) device id per training cycle.
        n_folds: Number of folds.
        seed: RNG seed of the assignment (fixed by default; do not tie it to the run seed).

    Returns:
        Dict patient id -> fold index in [0, n_folds).
    """
    patients, devices = np.asarray(patients), np.asarray(devices)
    main_dev = {}
    for p in sorted(set(patients.tolist())):
        vals, counts = np.unique(devices[patients == p], return_counts=True)
        main_dev[p] = int(vals[np.argmax(counts)])
    rng = np.random.default_rng(seed)
    fold, k = {}, 0
    for d in sorted(set(main_dev.values())):
        ids = [p for p in sorted(main_dev) if main_dev[p] == d]
        rng.shuffle(ids)
        for p in ids:  # round-robin across devices, so fold sizes stay balanced overall
            fold[p] = k % n_folds
            k += 1
    return fold


def cv_split(patients, devices, n_folds, fold):
    """(train_idx, val_idx) of grouped-CV fold `fold`; disjoint by patient. See `patient_folds`."""
    assign = patient_folds(patients, devices, n_folds)
    is_val = np.array([assign[p] == fold for p in np.asarray(patients).tolist()])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)
