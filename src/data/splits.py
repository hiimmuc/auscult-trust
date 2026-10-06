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

    Stratified by device so every device's patients are divided in roughly `fracs` proportion in every split. A
    device with only one patient alternates which group that patient goes to across splits (seed-determined),
    rather than always being dropped into the same group or excluded.

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
            if len(ids) == 1:
                (cal if rng.random() < fracs[0] else ev).append(ids[0])
                continue
            cut = min(max(1, round(len(ids) * fracs[0])), len(ids) - 1)
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
