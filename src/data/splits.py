"""Patient-disjoint splits, device-held-out splits, and versioned split files.

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


def device_holdout(df, device, val_frac=0.15, seed=0):
    """Hold out one whole device as test. Ignores any official split.

    Any patient recorded on the held-out device is removed from train and val,
    so patients stay disjoint.

    Args:
        df: Cycle table with `patient` and `device` columns.
        device: Device name to hold out.
        val_frac: Fraction of remaining patients used for validation.
        seed: RNG seed.

    Returns:
        Dict with `train`, `val`, `test` lists of patient ids.
    """
    test = set(df.loc[df.device == device, "patient"])
    rest = set(df.patient) - test
    s = split_patients(rest, (1 - val_frac, val_frac), seed, ("train", "val"))
    return {**s, "test": sorted(test)}


def assert_disjoint(split):
    """Raise if any patient appears in more than one split.

    Args:
        split: Dict name -> list of patient ids.
    """
    seen = {}
    for name, ids in split.items():
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
