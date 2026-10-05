"""Write a versioned ICBHI patient-disjoint split file.

With `official_split` in the config: official test patients as test, the rest of the official
train patients split 85/15 into train/val. Two patients have recordings on both official sides:
they go to test, and their official-train recordings are listed under `drop` and never used.
So the split is patient-disjoint and the test set equals the official one.
Without it: random 70/15/15 patient split (seed 0).

Usage: python -m src.make_split configs/split_icbhi_official.yaml
"""
import sys
from pathlib import Path

import yaml

from src.data.icbhi import cycle_table
from src.data.splits import save_split, split_patients


def official_split(df, official, val_frac=0.15, seed=0):
    """Patient split from the official recording-level ICBHI split.

    Args:
        df: Cycle table.
        official: Dict recording stem -> "train"/"test".
        val_frac: Fraction of official-train patients used for validation.
        seed: RNG seed.

    Returns:
        Dict with `train`, `val`, `test` lists of patient ids and `drop` list of recording stems.
    """
    side = df.stem.map(official)
    assert side.notna().all(), f"stems missing from official split: {df.stem[side.isna()].unique()[:5]}"
    test = set(df.patient[side == "test"])
    s = split_patients(set(df.patient) - test, (1 - val_frac, val_frac), seed, ("train", "val"))
    drop = sorted(set(df.stem[(side == "train") & df.patient.isin(test)]))
    return {**s, "test": sorted(test), "drop": drop}


if __name__ == "__main__":
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    df = cycle_table(cfg["icbhi_root"])
    if cfg.get("official_split"):
        off = dict(line.split()[:2] for line in Path(cfg["official_split"]).read_text().splitlines() if line.strip())
        split = official_split(df, off)
    else:
        split = split_patients(df.patient, seed=0)
    print({k: len(v) for k, v in split.items()}, save_split(split, cfg["split_file"]))
