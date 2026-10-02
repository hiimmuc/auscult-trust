"""Write the versioned ICBHI patient-disjoint split (70/15/15, seed 0).

Usage: python -m src.make_split configs/tier0_icbhi_baseline.yaml
"""
import sys

import yaml

from src.data.icbhi import cycle_table
from src.data.splits import save_split, split_patients

if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1]))
    split = split_patients(cycle_table(cfg["icbhi_root"]).patient, seed=0)
    print({k: len(v) for k, v in split.items()}, save_split(split, cfg["split_file"]))
