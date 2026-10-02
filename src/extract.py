"""Extract and cache frozen OPERA-CT embeddings for every ICBHI cycle.

Usage: python -m src.extract configs/tier0_icbhi_baseline.yaml
"""
import sys

import torch
import yaml

from src.data.icbhi import cycle_table
from src.models.cache import extract
from src.models.opera import OperaCT

if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1]))
    torch.set_num_threads(8)
    print(extract(cycle_table(cfg["icbhi_root"]), OperaCT(), cfg["encoder"], cfg["cache_root"]))
