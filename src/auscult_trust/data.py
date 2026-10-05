"""Cached features of all ICBHI cycles (`src.features`), the train+val ("dev") and test tables, patient-grouped folds."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.data.icbhi import cycle_table
from src.data.splits import load_split, split_frames
from src.encoders.base import DEVICE, N_FRAMES
from src.features import cache_key


class FeatureData:
    """Cached features of all cycles plus the train+val ("dev") and test tables."""

    def __init__(self, cfg):
        df = cycle_table(cfg["icbhi_root"])
        out = Path(cfg["cache_dir"])
        assert json.loads((out / "keys.json").read_text()) == [cache_key(r) for r in df.itertuples()], "cache/table mismatch"
        done = np.load(out / "done.npy")
        if not done.all():
            assert os.environ.get("ALLOW_PARTIAL"), f"{(~done).sum()} rows not cached: run src.features first"
        df["row"] = np.arange(len(df))
        self.y = df.label.to_numpy()  # label of every cache row
        part = split_frames(df[done], load_split(cfg["split_file"])[0])
        self.dev = pd.concat([part["train"], part["val"]], ignore_index=True)
        self.test = part["test"]
        self.layers = torch.from_numpy(np.load(out / "layers.npy"))  # (N, L, 32, D) float16, RAM
        self.tokens = np.load(out / "tokens.npy", mmap_mode="r") if (out / "tokens.npy").exists() else None  # ViT encoders only
        k = np.clip(np.round((df.end - df.start).clip(upper=8).values / 8 * N_FRAMES), 1, N_FRAMES).astype(int)
        self.mask = torch.from_numpy(np.arange(N_FRAMES)[None] < k[:, None])  # real (non-repeated) frames
        meta = json.loads((out / "meta.json").read_text())
        self.encoder, self.token_layer = meta["encoder"], meta["token_layer"]

    def batch(self, rows, mode):
        b = {"mask": self.mask[rows].to(DEVICE)}
        if mode == "frames":
            b["f"] = self.layers[rows].to(DEVICE).float()
        else:
            b["tok"] = torch.from_numpy(np.stack([self.tokens[i] for i in rows])).to(DEVICE).float()
        return b


def folds(D, k):
    """Patient-grouped folds over the dev table: list of (train rows, held-out rows)."""
    return [(D.dev.row.values[a], D.dev.row.values[b])
            for a, b in GroupKFold(k).split(D.dev, D.dev.label, D.dev.patient)]
