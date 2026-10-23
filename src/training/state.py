"""Epoch-boundary restart state and validated completion records."""
import json
import random

import numpy as np
import torch

from src.artifacts import atomic_json, verify_file
from src.runs import sha256_file


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if state is None:
        raise ValueError("Legacy checkpoint has no RNG state; use a new RUN_ID for reproducible training")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def publish_completion(out, checkpoints):
    files = [p for p in out.iterdir() if p.suffix in (".json", ".npz") and p.name != "completion.json"]
    files += [p for p in checkpoints.glob("*.pth") if p.name != "last.pth"]
    atomic_json(out / "completion.json", {"schema": 1, "files": {
        str(p.relative_to(out)) if p.is_relative_to(out) else str(p.resolve()): sha256_file(p) for p in files}})


def completed(out, checkpoints=None):
    if not (out / "report.json").exists() and not (out / "completion.json").exists():
        return False
    marker = out / "completion.json"
    if not marker.exists():
        if checkpoints is not None and (checkpoints / "last.pth").exists():
            return False  # interrupted finalization: replay it from the final saved epoch
        raise ValueError(f"Uncertified legacy/partial unit {out}; use a new RUN_ID (do not overwrite research records)")
    record = json.loads(marker.read_text())
    if record.get("schema") != 1 or "report.json" not in record.get("files", {}):
        raise ValueError(f"Invalid completion record: {out}")
    for path, expected in record["files"].items():
        verify_file(out / path, expected)
    report = json.loads((out / "report.json").read_text())
    if not report.get("history"):
        raise ValueError(f"Empty training report: {out}")
    return True
