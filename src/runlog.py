"""Run logging and artefact paths. Everything lives inside the repo and is gitignored.

    outputs/<exp>/<run_id>/meta.json          once per run: config, command, git commit, hardware
    outputs/<exp>/<run_id>/summary.txt        what the entry point printed (`start_summary`)
    outputs/<exp>/<run_id>/<cell>/seed<N>.json  one record per seed (config, split hash, encoder, metrics)
    checkpoints/<exp>/<run_id>/<cell>/seed<N>.pt  trained heads; other artefacts as <exp>/<run_id>/<name>.pt

`exp` is the config file stem, `cell` one variant inside it (`fm`, `fm+branch`, ...), `run_id` is
`YYYYMMDD-HHMMSS[_<tag>]`. Set `$RUN_TAG` for a human label, or `$RUN_ID` to share one id across
processes. Embedding caches stay in `$cache_root` (data, not results).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_id():
    """Run identifier, fixed for the life of the process.

    Returns:
        `$RUN_ID` if set, else `YYYYMMDD-HHMMSS` plus `_$RUN_TAG` when a tag is set.
    """
    tag = os.environ.get("RUN_TAG")
    return os.environ.setdefault("RUN_ID", time.strftime("%Y%m%d-%H%M%S") + (f"_{tag}" if tag else ""))


def run_dir(exp, kind="outputs"):
    """Directory of this run, created on demand.

    Args:
        exp: Experiment name (config file stem).
        kind: `outputs` or `checkpoints`.

    Returns:
        Path `<kind>/<exp>/<run_id>`.
    """
    d = Path(kind) / exp / run_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def write_meta(exp, cfg):
    """Write `meta.json` for this run once: config, command line, git state, hardware.

    Args:
        exp: Experiment name.
        cfg: Config dict.
    """
    f = run_dir(exp) / "meta.json"
    if f.exists():
        return
    try:
        import torch
        hw = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:
        hw = {}
    f.write_text(json.dumps({"exp": exp, "run_id": run_id(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "argv": sys.argv, "git_commit": _git("rev-parse", "--short", "HEAD"),
                             "git_dirty": bool(_git("status", "--porcelain")), **hw, "config": cfg},
                            indent=1, default=str))


def log_run(exp, cell, seed, cfg, data_hash, encoder, metrics):
    """Write one seed record to `outputs/<exp>/<run_id>/<cell>/seed<seed>.json`.

    Args:
        exp: Experiment name (config file stem).
        cell: Variant within the experiment.
        seed: Random seed.
        cfg: Config dict used for the run.
        data_hash: Hash of the split file used.
        encoder: Encoder name, e.g. "opera_ct_v2".
        metrics: JSON-serialisable dict of results. Negative results are logged too.

    Returns:
        Path of the written file.
    """
    write_meta(exp, cfg)
    out = run_dir(exp) / cell
    out.mkdir(exist_ok=True)
    path = out / f"seed{seed}.json"
    path.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cell": cell, "seed": seed,
                                "data_hash": data_hash, "encoder": encoder, "config": cfg, "metrics": metrics},
                               indent=1, default=float))
    return path


def ckpt_path(exp, cell, seed):
    """Path for a trained head: `checkpoints/<exp>/<run_id>/<cell>/seed<seed>.pt`."""
    d = run_dir(exp, "checkpoints") / cell
    d.mkdir(exist_ok=True)
    return d / f"seed{seed}.pt"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):  # isatty, encoding, ... from the real stdout
        return getattr(self.streams[0], name)


def start_summary(exp):
    """From now on, copy stdout of this process to `outputs/<exp>/<run_id>/summary.txt`.

    Call once at the top of an entry point's `__main__` block.
    """
    sys.stdout = _Tee(sys.__stdout__, open(run_dir(exp) / "summary.txt", "a"))
