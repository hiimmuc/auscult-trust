"""Run directories: one new directory per run, never overwritten. Everything lives in the repo and is gitignored.

    outputs/<exp>/<run_id>/meta.json            once per run: config, command, git commit, hardware
    outputs/<exp>/<run_id>/summary.txt          what the entry point printed (`start_summary`)
    outputs/<exp>/<run_id>/<cell>/<unit>/       small artefacts of one unit (reports, predictions, config)
    checkpoints/<exp>/<run_id>/<cell>/<unit>/   weights of that unit
    outputs/<exp>/latest -> <run_id>            symlink to the newest run

`exp` is the experiment name (config file stem), `cell` one variant, `unit` one fit (`cv0`, `seed3`, `fix_seed3`),
`run_id` is `YYYYMMDD-HHMMSS[_<tag>]`. Set `$RUN_TAG` for a human label, or `$RUN_ID` to resume or share one id across
processes: a fresh id never touches earlier runs.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.paths import CHECKPOINTS, OUTPUTS

_KINDS = {"outputs": OUTPUTS, "checkpoints": CHECKPOINTS}


def run_id():
    """Run identifier, fixed for the life of the process.

    Returns:
        `$RUN_ID` if set, else `YYYYMMDD-HHMMSS` plus `_$RUN_TAG` when a tag is set.
    """
    tag = os.environ.get("RUN_TAG")
    return os.environ.setdefault(
        "RUN_ID", time.strftime("%Y%m%d-%H%M%S") + (f"_{tag}" if tag else "")
    )


def run_dir(exp, kind="outputs"):
    """Directory of this run, created on demand; `<kind>/<exp>/latest` is pointed at it.

    Args:
        exp: Experiment name.
        kind: `outputs` or `checkpoints`.

    Returns:
        Path `<kind>/<exp>/<run_id>`.
    """
    d = _KINDS[kind] / exp / run_id()
    d.mkdir(parents=True, exist_ok=True)
    latest = d.parent / "latest"
    if latest.is_symlink() or not latest.exists():
        latest.unlink(missing_ok=True)
        latest.symlink_to(d.name)
    return d


def unit_dir(exp, cell, unit, kind="outputs"):
    """Directory of one fit inside this run, created on demand. Existing content is kept (resume), never replaced."""
    d = run_dir(exp, kind) / cell / unit
    d.mkdir(parents=True, exist_ok=True)
    return d


def twin_dir(path, kind="checkpoints"):
    """The `<kind>` directory that mirrors an `outputs/` or `checkpoints/` unit directory."""
    path = Path(path).resolve()
    for root in _KINDS.values():
        if path.is_relative_to(root.resolve()):
            return _KINDS[kind] / path.relative_to(root.resolve())
    raise ValueError("{} is neither under outputs/ nor checkpoints/".format(path))


def sha256_file(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, cwd=Path(__file__).parent
        ).stdout.strip()
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

        hw = {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        hw = {}
    f.write_text(
        json.dumps(
            {
                "exp": exp,
                "run_id": run_id(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "argv": sys.argv,
                "git_commit": _git("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                **hw,
                "config": cfg,
            },
            indent=1,
            default=str,
        )
    )


def log_run(exp, cell, seed, cfg, metrics, data_hash=None):
    """Write one seed record to `outputs/<exp>/<run_id>/<cell>/seed<seed>.json`.

    Args:
        exp: Experiment name.
        cell: Variant within the experiment.
        seed: Random seed.
        cfg: Config dict used for the run.
        metrics: JSON-serialisable dict of results. Negative results are logged too.
        data_hash: Hash of the split file used.

    Returns:
        Path of the written file.
    """
    write_meta(exp, cfg)
    out = run_dir(exp) / cell
    out.mkdir(exist_ok=True)
    path = out / f"seed{seed}.json"
    path.write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cell": cell,
                "seed": seed,
                "data_hash": data_hash,
                "config": cfg,
                "metrics": metrics,
            },
            indent=1,
            default=float,
        )
    )
    return path


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
