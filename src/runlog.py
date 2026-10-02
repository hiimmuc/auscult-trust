"""Run logging. Every run records config, seed, data hash and encoder."""
import json
import time
from pathlib import Path


def log_run(name, cfg, seed, data_hash, encoder, metrics, root="experiments"):
    """Write one run record to `<root>/<name>/seed<seed>.json`.

    Args:
        name: Experiment name (usually the config file stem).
        cfg: Config dict used for the run.
        seed: Random seed.
        data_hash: Hash of the split file used.
        encoder: Encoder name, e.g. "opera_ct" or "ast".
        metrics: JSON-serialisable dict of results. Negative results are logged too.
        root: Output root directory.

    Returns:
        Path of the written file.
    """
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed{seed}.json"
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg, "seed": seed,
           "data_hash": data_hash, "encoder": encoder, "metrics": metrics}
    path.write_text(json.dumps(rec, indent=1, default=float))
    return path
