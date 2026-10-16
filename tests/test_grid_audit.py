import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("grid", "scripts/grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)
UNIT = Path("outputs/train/grid-v1/ast+ft+ce/seed0")


@pytest.mark.skipif(not (UNIT / "report.json").exists(), reason="needs a finished grid-v1 unit")
def test_audit_passes_a_real_unit_and_flags_corruption(tmp_path, monkeypatch):
    d = tmp_path / UNIT
    shutil.copytree(UNIT, d)
    monkeypatch.setattr(grid, "REPO", tmp_path)
    assert grid.audit() == 0
    args = json.loads((d / "train_args.json").read_text())
    (d / "train_args.json").write_text(json.dumps({**args, "learning_rate": 1.0}))
    z = dict(np.load(d / "test_preds.npz"))
    z["probs"][-1, 0] = np.nan
    z["preds"] = z["preds"][:-1]  # one epoch missing
    z["epochs"] = z["epochs"][:-1]
    np.savez(d / "test_preds.npz", **z)
    assert grid.audit() >= 3
