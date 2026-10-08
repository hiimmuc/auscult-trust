import json

import numpy as np

from src.evaluation import lodo


def _write_unit(root, cell, seed, shift, rng):
    d = root / cell / f"seed{seed}"
    d.mkdir(parents=True)

    def probs(n, sharp):
        y = rng.integers(0, 4, n)
        logits = rng.normal(0, 1, (n, 4))
        logits[np.arange(n), y] += sharp
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return p.astype(np.float16), y, rng.integers(0, 12, n) + 100 * (shift > 0)

    cp, cy, cpat = probs(600, 2.0)
    tp, ty, tpat = probs(500, 2.0 - shift)  # a shifted device is predicted less sharply
    np.savez(d / "cal_preds.npz", epochs=np.array([2]), probs=cp[None], labels=cy, device=cy * 0, patient=cpat)
    np.savez(d / "test_preds.npz", epochs=np.array([1, 2]), probs=np.stack([tp, tp]), labels=ty, device=ty * 0, patient=tpat)
    (d / "report.json").write_text(json.dumps({"fixed": {"2": {"test": {"all": {"sp": 60.0, "se": 50.0, "score": 55.0, "hs": 54.5}}}}}))


def test_lodo_pipeline_prefers_the_arm_with_less_shift(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(3):
        _write_unit(tmp_path, "baseline+lodo_litt3200", s, 1.2, rng)  # strong shift
        _write_unit(tmp_path, "sc+sc_static+lodo_litt3200", s, 0.0, rng)  # no shift
    units = lodo.collect(tmp_path, ["baseline", "sc+sc_static"], ["Litt3200"], [0, 1, 2])
    rows = lodo.fold_table(units, ["baseline", "sc+sc_static"])
    base, sc = rows
    assert base["delta"] > 0.1 and abs(sc["delta"]) < 0.08  # shifted device under-covers, unshifted is near nominal
    stat = lodo.h2_statistic(units, "baseline", "sc+sc_static", ["Litt3200"], n_boot=50)
    assert stat["D"] > 0.05 and stat["ci"][0] < stat["D"] < stat["ci"][1] + 1e-9
    gate = lodo.tested_folds(units, "baseline", n_split=20) if False else lodo.tested_folds(units, "baseline", n_splits=20)
    assert gate["Litt3200"][0] > gate["Litt3200"][1]  # the shift is outside the no-shift null
