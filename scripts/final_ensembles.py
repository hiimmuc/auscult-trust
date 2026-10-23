"""Frozen E1/E1+E3 extensions on KAUH and recording windows; raw mean softmax, no fitting.

Run after final_evaluation.py and after retrieving the fixed member checkpoints. Classification
with k-rate on the original cycle benchmark remains a separate exploratory result.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from final_evaluation import BASE, SC, RUN, event, export_ok, run, sha
from src.evaluation.lodo_ttsc import report_epoch
from src.artifacts import file_hashes, record_prediction, atomic_npz, immutable_json

E1 = ["beats+ft+pafa", "beats+ft+sg_scl", "beats+ft+patchmix_cl"]
E3 = ["htsat+ft+patchmix_cl", "htsat+ft+sg_scl"]


def mean_exports(paths, out, member_hashes):
    arrays = [dict(np.load(p, allow_pickle=True)) for p in paths]
    keys = [k for k in arrays[0] if k not in ("probs", "ckpt_sha256", "view_probs")]
    for k in keys:
        assert all(np.array_equal(z[k], arrays[0][k]) for z in arrays[1:]), (out, k)
    mean = np.mean([z["probs"].astype(np.float64) for z in arrays], axis=0)
    assert np.isfinite(mean).all() and np.allclose(mean.sum(1), 1, atol=1e-5)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(member_hashes, sort_keys=True).encode()).hexdigest()
    atomic_npz(out, **{k: arrays[0][k] for k in keys}, probs=mean,
                        ckpt_sha256=np.array(digest), member_hashes=np.array(json.dumps(member_hashes)))
    record_prediction(out, {"schema": 1, "rule": "equal_weight_mean_softmax",
                           "files": file_hashes([*paths, __file__]), "member_hashes": member_hashes})
    event(status="ensemble_exported", artifact=str(out), sha256=sha(out), members=member_hashes)


def main():
    for sc in (False, True):
        arm = "auscult" if sc else "baseline"
        suffix = "+sc+sc_static" if sc else ""
        for seed in range(5):
            paths, member_hashes = {}, {}
            for cell in E1 + E3:
                unit = (Path("outputs/train/lodo-v1") / (cell + suffix) if sc else Path("outputs/train/grid-v1") / cell) / f"seed{seed}"
                epoch = report_epoch(unit)
                ckpt = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / f"report_epoch_{epoch}.pth"
                ckpt_sha = sha(ckpt)
                member_hashes[cell] = ckpt_sha
                inputs = [ckpt, unit / "train_args.json", unit / "test_preds.npz"]
                if sc:
                    inputs.append(unit / "sc_state.npz")
                frozen = Path("outputs/train") / RUN / ("member_" + cell + suffix) / f"seed{seed}"
                frozen.mkdir(parents=True, exist_ok=True)
                immutable_json(frozen / "freeze.json", {"unit": str(unit), "epoch": epoch,
                    "seed": seed, "sc": sc, "inputs": {str(p): sha(p) for p in inputs}})
                for dataset, module in (("screening", "screening"), ("kauh", "kauh_infer")):
                    out = (Path("outputs") / dataset / RUN / arm / f"seed{seed}.npz" if cell == E1[0]
                           else Path("outputs") / dataset / RUN / "members" / (cell + suffix) / f"seed{seed}.npz")
                    if not export_ok(out, ckpt_sha, sc, unit=unit, checkpoint=ckpt, dataset=dataset):
                        if cell == E1[0]:
                            raise RuntimeError("Run final_evaluation.py first: " + str(out))
                        run(["-m", "src.evaluation." + module, *(["infer"] if dataset == "screening" else []),
                             unit, "--ckpt", ckpt.name, "--out", out, *(["--sc"] if sc else [])])
                        export_ok(out, ckpt_sha, sc, unit=unit, checkpoint=ckpt, dataset=dataset)
                    paths[cell, dataset] = out
            for name, cells in (("e1", E1), ("e1_e3", E1 + E3)):
                hashes = {c: member_hashes[c] for c in cells}
                for dataset in ("screening", "kauh"):
                    out = Path("outputs") / dataset / RUN / "extensions" / name / arm / f"seed{seed}.npz"
                    mean_exports([paths[c, dataset] for c in cells], out, hashes)
    for name in ("e1", "e1_e3"):
        kr = f"outputs/kauh/{RUN}/extensions/{name}"
        sr = f"outputs/screening/{RUN}/extensions/{name}"
        run(["-m", "src.evaluation.kauh", "--dir", kr, "--base", "baseline", "--arms", "auscult",
             "--boot", "1000", "--null_splits", "200"])
        run(["-m", "src.evaluation.screening", "analyse", "--dir", sr, "--kauh_dir", kr,
             "--pairs", "baseline:baseline", "auscult:auscult"])
    event(status="extensions_complete", groups=["E1", "E1+E3"], rule="raw equal-weight mean softmax")


if __name__ == "__main__":
    main()
