"""Run the frozen PAFA baseline/static-SC evaluation with an append-only command log.

Run from the repository root with the training Python environment. Existing prediction exports are
reused only when their checkpoint hash and SC flag match. No training or model selection is done.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from src.runs import sha256_file as sha

RUN = "final-pafa-static"
LOG = Path("logs") / RUN
BASE = Path("outputs/train/grid-v1/beats+ft+pafa")
SC = Path("outputs/train/lodo-v1/beats+ft+pafa+sc+sc_static")


def event(**data):
    LOG.mkdir(parents=True, exist_ok=True)
    data["utc"] = datetime.now(timezone.utc).isoformat()
    with (LOG / "events.jsonl").open("a") as f:
        f.write(json.dumps(data) + "\n")
    print(json.dumps(data), flush=True)


def run(args):
    cmd = [sys.executable, *map(str, args)]
    event(status="started", command=shlex.join(cmd))
    result = subprocess.run(cmd)
    event(status="passed" if result.returncode == 0 else "failed", command=shlex.join(cmd), exit_code=result.returncode)
    result.check_returncode()


def export_ok(path, checkpoint_sha, sc):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=True) as z:
        assert str(z["ckpt_sha256"]) == checkpoint_sha, f"checkpoint mismatch: {path}"
        assert bool(z["sc"]) == sc, f"SC mismatch: {path}"
        assert np.isfinite(z["probs"]).all()
        assert np.allclose(z["probs"].sum(1), 1, atol=1e-5)
    event(status="verified", artifact=str(path), sha256=sha(path))
    return True


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ["PYTHONPATH"] = "."
    event(status="campaign_started", run=RUN,
          git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          script_sha256=sha(__file__))
    # Record the exact evaluation inputs, including all audio and labels used downstream.
    from src.paths import DATA
    from src.evaluation.lodo_ttsc import load_args
    data_files = sorted((DATA / "raw/kauh").rglob("*"))
    args = load_args(BASE / "seed0")
    folder = Path(args.data_folder)
    data_files += sorted((folder / "audio_test_data").glob("*"))
    data_files.append(folder / "official_split.txt")
    hashes = {str(p): sha(p) for p in data_files if p.is_file()}
    input_path = LOG / "input_hashes.json"
    input_path.write_text(json.dumps(hashes, indent=2) + "\n")
    data_sha = sha(input_path)
    for seed in range(5):
        for name, parent, sc in (("baseline", BASE, False), ("auscult", SC, True)):
            unit = parent / f"seed{seed}"
            checkpoint = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / "report_epoch_100.pth"
            checkpoint_sha = sha(checkpoint)
            inputs = [checkpoint, unit / "train_args.json", unit / "test_preds.npz", unit / "report.json"]
            if sc:
                inputs.append(unit / "sc_state.npz")
            frozen = Path("outputs/train") / RUN / name / f"seed{seed}"
            frozen.mkdir(parents=True, exist_ok=True)
            manifest = {"condition": name, "seed": seed, "epoch": 100, "unit": str(unit),
                        "checkpoint_sha256": checkpoint_sha, "data_manifest_sha256": data_sha,
                        "inputs": {str(p): sha(p) for p in inputs}, "sc": sc}
            if sc:
                with np.load(unit / "sc_state.npz") as z:
                    assert str(z["sc_mode"]) == "static"
                    manifest["s_ref_sha256"] = hashlib.sha256(z["reference"].tobytes()).hexdigest()
            (frozen / "freeze.json").write_text(json.dumps(manifest, indent=2) + "\n")
            event(status="frozen", condition=name, seed=seed, checkpoint_sha256=checkpoint_sha)
            export = unit / "export"
            if not (export / "model.sha256").exists():
                run(["-m", "src.training.patchmix_cl.export_probs", unit, "--ckpt", checkpoint.name, "--num_workers", "0"])
            assert (export / "model.sha256").read_text().split()[0] == checkpoint_sha
            for dataset, module in (("screening", "screening"), ("kauh", "kauh_infer")):
                out = Path("outputs") / dataset / RUN / name / f"seed{seed}.npz"
                if not export_ok(out, checkpoint_sha, sc):
                    run(["-m", "src.evaluation." + module, *(["infer"] if dataset == "screening" else []),
                         unit, "--ckpt", checkpoint.name, "--out", out, *(["--sc"] if sc else [])])
                    export_ok(out, checkpoint_sha, sc)
        # Isolation contrast: baseline weights with static SC from the corresponding SC run.
        unit, sc_unit = BASE / f"seed{seed}", SC / f"seed{seed}"
        checkpoint = Path(str(unit).replace("outputs/", "checkpoints/", 1)) / "report_epoch_100.pth"
        out = Path("outputs/kauh") / RUN / "baseline_ttsc" / f"seed{seed}.npz"
        if not export_ok(out, sha(checkpoint), True):
            run(["-m", "src.evaluation.kauh_infer", unit, "--ckpt", checkpoint.name, "--out", out,
                 "--sc", "--sc_state", sc_unit / "sc_state.npz", "--sc_args", sc_unit / "train_args.json"])
            export_ok(out, sha(checkpoint), True)
    run(["-m", "src.evaluation.kauh", "--dir", f"outputs/kauh/{RUN}", "--base", "baseline",
         "--arms", "auscult", "baseline_ttsc", "--boot", "1000", "--null_splits", "200"])
    run(["-m", "src.evaluation.screening", "analyse", "--dir", f"outputs/screening/{RUN}",
         "--kauh_dir", f"outputs/kauh/{RUN}", "--pairs", "baseline:baseline", "baseline:baseline_ttsc", "auscult:auscult"])
    run(["-m", "api.bundle", "--out", f"app/bundle-{RUN}"])
    event(status="campaign_complete", run=RUN)


if __name__ == "__main__":
    main()
