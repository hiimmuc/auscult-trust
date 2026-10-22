"""Assemble what the Hugging Face Space needs under build/ (nothing is uploaded).

    python scripts/stage_space.py

build/space/    public Space repo: Dockerfile, requirements, src/, api/, and the reference code the model builds from.
build/weights/  private model repo: checkpoints, BEATs pretrained weights, SC state and a manifest with relative paths.
                Its tree mirrors the container's /app, so `snapshot_download(local_dir="/app")` puts every file in place.
Demo sample WAVs (ICBHI recordings) stay out.
"""

import json
import os
import shutil
from pathlib import Path

from src.paths import DATA, REPO, REPOS

REFERENCE_REPOS = ["PAFA", "ADD-RSC", "patch-mix_contrastive_learning", "QLung", "SG-SCL"]
BEATS = "models/beats/BEATs_iter3_plus_AS2M.pt"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
API_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "requirements*.txt", "bundle.py")  # bundle.py needs the training outputs


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    out = REPO / "build"
    space, weights = out / "space", out / "weights"
    shutil.rmtree(out, ignore_errors=True)

    space.mkdir(parents=True)
    for f in ("Dockerfile", "README.md", "requirements.txt", "entrypoint.sh"):
        shutil.copy2(REPO / "app/deploy/space" / f, space / f)
    shutil.copytree(REPO / "src", space / "auscult-trust/src", ignore=IGNORE)
    shutil.copytree(REPO / "api", space / "auscult-trust/api", ignore=API_IGNORE)
    for r in REFERENCE_REPOS:
        shutil.copytree(REPOS / r, space / "repos" / r, ignore=IGNORE)

    manifest = json.loads((REPO / "app/bundle/manifest.json").read_text())
    root = weights / "auscult-trust"
    for p in manifest["pipelines"].values():
        ckpt = Path(p["checkpoint"]).relative_to(REPO)
        link(REPO / ckpt, root / ckpt)
        link(REPO / p["unit"] / "train_args.json", root / p["unit"] / "train_args.json")
        p["checkpoint"] = str(ckpt)
        if "sc" in p:
            link(REPO / "app/bundle" / p["sc"]["state"], root / "app/bundle" / p["sc"]["state"])
    manifest["samples"] = []
    (root / "app/bundle").mkdir(parents=True, exist_ok=True)
    (root / "app/bundle/manifest.json").write_text(json.dumps(manifest, indent=1))
    link(DATA / BEATS, weights / "data" / BEATS)
    print(out, "ready: push space/ to a Docker Space and weights/ to a private model repo")


if __name__ == "__main__":
    main()
