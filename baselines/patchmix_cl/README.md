# Patch-Mix CL reproduction

Working copy of `../../repos/patch-mix_contrastive_learning` (reference repo, read-only). Changes vs upstream: `repro_changes.patch`.
Run from this directory. Setup (once): `uv venv --python /usr/bin/python3.10 .venv`, then install torch 2.9.1+cu128, `timm==0.4.5`, librosa pandas tqdm wget scikit-learn soundfile audiomentations nlpaug cmapy opencv-python-headless matplotlib numpy (see `../../auscult-trust/report.md` section 8).
Run: `setsid nohup ./run_until_done.sh <seed> &` (resumes from `save/<tag>/last.pth`). Summary: `.venv/bin/python summarize_repro.py`.
