"""One-off move of the pre-restructure Stage 1 artefacts into the layout of `src/runs.py` (same filesystem, no copy).

    python scripts/migrate_legacy_layout.py            dry run: print the move list, write outputs/migration_map.json
    python scripts/migrate_legacy_layout.py --apply    execute it (refuses while a training process runs)

    baselines/patchmix_cl/save/icbhi_ast_patchmix_cl_<V>_fold<k>  -> outputs|checkpoints/stage1_v8/legacy-v8/<V>/cv<k>/
    baselines/patchmix_cl/save/icbhi_ast_patchmix_cl_bs8_..._seed<N>_best_param -> .../P0_repro/seed<N>/
    save/results.json                       -> outputs/stage1_v8/legacy-v8/results_legacy.json
    *.pth go to checkpoints/, everything else to outputs/
    data/icbhi_dataset, pretrained_models   -> <DATA>/processed/patchmix_icbhi, <DATA>/models/ast (symlink left behind)
    logs/                                   -> logs/stage1_v8/
    .venv                                   -> .venv-train
    run_repro.sh, run_until_done.sh, scripts/ -> <wrapper>/archive/auscult-trust-phase0/baselines/patchmix_cl/
The v8 runs use an older SC bound and arms (P2, P3) than v9, so they are kept as evidence for report.md, not as v9 results.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import CHECKPOINTS, DATA, OUTPUTS, REPO  # noqa: E402

OLD = REPO / "baselines" / "patchmix_cl"
RUN = "legacy-v8"
ARCHIVE = REPO.parent / "archive" / "auscult-trust-phase0" / "baselines" / "patchmix_cl"


def plan():
    moves = []  # (src, dst, symlink_back)
    for d in sorted((OLD / "save").glob("icbhi_ast_patchmix_cl_*")):
        if (m := re.fullmatch(r"icbhi_ast_patchmix_cl_(.+)_fold(\d+)", d.name)):
            cell, unit = m.group(1), f"cv{m.group(2)}"
        elif (m := re.fullmatch(r"icbhi_ast_patchmix_cl_bs8_lr5e-5_ep50_seed(\d+)_best_param", d.name)):
            cell, unit = "P0_repro", f"seed{m.group(1)}"
        else:
            print("skip (unrecognised):", d)
            continue
        for f in sorted(d.iterdir()):
            kind = CHECKPOINTS if f.suffix == ".pth" else OUTPUTS
            moves.append((f, kind / "stage1_v8" / RUN / cell / unit / f.name, False))
    if (OLD / "save" / "results.json").exists():
        moves.append((OLD / "save" / "results.json", OUTPUTS / "stage1_v8" / RUN / "results_legacy.json", False))
    moves += [(OLD / "data" / "icbhi_dataset", DATA / "processed" / "patchmix_icbhi", True),
              (OLD / "pretrained_models" / "audioset_10_10_0.4593.pth", DATA / "models" / "ast" / "audioset_10_10_0.4593.pth", True)]
    moves += [(f, REPO / "logs" / "stage1_v8" / f.name, False) for f in sorted((OLD / "logs").glob("*"))]
    moves += [(OLD / n, ARCHIVE / n, False) for n in ("run_repro.sh", "run_until_done.sh", "scripts") if (OLD / n).exists()]
    moves.append((OLD / ".venv", REPO / ".venv-train", False))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    moves = [(s, d, b) for s, d, b in plan() if s.exists() or s.is_symlink()]
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "migration_map.json").write_text(json.dumps([[str(s), str(d)] for s, d, _ in moves], indent=1))
    for s, d, _ in moves:
        print(f"{s.relative_to(REPO.parent)} -> {d.relative_to(REPO.parent)}")
    print(f"{len(moves)} moves; map written to outputs/migration_map.json")
    if not a.apply:
        return
    busy = subprocess.run(["pgrep", "-af", "patchmix_cl|run_stage1|run_until_done"], capture_output=True, text=True).stdout
    busy = [line for line in busy.splitlines() if "migrate_legacy_layout" not in line and "pgrep" not in line]
    if busy:
        sys.exit("refusing to move while training runs:\n" + "\n".join(busy))
    for s, d, back in moves:
        assert not d.exists() or d.is_symlink(), f"destination exists: {d}"
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.is_symlink():
            d.unlink()  # the temporary link to the old location
        s.rename(d)
        if back:
            s.symlink_to(d)
    print("done")


if __name__ == "__main__":
    main()
