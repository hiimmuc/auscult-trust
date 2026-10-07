"""Prepare the downloaded datasets for training and evaluation (run `bash scripts/download_data.sh` first). Idempotent.

    python scripts/prepare_data.py

ICBHI: the database ships with 91 recordings named after the wrong stethoscope (`filename_differences.txt`, fixed by the
`script.sh` shipped with the database); they are renamed to `Meditron`. Then `processed/patchmix_icbhi/` is built for
the training code: `audio_test_data/` (symlinks to every wav and txt of the raw database) and `official_split.txt`
(the official 60/40 train/test split, taken from the Patch-Mix CL reference repository).
KAUH: checked only (336 wav = 112 patients x 3 filters).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DATA, REPOS  # noqa: E402

ICBHI = DATA / "raw" / "icbhi" / "ICBHI_final_database"
PROCESSED = DATA / "processed" / "patchmix_icbhi"
SPLIT_SOURCE = REPOS / "patch-mix_contrastive_learning" / "data" / "icbhi_dataset" / "official_split.txt"


def fix_icbhi_names():
    """Rename the recordings listed in filename_differences.txt from `..._AKGC417L` to `..._Meditron`. Returns the count."""
    renamed = 0
    for line in (ICBHI / "filename_differences.txt").read_text().splitlines():
        old = line.strip().strip("'")
        if not old:
            continue
        new = old[: -len("AKGC417L")] + "Meditron"
        for ext in (".txt", ".wav"):
            src, dst = ICBHI / (old + ext), ICBHI / (new + ext)
            if src.exists() and not dst.exists():
                src.rename(dst)
                renamed += 1
    return renamed


def link_icbhi():
    """Symlink every raw wav/txt into `processed/patchmix_icbhi/audio_test_data` and copy the official split."""
    audio = PROCESSED / "audio_test_data"
    audio.mkdir(parents=True, exist_ok=True)
    for f in sorted(ICBHI.glob("*")):
        if f.suffix in (".wav", ".txt") and f.name[0].isdigit():
            link = audio / f.name
            if not link.is_symlink():
                link.symlink_to(f.resolve())
    (PROCESSED / "official_split.txt").write_text(SPLIT_SOURCE.read_text())


def main():
    assert ICBHI.is_dir(), f"{ICBHI} missing: run scripts/download_data.sh"
    assert SPLIT_SOURCE.is_file(), f"{SPLIT_SOURCE} missing: run scripts/download_data.sh"
    print(f"ICBHI: renamed {fix_icbhi_names()} files")
    link_icbhi()
    wavs = len(list(ICBHI.glob("*.wav")))
    split = [line.split("\t")[1] for line in (PROCESSED / "official_split.txt").read_text().splitlines() if line.strip()]
    assert wavs == 920 and len(split) == 920, f"expected 920 recordings, found {wavs} wav and {len(split)} split entries"
    print(f"ICBHI: {wavs} recordings, split train {split.count('train')} / test {split.count('test')} -> {PROCESSED}")
    kauh = len(list((DATA / "raw" / "kauh" / "Audio Files").glob("*.wav")))
    print(f"KAUH: {kauh} wav files" + ("" if kauh == 336 else " (expected 336)"))


if __name__ == "__main__":
    main()
