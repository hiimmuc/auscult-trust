"""Build the on-disk spectrogram cache for the given encoders once, so that training jobs start in seconds and share it.

    python -m src.training.patchmix_cl.build_cache ast htsat opera_ct clap hear

Run it before a sweep; jobs without a cache still work (they build and save it themselves, but then every parallel job repeats
the same CPU-heavy extraction). The cache is skipped for runs with spectrum correction, which depend on the training devices.
"""

import sys

from .config import parse_args
from .dataset import ICBHIDataset
from .encoders import NAMES


def main():
    names = sys.argv[1:] or NAMES
    for name in names:
        # --freeze_encoder only lifts the frozen-only check; the cache does not depend on it
        args = parse_args(
            ["--config", "experiments/train/base.yaml", "--encoder", name, "--freeze_encoder"]
        )
        for train in (True, False):
            ds = ICBHIDataset(train, None, args)
            print(name, "train" if train else "test", ds.images.shape, flush=True)


if __name__ == "__main__":
    main()
