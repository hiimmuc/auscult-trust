"""Registered stop rule of sc_cons on the 5-fold screen (docs/prereg-v9-amendment-2.md, section 3.1).

Stop sc_cons if its mean Se over epochs 10-20 is more than 3 points below sc_gain_v2 (same bound as chosen in screen-ast-c0),
or its worst-device mean Score over the same epochs is more than 3 points below. Prints PASS or STOP and the numbers.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/train/screen-ast-cv5")
GAIN = json.load(open("outputs/train/screen-ast-c0/screen.json"))["arms"]["sc_gain"]["cell"].replace("sc_gain", "sc_gain_v2")


def stats(cell, folds=5, epochs=range(10, 21)):
    reps = [json.loads((ROOT / cell / "cv{}".format(k) / "report.json").read_text()) for k in range(folds)]
    hist = [{h["epoch"]: h for h in r["history"]} for r in reps]
    se = np.mean([h[e]["val"]["se"] for h in hist for e in epochs])
    devs = sorted({d for h in hist for e in epochs for d in h[e]["val_devices"]})
    by_dev = {d: np.mean([h[e]["val_devices"][d]["score"] for h in hist for e in epochs if d in h[e]["val_devices"]]) for d in devs}
    return se, min(by_dev.values()), by_dev


cons, gain = stats("sc_cons"), stats(GAIN)
print("sc_cons: mean Se {:.2f}, worst-device Score {:.2f} {}".format(cons[0], cons[1], {k: round(v, 1) for k, v in cons[2].items()}))
print("{}: mean Se {:.2f}, worst-device Score {:.2f} {}".format(GAIN, gain[0], gain[1], {k: round(v, 1) for k, v in gain[2].items()}))
stop = cons[0] < gain[0] - 3 or cons[1] < gain[1] - 3
print("STOP sc_cons (registered rule)" if stop else "PASS sc_cons continues to refit and LODO")
