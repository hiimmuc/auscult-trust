"""Table of the target-sample sweep: held-out coefficient from N held-out patients against the full estimate.

    python scripts/target_n_table.py [--run lodo-ast]

Reads outputs/train/<run>/target_n/<cell>/seed<s>.json. Per fold and N: draws are averaged within a seed, then mean and SD
over seeds of |Delta| (prior-matched), Score and coverage with the N-patient coefficient and with the full coefficient on the
same evaluation patients, and the RMS dB distance of the coefficient to the full estimate.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="lodo-ast")
a = ap.parse_args()
root = Path("outputs/train") / a.run / "target_n"
L = ["| Cell | N | Seeds | Draws/seed | Eval patients | Coef RMS dB to full | abs Delta, N | abs Delta, full | Score, N | Score, full | Coverage, N | Coverage, full |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for cell in sorted(p.name for p in root.iterdir() if p.is_dir()):
    by_n = defaultdict(list)  # N token -> per-seed means
    for f in sorted((root / cell).glob("seed*.json")):
        rows = json.loads(f.read_text())["rows"]
        for tok in dict.fromkeys(r["N_token"] for r in rows):
            rs = [r for r in rows if r["N_token"] == tok]
            by_n[tok].append({"draws": len(rs), "eval": np.mean([r["eval_patients"] for r in rs]),
                              "coef": np.mean([r["coef_db_rms_to_full"] for r in rs]),
                              **{"{}_{}".format(k, w): np.mean([abs(r[w][k]) if k == "delta" else r[w][k] for r in rs])
                                 for k in ("delta", "score", "coverage") for w in ("n", "full")}})
    for tok, seeds in by_n.items():
        m = {k: (float(np.mean([s[k] for s in seeds])), float(np.std([s[k] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0)
             for k in seeds[0]}
        res["{}|{}".format(cell, tok)] = m
        f = lambda k, d=3: "{:.{d}f} ± {:.{d}f}".format(*m[k], d=d)
        L.append("| {} | {} | {} | {:.0f} | {:.1f} | {} | {} | {} | {} | {} | {} | {} |".format(
            cell, tok, len(seeds), m["draws"][0], m["eval"][0], f("coef", 2), f("delta_n"), f("delta_full"),
            f("score_n", 2), f("score_full", 2), f("coverage_n"), f("coverage_full")))
md = "\n".join(L) + "\n"
(root / "target_n_summary.md").write_text(md)
(root / "target_n_summary.json").write_text(json.dumps(res, indent=1))
print(md)
