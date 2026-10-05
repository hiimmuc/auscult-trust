"""Part II runner: coverage and device decodability per ladder rung, on KAUH filter pairs (E2) and ICBHI device-held-out (E1).

Rungs per encoder (from the finished ICBHI ladder `ladder-<enc>-v1`; no epoch or cell is chosen on a shifted target):
  L0  `probe` logistic regression on the frozen clip mean, C from the ladder probe
  L1  best frozen head cell by ladder CV score (mode `frames`)
  L2  best plain `ft-k*` cell by ladder CV score (ViT encoders with a token cache: ast, hear)
Every unit trains a rung on source train patients, calibrates the conformal layer (V1 split, V2 Mondrian) on source
calibration patients, and evaluates on test sets with the same patients; decodability is the AUC of a domain classifier
on the rung's own embedding (source test vs target test, within class).
  E2  KAUH 8 s windows, 5 patient-disjoint 60/20/20 partitions x 6 ordered filter pairs (target = other filter, same test patients)
  E1  ICBHI, one device held out (`device_holdout_official`): train/cal on other devices, test on the held device's official
      test patients (`test_unseen`) and official train patients (`test_seen`, in OPERA pretraining)

Usage: RUN_ID=shift-v1 python -m src.shift_study e1|e2 <ast|hear|clap|opera_ct|opera_ce> [--seeds 3] [--rungs probe,head,ft]
       RUN_ID=shift-v1 python -m src.shift_study table
Resumable: finished units (`outputs/shift_<enc>/<RUN_ID>/<e1|e2>-<rung>-<unit>/seed<N>.json`) are skipped.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.auscult_trust.conformal import ConformalLayer
from src.auscult_trust.data import FeatureData
from src.auscult_trust.fit import fit, metrics, predict
from src.auscult_trust.model import make_model, mode_of
from src.auscult_trust.probe import clip_features, fit_lr
from src.data.icbhi import cycle_table
from src.data.kauh import FILTERS, ORDERED_PAIRS, file_table, partitions, windows
from src.data.splits import device_holdout_official, load_split, split_frames, with_official
from src.encoders.base import DEVICE, N_FRAMES
from src.eval.shift import decodability
from src.features import cache_key
from src.runlog import log_run, run_dir

ALPHA = 0.1


class KauhData(FeatureData):
    """Cached KAUH windows with the `FeatureData` batch interface (every window is a full 8 s: all frames real)."""

    def __init__(self, cfg, root="../data/raw/kauh"):
        self.table = windows(file_table(root))
        out = Path(cfg["kauh_cache"])
        assert json.loads((out / "keys.json").read_text()) == [cache_key(r) for r in self.table.itertuples()], "cache/table mismatch"
        assert np.load(out / "done.npy").all()
        self.y = self.table.label.to_numpy()
        self.layers = torch.from_numpy(np.load(out / "layers.npy"))
        self.tokens = np.load(out / "tokens.npy", mmap_mode="r") if (out / "tokens.npy").exists() else None
        self.mask = torch.ones(len(self.table), N_FRAMES, dtype=torch.bool)
        meta = json.loads((out / "meta.json").read_text())
        self.encoder, self.token_layer = meta["encoder"], meta["token_layer"]


def ladder_rungs(enc):
    """Rung definitions {name: {cell, epochs, C}} read from the finished ICBHI ladder of `enc`."""
    d = Path(f"outputs/ladder_{enc}/ladder-{enc}-v1")
    summ = json.loads((d / "cv_summary.json").read_text())

    def best(keep):
        c = max((c for c in summ if keep(c)), key=lambda c: summ[c]["cv_score"], default=None)
        return None if c is None else {"cell": c, "epochs": summ[c]["best_epoch"]}
    rungs = {"probe": {"cell": "probe", "C": json.loads((d / "probe-lr/seed0.json").read_text())["metrics"]["C"]},
             "head": best(lambda c: mode_of(c) == "frames"), "ft": best(lambda c: c.startswith("ft-") and "+" not in c)}
    return {k: v for k, v in rungs.items() if v}


@torch.no_grad()
def embed(model, D, rows, mode):
    model.eval()
    out = []
    for i in range(0, len(rows), 128):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=mode == "tokens"):
            out.append(model.embed(D.batch(rows[i:i + 128], mode)).float().cpu().numpy())
    return np.concatenate(out)


def train_rung(cfg, D, rung, tr, sets, seed, epoch_scale=1.0):
    """Train a rung on rows `tr`; return {set name: (probs, embedding)} for the named row sets."""
    if rung["cell"] == "probe":
        X = clip_features(D)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        f = fit_lr(X[tr], D.y[tr], rung["C"])
        return {n: (f(X[r]), (X[r] - mu) / sd) for n, r in sets.items()}
    cell, mode = rung["cell"], mode_of(rung["cell"])
    model, _ = fit(cfg, D, cell, tr, {}, max(1, round(rung["epochs"] * epoch_scale)), seed)
    return {n: (predict(model, D, r, mode, 128), embed(model, D, r, mode)) for n, r in sets.items()}


def evaluate(out, D, sets, src_cal, targets, ref_src=None, groups=None):
    """Metrics of one unit. `src_cal` calibrates; each name in `targets` is scored; decodability vs `ref_src` if given."""
    p_cal = out[src_cal][0]
    res = {"n_cal": int(len(sets[src_cal]))}
    for name in targets:
        p, z = out[name]
        y = D.y[sets[name]]
        m = {"n": int(len(y)), "icbhi_score": metrics(y, p)["score"], "recall": metrics(y, p)["recall"]}
        for v in ("split", "mondrian"):
            m[v] = ConformalLayer(ALPHA, v).calibrate(p_cal, D.y[sets[src_cal]]).evaluate(p, y)
        if ref_src and name != ref_src:
            try:
                m["decodability_auc"] = decodability(out[ref_src][1], z, groups[ref_src], groups[name], y_src=D.y[sets[ref_src]], y_tgt=y)
            except ValueError:  # degenerate folds on tiny sets (Litt3200 test_seen, 28 cycles): no AUC
                pass
        res[name] = m
    return res


def run_e2(enc, cfg, seeds_ft):
    D = KauhData(cfg)
    t, h = D.table, hashlib.sha256(b"kauh-partitions-5").hexdigest()
    n_dev = sum(len(v) for k, v in split_frames(cycle_table(cfg["icbhi_root"]), load_split(cfg["split_file"])[0]).items() if k in ("train", "val"))
    exp = f"shift_{enc}"
    for rname, rung in cfg["rungs"].items():
        for i, part in enumerate(partitions(t.drop_duplicates("patient"))):
            for src, tgt in ORDERED_PAIRS:
                cell = f"e2-{rname}-{src}2{tgt}"
                if (run_dir(exp) / cell / f"seed{i}.json").exists():
                    continue
                rows = lambda pats, f: np.flatnonzero((t.patient.isin(pats) & (t["filter"] == f)).values)
                sets = {"train": rows(part["train"], src), "cal": rows(part["cal"], src), "test_src": rows(part["test"], src), "test_tgt": rows(part["test"], tgt)}
                if min(len(v) for v in sets.values()) == 0:
                    continue
                # ponytail: steps-matched epochs (KAUH train set is ~40x smaller than ICBHI dev), no epoch selection on target
                out = train_rung(cfg, D, rung, sets["train"], sets, i, n_dev / len(sets["train"]))
                groups = {k: t.patient.values[v] for k, v in sets.items()}
                log_run(exp, cell, i, cfg, h, D.encoder, evaluate(out, D, sets, "cal", ["test_src", "test_tgt"], "test_src", groups))
                print(f"{cell} partition {i} done", flush=True)


def run_e1(enc, cfg, seeds_ft):
    D = FeatureData(cfg)
    df = with_official(cycle_table(cfg["icbhi_root"]), load_split(cfg["split_file"])[0])
    keys = [cache_key(r) for r in cycle_table(cfg["icbhi_root"]).itertuples()]
    row_of = {k: i for i, k in enumerate(keys)}
    df["row"] = [row_of[cache_key(r)] for r in df.itertuples()]
    h, exp = load_split(cfg["split_file"])[1], f"shift_{enc}"
    for rname, rung in cfg["rungs"].items():
        for dev in sorted(df.device.unique()):
            part = device_holdout_official(df, dev)
            if len(part["test_unseen"]) == 0:
                continue  # LittC2SE: no official-test recordings
            sets = {"train": part["train"].row.values, "cal": part["cal"].row.values, "test_unseen": part["test_unseen"].row.values,
                    "test_seen": part["test_seen"].row.values}
            groups = {k: part[k].patient.values for k in sets}
            sets = {k: v for k, v in sets.items() if len(v)}
            for s in range(1 if rname == "probe" else seeds_ft):
                cell = f"e1-{rname}-{dev}"
                if (run_dir(exp) / cell / f"seed{s}.json").exists():
                    continue
                out = train_rung(cfg, D, rung, sets["train"], sets, s)
                # decodability: held-out device vs the other devices' official test (the calibration set)
                m = evaluate(out, D, sets, "cal", [k for k in sets if k.startswith("test")], "cal", groups)
                log_run(exp, cell, s, cfg, h, D.encoder, {**m, "n_dropped": part["n_dropped"], "n_train": len(sets["train"])})
                print(f"{cell} seed{s} done", flush=True)


def table():
    """Mean over partitions/seeds per (experiment, rung, unit, test set): Score, V1/V2 coverage and deficit, decodability."""
    import glob
    from collections import defaultdict
    acc = defaultdict(list)
    for f in sorted(glob.glob(f"outputs/shift_*/{os.environ['RUN_ID']}/e*/seed*.json")):
        enc = f.split("/")[1].removeprefix("shift_")
        r = json.loads(Path(f).read_text())
        for name, m in r["metrics"].items():
            if isinstance(m, dict) and "split" in m:
                acc[(enc, r["cell"], name)].append(m)
    print(f"{'encoder':10s} {'cell':28s} {'test set':11s} {'n':>4s} {'Score':>6s} {'V1 cov':>7s} {'V1 def':>7s} {'V2 cov':>7s} {'AUC':>5s} runs")
    for (enc, cell, name), ms in sorted(acc.items()):
        mean = lambda g: np.nanmean([g(m) for m in ms])
        auc = [m["decodability_auc"] for m in ms if "decodability_auc" in m]
        print(f"{enc:10s} {cell:28s} {name:11s} {mean(lambda m: m['n']):4.0f} {mean(lambda m: m['icbhi_score']):6.3f} "
              f"{mean(lambda m: m['split']['coverage']):7.3f} {mean(lambda m: m['split']['deficit']):7.3f} "
              f"{mean(lambda m: m['mondrian']['coverage']):7.3f} {np.mean(auc) if auc else float('nan'):5.2f} {len(ms)}")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
        sys.exit()
    exp_kind, enc = sys.argv[1], sys.argv[2]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    cfg = yaml.safe_load(Path(f"configs/ladder_{enc}.yaml").read_text())
    cfg["kauh_cache"] = f"../data/cache/{yaml.safe_load(Path(f'configs/extract_{enc}_kauh.yaml').read_text())['name']}"
    allr = ladder_rungs(enc)
    cfg["rungs"] = {k: allr[k] for k in arg("--rungs", "probe,head,ft").split(",") if k in allr}
    (run_e1 if exp_kind == "e1" else run_e2)(enc, cfg, int(arg("--seeds", 3)))
