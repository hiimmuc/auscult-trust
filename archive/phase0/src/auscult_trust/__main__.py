"""AuscultTrust CLI: head variants and partial fine-tuning on cached ViT features (AST, HeAR), selected by grouped CV. Resumable.

Protocol (the official test set is touched only by `final`):
  cv-head   grouped 5-fold CV over the train+val patients for head variants
            (layer mix: last | concat | scalar) x (pooling over real frames: mean | max | meanmax | attn)
  cv-ft     the same CV for the last k ViT blocks fine-tuned from cached tokens (k = 0 is frozen blocks)
  select    mean CV curve per cell -> best epoch and CV Score; writes cv_summary.json
  final     train on all train+val patients for the CV-chosen epochs, evaluate on test once per seed
  probe     L0 rung: logistic regression on the frozen last layer (C by grouped CV), test once, conformal sets
  final-conformal  as `final`, but train on 4/5 of the dev patients and calibrate the conformal layer (V1 split, V2
            Mondrian) on the held-out 1/5 (patients disjoint); reports coverage, deficit, set size on test

Resume: re-run the same command. Finished (cell, fold) results are skipped; an interrupted fine-tuning run
continues from its last finished epoch (`checkpoints/<exp>/<run_id>/<cell>/fold<f>.pt`). Keep RUN_ID fixed.

Usage: RUN_ID=ast-v1 python -m src.auscult_trust <cv-head|cv-ft|select|final|final-conformal|probe> configs/ft_ast.yaml
  [--cells a,b | --cells report | --top N | --pool mean|max|meanmax|attn|auto | --alpha 0.1]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.auscult_trust.conformal import ConformalLayer
from src.auscult_trust.data import FeatureData, folds
from src.auscult_trust.fit import fit, metrics, predict
from src.auscult_trust.model import mode_of
from src.auscult_trust.probe import run_probe
from src.data.splits import load_split
from src.eval.metrics import mean_sd
from src.runlog import ckpt_path, log_run, run_dir, start_summary, write_meta


def cv(cfg, D, cells, exp, n_folds, epochs):
    fs = folds(D, cfg["cv_folds"])[:n_folds]
    for cell in cells:
        for f, (tr, te) in enumerate(fs):
            out = run_dir(exp) / "cv" / cell / f"fold{f}.json"
            if out.exists():
                print(f"skip {cell} fold{f} (done)", flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            ck = run_dir(exp, "checkpoints") / "cv" / cell / f"fold{f}.pt" if mode_of(cell) == "tokens" else None
            print(f"CV {cell} fold {f + 1}/{len(fs)}", flush=True)
            _, curve = fit(cfg, D, cell, tr, {"held_out": te}, epochs, 0, ck)
            out.write_text(json.dumps({"cell": cell, "fold": f, "curve": [c["held_out"] for c in curve]}))
            if ck and Path(ck).exists():
                Path(ck).unlink()  # fold finished; the result file is the record


def select(exp):
    summ = {}
    for d in sorted((run_dir(exp) / "cv").glob("*")):
        runs = [json.loads(f.read_text())["curve"] for f in sorted(d.glob("fold*.json"))]
        if not runs:  # cell directory left by an interrupted first fold
            continue
        sc = np.array([[e["score"] for e in r] for r in runs])  # (folds, epochs)
        sm = np.stack([np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3, mode="valid") for x in sc])  # 3-epoch smoothing
        b = int(sm.mean(0).argmax())
        summ[d.name] = {"best_epoch": b + 1, "cv_score": float(sc[:, b].mean()), "cv_sd": float(sc[:, b].std(ddof=1)) if len(sc) > 1 else 0.0,
                        "f1_crackle": float(np.mean([r[b]["f1_crackle"] for r in runs])), "f1_wheeze": float(np.mean([r[b]["f1_wheeze"] for r in runs])),
                        "n_folds": len(runs)}
    (run_dir(exp) / "cv_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"{'cell':22s} {'epoch':>5s} {'CV Score':>16s} {'crackle F1':>10s} {'wheeze F1':>9s} folds")
    for c, s in sorted(summ.items(), key=lambda x: -x[1]["cv_score"]):
        print(f"{c:22s} {s['best_epoch']:5d} {s['cv_score']:.3f} ± {s['cv_sd']:.3f} {s['f1_crackle']:10.3f} {s['f1_wheeze']:9.3f} {s['n_folds']}")
    return summ


def final(cfg, D, cells, exp, seeds, h):
    summ = json.loads((run_dir(exp) / "cv_summary.json").read_text())
    tr, te = D.dev.row.values, D.test.row.values
    for cell in cells:
        ep = summ[cell]["best_epoch"]
        for s in seeds:
            if (run_dir(exp) / f"final-{cell}" / f"seed{s}.json").exists():
                print(f"skip final {cell} seed{s} (done)", flush=True)
                continue
            ck = run_dir(exp, "checkpoints") / f"final-{cell}" / f"seed{s}.resume.pt"
            model, _ = fit(cfg, D, cell, tr, {}, ep, s, ck)  # epochs fixed by CV; test is evaluated once, after training
            p = predict(model, D, te, mode_of(cell), 128)
            m = metrics(D.test.label.values, p)
            m = {"icbhi_score": m["score"], "recall": m["recall"], "f1": m["f1"], "epochs": ep, "cv_score": summ[cell]["cv_score"],
                 "test_pred": p.argmax(1).tolist()}
            log_run(exp, f"final-{cell}", s, cfg, h, D.encoder, m)
            torch.save({k: v for k, v in model.state_dict().items() if "frozen" not in k}, ckpt_path(exp, f"final-{cell}", s))
            Path(ck).unlink(missing_ok=True)
    print(f"{'cell':22s} {'epochs':>6s} {'test Score':>16s}")
    for cell in cells:
        sc = [json.loads(f.read_text())["metrics"]["icbhi_score"] for f in sorted((run_dir(exp) / f"final-{cell}").glob("seed*.json"))]
        print(f"{cell:22s} {summ[cell]['best_epoch']:6d} {mean_sd(sc)[0]:.3f} ± {mean_sd(sc)[1]:.3f}  (n={len(sc)})")


def final_conformal(cfg, D, cells, exp, seeds, h, alpha):
    """Train on dev minus one patient-grouped calibration fold, calibrate, then evaluate once on test.

    The fold for seed s is `s % cv_folds`. The model sees 4/5 of the dev patients, so its test Score differs
    slightly from `final`. Results go to cell `conf-<cell>`.
    """
    summ = json.loads((run_dir(exp) / "cv_summary.json").read_text())
    fs = folds(D, cfg["cv_folds"])
    te, y_te = D.test.row.values, D.test.label.values
    for cell in cells:
        ep, mode = summ[cell]["best_epoch"], mode_of(cell)
        for s in seeds:
            if (run_dir(exp) / f"conf-{cell}" / f"seed{s}.json").exists():
                print(f"skip conformal {cell} seed{s} (done)", flush=True)
                continue
            tr, cal = fs[s % len(fs)]
            ck = run_dir(exp, "checkpoints") / f"conf-{cell}" / f"seed{s}.resume.pt"
            model, _ = fit(cfg, D, cell, tr, {}, ep, s, ck)
            p_cal, p_te = predict(model, D, cal, mode, 128), predict(model, D, te, mode, 128)
            y_cal = D.y[cal]
            m = {"icbhi_score": metrics(y_te, p_te)["score"], "epochs": ep, "n_cal": len(cal), "n_cal_patients": int(D.dev[D.dev.row.isin(cal)].patient.nunique())}
            for v in ("split", "mondrian"):
                m[v] = ConformalLayer(alpha, v).calibrate(p_cal, y_cal).evaluate(p_te, y_te)
            log_run(exp, f"conf-{cell}", s, cfg, h, D.encoder, m)
            Path(ck).unlink(missing_ok=True)
    print(f"{'cell':22s} {'variant':9s} {'coverage':>16s} {'size':>5s} {'crackle cov':>11s} {'wheeze cov':>10s}")
    for cell in cells:
        runs = [json.loads(f.read_text())["metrics"] for f in sorted((run_dir(exp) / f"conf-{cell}").glob("seed*.json"))]
        for v in ("split", "mondrian"):
            cov = mean_sd([r[v]["coverage"] for r in runs])
            print(f"{cell:22s} {v:9s} {cov[0]:.3f} ± {cov[1]:.3f} {np.mean([r[v]['size'] for r in runs]):5.2f} "
                  f"{np.mean([r[v]['per_class'][1]['coverage'] for r in runs]):11.3f} {np.mean([r[v]['per_class'][2]['coverage'] for r in runs]):10.3f}")


def head_cells(cfg):
    g = cfg["head_grid"]
    return [f"{lm}-{p}" for lm in g["layer_mode"] for p in g["pool"]]


def ft_cells(cfg, pool):
    return [f"ft-k{k}-{pool}" for k in cfg["ft_ks"]] + [f"lora-r{r}-{pool}" for r in cfg.get("lora_ranks", [])]


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    cfg = yaml.safe_load(path.read_text())
    exp = cfg["exp"]

    def arg(name, default=None):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    start_summary(exp)
    write_meta(exp, cfg)
    if cmd == "select":
        select(exp)
        sys.exit()
    D, h = FeatureData(cfg), load_split(cfg["split_file"])[1]
    if cmd == "probe":
        m = run_probe(cfg, D, float(arg("--alpha", 0.1)))
        log_run(exp, "probe-lr", 0, cfg, h, D.encoder, m)
        print(f"probe-lr C={m['C']} test Score {m['icbhi_score']:.3f} CI [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] "
              f"coverage V1 {m['split']['coverage']:.3f} V2 {m['mondrian']['coverage']:.3f}")
    elif cmd == "cv-head":
        cv(cfg, D, (arg("--cells") or ",".join(head_cells(cfg))).split(","), exp, cfg["cv_folds"], cfg["head"]["epochs"])
        select(exp)
    elif cmd == "cv-ft":
        pool = arg("--pool", "auto")
        if pool == "auto":  # pooling of the best head-only cell that uses the last layer, else the config default
            f = run_dir(exp) / "cv_summary.json"
            heads = {c: v for c, v in (json.loads(f.read_text()) if f.exists() else {}).items() if c.startswith("last-")}
            pool = max(heads, key=lambda c: heads[c]["cv_score"]).split("-")[1] if heads else cfg["ft_pool"]
            print(f"fine-tuning pool: {pool}", flush=True)
        cv(cfg, D, (arg("--cells") or ",".join(ft_cells(cfg, pool))).split(","), exp, cfg["ft_cv_folds"], cfg["ft"]["epochs"])
        select(exp)
    elif cmd in ("final", "final-conformal"):
        summ = json.loads((run_dir(exp) / "cv_summary.json").read_text())

        def best(keep):
            return max((c for c in summ if keep(c)), key=lambda c: summ[c]["cv_score"], default=None)

        if arg("--cells") == "report":  # frozen baseline, best head-only cell, best fine-tuned cell (by CV)
            cells = list(dict.fromkeys(c for c in ["last-mean", best(lambda c: mode_of(c) == "frames"), best(lambda c: mode_of(c) == "tokens")] if c in summ))
        else:
            cells = arg("--cells").split(",") if arg("--cells") else sorted(summ, key=lambda c: -summ[c]["cv_score"])[:int(arg("--top", 3))]
        if cmd == "final":
            final(cfg, D, cells, exp, cfg["seeds"], h)
        else:
            final_conformal(cfg, D, cells, exp, cfg["seeds"], h, float(arg("--alpha", 0.1)))
