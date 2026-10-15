"""Method x encoder grid on the ICBHI official split: cell list, smoke test, job lists for two machines.

    python scripts/grid.py cells                       # print the cells
    python scripts/grid.py smoke [--only ast] [--data] # a few synthetic training steps per cell: errors, peak GPU memory, step time
    python scripts/grid.py jobs --smoke a.json b.json  # logs/campaigns/grid-v1-{A,B}.txt, balanced by measured step time
    python scripts/grid.py table [--epoch last|best_test]  # results table -> outputs/train/grid-v1/table_<epoch>.md

Cell = `<encoder>+<ft|frozen>+<method>`, run `grid-v1`, seeds 0-4, `--selection fixed --report_epochs <last epoch>`
(per-epoch test predictions are kept, so any epoch rule can be applied afterwards). Each method keeps the settings of its
reference training script (experiments/train/grid/<method>.yaml); the encoder file only sets the encoder.
`smoke --data` also builds the real data path of the cells whose data differs (QLung quality cache, 5 s cycles, an 80/20 fold).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUN = "grid-v1"
ENCODERS = ["ast", "htsat", "opera_ct", "beats", "dass"]
FT = ["ce", "patchmix", "patchmix_cl", "pafa", "sg_scl", "qlung", "add_rsc", "ce_pafa_cfg"]
FROZEN = ["ce", "patchmix", "pafa", "qlung", "add_rsc", "ce_pafa_cfg"]  # loss-only methods on a frozen encoder train only the projector
ENCODER_ORDER = ["ast", "beats", "htsat", "opera_ct", "dass"]  # job order: DASS (slowest) last
FIXED_LENGTH = {"htsat", "opera_ct", "dass"}  # input length fixed by the encoder: PAFA's 5 s cycles do not apply
TRAIN_CYCLES, TEST_CYCLES = 4142, 2756
PY = ".venv-train/bin/python"
DASS_PY = "LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$HOME/miniconda3/envs/cuda128/lib .venv-dass/bin/python"


def cells():
    """List of dicts: cell, encoder, mode, method, config files, extra command-line arguments."""
    out = []
    for enc in ENCODERS:
        for mode, methods in (("ft", FT), ("frozen", FROZEN)):
            for m in methods:
                f = m + ("_beats" if enc == "beats" and m in ("pafa", "ce_pafa_cfg") else "")
                cfg = ["experiments/train/base.yaml", f"experiments/train/encoder_{enc}.yaml", f"experiments/train/grid/{f}.yaml"]
                cfg += ["experiments/train/freeze_encoder.yaml"] if mode == "frozen" else []
                extra = ["--desired_length", "8"] if enc in FIXED_LENGTH and m in ("pafa", "ce_pafa_cfg") else []
                out.append(dict(cell=f"{enc}+{mode}+{m}", encoder=enc, mode=mode, method=m, config=cfg, extra=extra))
    return out


def parse(c, *more):
    from src.training.patchmix_cl.config import parse_args
    return parse_args(["--config", ",".join(c["config"]), "--cell", c["cell"], *c["extra"], *more])


def smoke_cell(c, steps):
    """Build the cell's model and run `steps` training steps and one evaluation batch on random images of the right shape."""
    import numpy as np
    import torch
    from src.training.patchmix_cl.encoders import get_encoder, image_shape
    from src.training.patchmix_cl.engine import evaluate, train_epoch
    from src.training.patchmix_cl.modeling import build_model

    args = parse(c, "--selection", "fixed", "--report_epochs", "1")
    args.class_nums = np.array([2063, 1215, 501, 363])
    t, f = image_shape(get_encoder(args.encoder), args)
    bs = args.batch_size
    batch = lambda: (torch.randn(bs, 1, t, f), torch.randint(0, 4, (bs,)), torch.randint(0, 4, (bs,)),
                     torch.randint(101, 110, (bs,)), torch.rand(bs))
    torch.cuda.reset_peak_memory_stats()
    model, classifier, projector, criterion, optimizer = build_model(args)
    scaler = torch.amp.GradScaler("cuda")
    train_epoch([batch()], model, classifier, projector, criterion, optimizer, 1, args, scaler)  # warm-up (cudnn, allocator)
    torch.cuda.synchronize()
    t0 = time.time()
    loss, _ = train_epoch([batch() for _ in range(steps)], model, classifier, projector, criterion, optimizer, 1, args, scaler)
    torch.cuda.synchronize()
    step = (time.time() - t0) / steps
    t0 = time.time()
    evaluate([batch() for _ in range(3)], model, classifier, args)
    torch.cuda.synchronize()
    ev = (time.time() - t0) / 3
    assert np.isfinite(loss), "loss {}".format(loss)
    epoch_min = (TRAIN_CYCLES // bs * step + TEST_CYCLES / bs * ev) / 60
    return dict(peak_gb=torch.cuda.max_memory_allocated() / 2**30, step_s=step, epoch_min=epoch_min,
                run_h=epoch_min * args.epochs / 60, epochs=args.epochs, loss=loss)


def smoke_data():
    """Real data path of the cells whose data differs from the cached default: QLung quality, 5 s AST/BEATs, 80/20 fold."""
    from src.training.patchmix_cl.dataset import build_loaders
    cs = {c["cell"]: c for c in cells()}
    for name, more in (("ast+ft+qlung", []), ("ast+ft+pafa", []), ("beats+ft+pafa", []), ("ast+ft+sg_scl", []),
                       ("ast+ft+ce", ["--test_fold", "0"])):
        args = parse(cs[name], "--selection", "fixed", "--report_epochs", "1", "--num_workers", "0", *more)
        tr, _, te, train_ds, test_ds, _, _ = build_loaders(args)
        images, labels, devices, patients, quality = next(iter(tr))
        print(name, *more, "train/test cycles", len(train_ds), len(test_ds), "image", tuple(images.shape),
              "quality mean {:.3f}".format(float(train_ds.quality.mean())), flush=True)
        if more:  # 80/20: patients disjoint, every patient in one of the two sides
            assert not set(train_ds.patients) & set(test_ds.patients)
            assert len(set(train_ds.patients) | set(test_ds.patients)) == 126


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    s = sub.add_parser("smoke")
    s.add_argument("--only", nargs="*", default=None, help="encoders to test (default all)")
    s.add_argument("--steps", type=int, default=6)
    s.add_argument("--data", action="store_true")
    s.add_argument("--out", default="logs/grid_smoke.json")
    s.add_argument("--one", default=None, help=argparse.SUPPRESS)  # internal: one cell in a fresh process
    j = sub.add_parser("jobs")
    j.add_argument("--smoke", nargs="+", required=True, help="smoke json per machine, in the order A B")
    j.add_argument("--seeds", type=int, default=5)
    j.add_argument("--pin", default=None, help="file of '<machine A|B> <cell> <seed>' lines: units already started stay on their machine")
    t = sub.add_parser("table")
    t.add_argument("--epoch", default="last", help="last, best_test (optimistic: best Score on test per seed) or an epoch number")
    a = ap.parse_args()

    if a.cmd == "cells":
        for c in cells():
            print(c["cell"], " ".join(c["config"][1:]), " ".join(c["extra"]))
    elif a.cmd == "smoke" and a.one:
        c = {c["cell"]: c for c in cells()}[a.one]
        print("RESULT " + json.dumps(smoke_cell(c, a.steps)))
    elif a.cmd == "smoke":
        if a.data:
            smoke_data()
        out = Path(a.out)
        res = json.loads(out.read_text()) if out.exists() else {}
        for c in cells():
            if a.only and c["encoder"] not in a.only or res.get(c["cell"], {}).get("ok"):
                continue
            py = DASS_PY if c["encoder"] == "dass" else PY
            p = subprocess.run(f"PYTHONPATH=. MPLBACKEND=Agg {py} scripts/grid.py smoke --one {c['cell']} --steps {a.steps}",
                               shell=True, capture_output=True, text=True, cwd=REPO)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            res[c["cell"]] = {"ok": True, **json.loads(line[-1][7:])} if line else {"ok": False, "error": p.stderr.strip().splitlines()[-3:]}
            r = res[c["cell"]]
            print("{:28s} {}".format(c["cell"], "peak {peak_gb:.1f} GB, step {step_s:.3f} s, run {run_h:.2f} h".format(**r) if r["ok"]
                                     else "FAILED " + " | ".join(r["error"])), flush=True)
            out.write_text(json.dumps(res, indent=1))
        print("{} ok, {} failed".format(sum(r["ok"] for r in res.values()), sum(not r["ok"] for r in res.values())))
    elif a.cmd == "jobs":
        write_jobs(a)
    else:
        table(a.epoch)


def write_jobs(a):
    """Two job lists with about equal measured hours.

    Order on both machines: the reference cells of each method (diagonal) first, then encoder by encoder with DASS last
    (its fine-tuning runs take 2.5-3.5 h, the others 0.1-1.5 h), longest jobs first within an encoder.
    """
    smokes = [json.loads(Path(p).read_text()) for p in a.smoke]
    diagonal = {"ast+ft+patchmix_cl", "ast+ft+patchmix", "ast+ft+ce", "ast+ft+pafa", "ast+ft+ce_pafa_cfg", "ast+ft+sg_scl",
                "ast+ft+qlung", "ast+ft+add_rsc", "beats+ft+pafa", "beats+ft+ce_pafa_cfg"}
    jobs = []
    for c in cells():
        epochs = smokes[0][c["cell"]]["epochs"]
        py = DASS_PY if c["encoder"] == "dass" else PY
        for s in range(a.seeds):
            line = "RUN_ID={} MEM_GB={:.0f} {} -m src.training.patchmix_cl.main --config {} --cell {} {}--selection fixed --report_epochs {} --seed {}".format(
                RUN, max(4, smokes[0][c["cell"]]["peak_gb"] * 1.25 + 1.5), py, ",".join(c["config"]), c["cell"],
                "".join(x + " " for x in c["extra"]), epochs, s)
            rank = 0 if c["cell"] in diagonal else 1 + ENCODER_ORDER.index(c["encoder"])
            jobs.append((rank, [m[c["cell"]]["run_h"] for m in smokes], line, (c["cell"], s)))
    pins = {}
    for l in Path(a.pin).read_text().splitlines() if a.pin else []:
        m, cell, seed = l.split()
        pins[(cell, int(seed))] = "AB".index(m)
    load, lists = [0.0] * len(smokes), [[] for _ in smokes]
    for rank, hours, line, unit in sorted(jobs, key=lambda j: (j[0], -max(j[1]))):
        k = pins.get(unit, min(range(len(smokes)), key=lambda i: load[i] + hours[i]))
        load[k] += hours[k]
        lists[k].append(line)
    for name, lines, h in zip("AB", lists, load):
        p = REPO / "logs" / "campaigns" / f"{RUN}-{name}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        print(f"{p}: {len(lines)} jobs, {h:.0f} GPU-hours sequential (divide by the parallel workers that fit)")


if __name__ == "__main__":
    main()


def unit_metrics(npz, epoch, quality=None, alpha=0.1, n_splits=20):
    """Full metrics of one seed at one epoch: per-device report, plus split-conformal (LAC) over test-patient halves.

    Conformal: 20 device-stratified calibration/evaluation splits of the test patients; coverage, worst-device coverage, mean
    set size; with `quality` (QLung score per test cycle) also the coverage in the low / mid / high quality tercile.
    """
    import numpy as np
    import pandas as pd
    from src.evaluation.conformal import lac_scores, split_threshold
    from src.processing.icbhi import DEVICES
    from src.processing.splits import calibration_eval_splits
    from src.training.patchmix_cl.reporting import per_device_report

    d = np.load(npz)
    epochs, y, dev, pat = d["epochs"], d["labels"], d["device"], d["patient"]
    if epoch == "best_test":
        scores = [per_device_report(y, p, dev)["all"]["score"] for p in d["preds"]]
        i = int(np.argmax(scores))
    else:
        i = len(epochs) - 1 if epoch == "last" else int(np.where(epochs == int(epoch))[0][0])
    rep = per_device_report(y, d["preds"][i], dev)
    probs = d["probs"][i].astype(np.float64)
    names = np.array(DEVICES)[dev]
    cov, worst, size, qcov = [], [], [], []
    edges = np.quantile(quality, [1 / 3, 2 / 3]) if quality is not None else None
    for sp in calibration_eval_splits(pd.DataFrame({"patient": pat, "device": names}), n_splits):
        cal = np.isin(pat, sp["calibration"])
        ev = np.isin(pat, sp["evaluation"]) & ~cal
        q = split_threshold(lac_scores(probs[cal], y[cal]), alpha)
        sets = probs[ev] >= 1 - q
        hit = sets[np.arange(ev.sum()), y[ev]]
        cov.append(hit.mean())
        worst.append(min(hit[names[ev] == n].mean() for n in set(names[ev])))
        size.append(sets.sum(1).mean())
        if edges is not None:
            b = np.digitize(quality[ev], edges)
            qcov.append([hit[b == k].mean() for k in range(3)])
    out = {k: v for k, v in rep["all"].items() if k in ("sp", "se", "score", "hs", "two_cls_se", "two_cls_score", "two_cls_hs")}
    w = min((r["score"], n) for n, r in rep.items() if n != "all")
    out.update(worst_dev_score=w[0], worst_dev=w[1], epoch=int(epochs[i]), cov=100 * np.mean(cov),
               worst_dev_cov=100 * np.mean(worst), set_size=float(np.mean(size)))
    if qcov:
        out.update({f"cov_q{k}": 100 * float(np.mean([c[k] for c in qcov])) for k in range(3)})
    return out


def table(epoch):
    """Markdown table over the grid cells (mean +- SD over seeds); missing cells are listed as pending."""
    import numpy as np
    from src.paths import DATA

    root = REPO / "outputs" / "train" / RUN
    qpath = DATA / "cache" / "train_images" / "quality_test_4cls_8s_repeat_16000hz_0.7_0.3.npy"
    quality8 = np.load(qpath) if qpath.exists() else None
    cols = [("sp", "Sp"), ("se", "Se"), ("score", "Score"), ("hs", "HS"), ("two_cls_se", "2c Se"), ("two_cls_score", "2c Score"),
            ("two_cls_hs", "2c HS"), ("worst_dev_score", "worst-dev Score"), ("cov", "Cov"), ("worst_dev_cov", "worst-dev Cov"),
            ("set_size", "set size"), ("cov_q0", "Cov low-Q"), ("cov_q1", "Cov mid-Q"), ("cov_q2", "Cov high-Q")]
    lines = [f"# {RUN}: ICBHI official split, epoch `{epoch}`, mean +- SD over seeds", "",
             "Sp/Se/Score/HS in %, 4-class; `2c` = 2-class view of the same predictions (2-class Sp = 4-class Sp). Cov = split-conformal "
             "LAC coverage at 90% over 20 device-stratified calibration/evaluation halves of the test patients; Q terciles = QLung quality "
             "score of the 8 s cycle.", "",
             "| cell | n | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for c in cells():
        units = sorted((root / c["cell"]).glob("seed*/test_preds.npz"))
        units = [u for u in units if (u.parent / "report.json").exists()]
        if not units:
            lines.append(f"| {c['cell']} | 0 | " + " | ".join("pending" for _ in cols) + " |")
            continue
        n_test = len(np.load(units[0])["labels"])
        q = quality8 if quality8 is not None and len(quality8) == n_test else None
        rs = [unit_metrics(u, epoch, q) for u in units]
        cell = lambda k: "-" if k not in rs[0] else "{:.{p}f} +- {:.{p}f}".format(
            np.mean([r[k] for r in rs]), np.std([r[k] for r in rs], ddof=1) if len(rs) > 1 else 0, p=2 if k == "set_size" else 1)
        lines.append(f"| {c['cell']} | {len(rs)} | " + " | ".join(cell(k) for k, _ in cols) + " |")
    out = root / f"table_{epoch}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
