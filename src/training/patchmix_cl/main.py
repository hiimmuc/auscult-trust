"""Train the lung-sound classifier: Patch-Mix CL on ICBHI with optional device-shift variants.

    python -m src.training.patchmix_cl.main --config experiments/train/base.yaml,experiments/train/sc.yaml \
        --cv_folds 3 --cv_fold 0 --seed 0

Writes `outputs/<exp>/<run_id>/<cell>/<unit>/` (train_args.json, report.json, test_preds.npz, sc_state.npz) and
`checkpoints/<exp>/<run_id>/<cell>/<unit>/` (last.pth while running; report_epoch_<E>.pth for a fixed refit, best.pth for a
cv/test-selected full run, nothing for a screening fold).
A finished unit is skipped; an interrupted one resumes from `last.pth` when the same RUN_ID is set.
"""

import hashlib
import json
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from src import runs

from .config import parse_args
from .dataset import build_loaders
from .engine import evaluate, train_epoch
from .modeling import build_model
from .optim import adjust_learning_rate
from .reporting import pick_epochs

warnings.filterwarnings("ignore")
_VOLATILE = {"num_workers", "print_freq"}


def save_model(model, args, epoch, path, classifier):
    """Weights for inference: classifier always, encoder only when it was trained (a frozen one is the pretrained file)."""
    torch.save(
        {
            "args": vars(args),
            "model": None if args.freeze_encoder else model.state_dict(),
            "epoch": epoch,
            "classifier": classifier.state_dict(),
        },
        path,
    )


def check_same_config(path, args):
    """Refuse to continue a unit whose saved config differs from this run's (no silent mixing of runs)."""
    new = json.loads(json.dumps(vars(args), default=str))
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k for k in new if k not in _VOLATILE and old.get(k) != new[k]}
        assert not diff, "{} exists with a different config ({}); use a new RUN_ID".format(
            path.parent, sorted(diff)
        )
    path.write_text(json.dumps(new, indent=2))


def save_sc_state(out_dir, train_dataset, test_dataset, args):
    """Save s_ref, the per-device spectra and SC coefficients (train and test) and the SC hash; returns the hash."""
    state = {"sc_mode": args.sc_mode, "reference": train_dataset.sc["reference"]}
    for name, ds in (("train", train_dataset), ("test", test_dataset)):
        for d, v in ds.sc["device_spectra"].items():
            state["{}_spectrum_{}".format(name, d)] = v
        for d, v in ds.sc["coefficients"].items():
            state["{}_coefficients_{}".format(name, d)] = v
    np.savez_compressed(out_dir / "sc_state.npz", **state)
    return hashlib.sha256(np.ascontiguousarray(state["reference"]).tobytes()).hexdigest()


def history_json(history):
    score = lambda r: None if r is None else r["all"]["score"]
    return [
        {
            "epoch": h["epoch"],
            "val_score": score(h["val"]),
            "test_score": score(h["test"]),
            "val_devices": (
                None
                if h["val"] is None
                else {k: v["score"] for k, v in h["val"].items() if k != "all"}
            ),
        }
        for h in history
    ]


def main():
    args = parse_args()
    out_dir, ckpt_dir = (
        runs.unit_dir(args.exp, args.cell, args.unit, kind) for kind in ("outputs", "checkpoints")
    )
    if (out_dir / "report.json").exists():
        print("done: {}".format(out_dir))
        return
    runs.write_meta(args.exp, {"config_files": args.config_files})
    check_same_config(out_dir / "train_args.json", args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    train_loader, val_loader, test_loader, train_dataset, test_dataset = build_loaders(args)
    model, classifier, projector, criterion, optimizer = build_model(args)
    scaler = torch.cuda.amp.GradScaler()
    s_ref_sha = (
        save_sc_state(out_dir, train_dataset, test_dataset, args) if train_dataset.sc else None
    )

    start_epoch, best_key, best_model, history, test_preds = 1, -1.0, None, [], {}
    last = ckpt_dir / "last.pth"
    if last.is_file():  # full resume (written every epoch); RNG state is not restored
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), classifier.load_state_dict(ck["classifier"])
        projector.load_state_dict(ck["projector"]), optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_model, best_key = ck["epoch"] + 1, ck["best_model"], ck["best_key"]
        history, test_preds = ck["history"], ck["test_preds"]
        print("=> resumed '{}' after epoch {}".format(last, ck["epoch"]))

    screening = args.cv_folds > 0  # selection never sees test data while screening

    def select_key(h):
        if (
            args.selection == "fixed"
        ):  # registered epoch from screening; later report epochs do not replace it
            return 1.0 if h["epoch"] == args.report_epoch_list[0] else -1.0
        return (h["val"] if args.selection == "cv" else h["test"])["all"]["score"]

    print(
        "Training {} epochs (cell={}, unit={}, selection={})".format(
            args.epochs, args.cell, args.unit, args.selection
        )
    )
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        t0 = time.time()
        _, acc = train_epoch(
            train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler
        )
        print(
            "Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(
                epoch, time.time() - t0, acc
            )
        )

        val_rep = (
            evaluate(val_loader, model, classifier, args)[0] if val_loader is not None else None
        )
        test_rep = None
        if not screening:
            test_rep, preds, probs, _, _ = evaluate(test_loader, model, classifier, args)
            test_preds[epoch] = (preds.astype(np.int8), probs.astype(np.float16))
        history.append({"epoch": epoch, "val": val_rep, "test": test_rep})
        t = (
            test_rep["all"]
            if test_rep
            else {"sp": float("nan"), "se": float("nan"), "score": float("nan")}
        )
        print(
            " * test Sp {:.2f}, Se {:.2f}, Score {:.2f} | val Score {}".format(
                t["sp"],
                t["se"],
                t["score"],
                "{:.2f}".format(val_rep["all"]["score"]) if val_rep else "-",
            )
        )

        key = select_key(history[-1])
        if key > best_key and (args.selection in ("cv", "fixed") or t["se"] > 5):
            best_key = key
            best_model = [
                {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for m in (model, classifier)
            ]
            print("Best ckpt is modified with key = {:.2f} when Epoch = {}".format(key, epoch))
        if args.selection == "fixed" and epoch in args.report_epoch_list:
            save_model(
                model, args, epoch, ckpt_dir / "report_epoch_{}.pth".format(epoch), classifier
            )

        torch.save(
            {
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_model": best_model,
                "best_key": best_key,
                "history": history,
                "test_preds": test_preds,
                "epoch": epoch,
            },
            str(last) + ".tmp",
        )
        (ckpt_dir / "last.pth.tmp").replace(
            last
        )  # atomic: a crash mid-save keeps the previous epoch

    picked = pick_epochs(history)
    if args.selection == "fixed":
        by_epoch = {h["epoch"]: h for h in history}
        picked["fixed"] = {str(e): by_epoch[e] for e in args.report_epoch_list}
    report = {
        "cell": args.cell,
        "unit": args.unit,
        "selection": args.selection,
        "sc_mode": args.sc_mode if args.spectrum_correction else None,
        "s_ref_sha256": s_ref_sha,
        **picked,
        "history": history_json(history),
    }
    if test_preds:  # per-epoch test predictions for paired, patient-level bootstrap CIs offline
        ep = sorted(test_preds)
        np.savez_compressed(
            out_dir / "test_preds.npz",
            epochs=np.array(ep),
            preds=np.stack([test_preds[e][0] for e in ep]),
            probs=np.stack([test_preds[e][1] for e in ep]),
            labels=np.asarray(test_dataset.labels),
            device=np.asarray(test_dataset.devices),
            patient=np.asarray(test_dataset.patients),
        )
    for name in ("last", "cv_selected", "test_best_optimistic"):
        if name in picked and picked[name]["test"] is not None:
            r = picked[name]["test"]["all"]
            print(
                "{:>22} (epoch {:>3}): Sp {:.2f} Se {:.2f} Score {:.2f} HS {:.2f} macroF1 {:.3f}{}".format(
                    name,
                    picked[name]["epoch"],
                    r["sp"],
                    r["se"],
                    r["score"],
                    r["hs"],
                    r["macro_f1"],
                    (
                        "  [OPTIMISTIC: epoch chosen on test]"
                        if name == "test_best_optimistic"
                        else ""
                    ),
                )
            )
    if (
        best_model is not None and args.selection != "fixed" and not screening
    ):  # frozen weights of the selected epoch
        model.load_state_dict(best_model[0]), classifier.load_state_dict(best_model[1])
        save_model(model, args, args.epochs, ckpt_dir / "best.pth", classifier)
    last.unlink()  # resume state; the final weights are the report_epoch/best files
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1)
    )  # written last: marks the unit as done


if __name__ == "__main__":
    main()
