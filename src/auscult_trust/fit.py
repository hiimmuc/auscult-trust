"""Training loop (resumable per epoch), prediction and metrics for AuscultTrust cells."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src import encoders
from src.auscult_trust.augment import ema_step, mixcl_term
from src.auscult_trust.model import make_model, parse_cell
from src.encoders.base import DEVICE
from src.eval.metrics import class_report, icbhi_score


@torch.no_grad()
def predict(model, D, rows, mode, bs):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=mode == "tokens"):
            out.append(model(D.batch(rows[i:i + bs], mode)).float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    rep = class_report(y, p.argmax(1))
    return {"score": float(icbhi_score(y, p.argmax(1))), "f1_crackle": float(rep["f1"][1]), "f1_wheeze": float(rep["f1"][2]),
            "recall": rep["recall"].tolist(), "f1": rep["f1"].tolist()}


def fit(cfg, D, cell, tr, evals, epochs, seed, ckpt=None):
    """Train `cell` on dev rows `tr` and evaluate `evals` ({name: rows}) after every epoch.

    Args:
        cfg: Config dict. `class_weight: balanced` weights the loss by inverse class frequency of `tr` (default: unweighted).
            Recipe flags of the cell (`+mix`, `+ema`) read `ft.mix` and `ft.ema_beta`.
        D: `FeatureData`.
        cell: Cell name.
        tr: Row indices (into the cached arrays) to train on.
        evals: Dict name -> row indices evaluated after each epoch.
        epochs: Number of epochs.
        seed: Seed.
        ckpt: Path for the per-epoch resume checkpoint, or None.

    Returns:
        Tuple (model, curve) with curve = list over epochs of {eval name: metrics}.
    """
    torch.manual_seed(seed)
    model, mode = make_model(cfg, cell, D)
    model = model.to(DEVICE)
    flags = parse_cell(cell)[3]
    patch_index = encoders.get(D.encoder).patch_index().to(DEVICE) if "mix" in flags else None
    p = cfg["ft"] if mode == "tokens" else cfg["head"]
    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    lora = [q for n, q in named if "lora_" in n]
    blocks = [q for n, q in named if "lora_" not in n and ("tail.train_blocks" in n or "tail.norm" in n)]
    used = {id(q) for q in lora + blocks}
    rest = [q for _, q in named if id(q) not in used]
    groups = [(rest, p["lr_head"]), (blocks, p.get("lr_block")), (lora, p.get("lr_lora", 2e-4))]
    opt = torch.optim.AdamW([{"params": ps, "lr": lr} for ps, lr in groups if ps], weight_decay=p["wd"])
    base = [g["lr"] for g in opt.param_groups]
    curve, start = [], 0
    if ckpt and Path(ckpt).exists():
        s = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        curve, start = s["curve"], s["epoch"]
        print(f"  resumed {ckpt} after epoch {start}", flush=True)
    y_tr = D.y[tr]
    cnt = np.bincount(y_tr, minlength=4).clip(min=1)
    weight = torch.tensor(len(y_tr) / (4 * cnt), dtype=torch.float32, device=DEVICE) if cfg.get("class_weight") == "balanced" else None
    for ep in range(start, epochs):
        for g, b0 in zip(opt.param_groups, base):  # cosine decay over the planned epochs
            g["lr"] = 0.5 * b0 * (1 + np.cos(np.pi * ep / epochs))
        model.train()
        perm = np.random.RandomState(seed * 1000 + ep).permutation(len(tr))
        for i in range(0, len(perm), p["bs"]):
            j = perm[i:i + p["bs"]]
            b = D.batch(tr[j], mode)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=mode == "tokens"):
                z = model.embed(b)
                loss = F.cross_entropy(model.head.out(z).float(), torch.tensor(y_tr[j], device=DEVICE), weight=weight)
                if "mix" in flags:
                    loss = loss + p["mix"]["con_weight"] * mixcl_term(model, b, z, patch_index, p["mix"])
            before = [q.detach().clone() for _, q in named] if "ema" in flags else []
            opt.zero_grad()
            loss.backward()
            opt.step()
            if "ema" in flags:
                ema_step([q for _, q in named], before, p["ema_beta"])
        curve.append({n: metrics(D.y[rows], predict(model, D, rows, mode, 128)) for n, rows in evals.items()})
        msg = " ".join(f"{n}:{m['score']:.3f}" for n, m in curve[-1].items())
        print(f"  {cell} epoch {ep + 1}/{epochs} {msg}", flush=True)
        if ckpt:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve, "epoch": ep + 1}, str(ckpt) + ".tmp")
            os.replace(str(ckpt) + ".tmp", ckpt)
    return model, curve

