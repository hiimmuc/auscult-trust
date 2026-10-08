"""Where does a frozen encoder keep abnormal-sound information? Linear probe per depth and per pooling.

Features come from the frozen encoder at every transformer layer (Swin: every stage) with mean or max pooling over the
patch tokens (HeAR: over the tokens of the four 2 s clips). A logistic-regression probe is fitted on the training
patients and scored on the held-out validation patients (the patient-level split of the training set; the test set is
never read). Reports AUROC of abnormal vs normal (score = 1 - p_normal), Sp and Se at the argmax.

Usage: python -m src.evaluation.layer_probe --encoder hear [--seeds 0 1 2] [--batch 16] [--out outputs/layer_probe]
"""

import argparse
import copy
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.evaluation.fm_diagnose import auroc
from src.paths import OUTPUTS
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.training.patchmix_cl.config import parse_args
from src.training.patchmix_cl.dataset import build_loaders
from src.training.patchmix_cl.encoders import get_encoder


def _pool(tokens, groups=1):
    """(B * groups, T, D) tokens -> mean and max features (B, D); `groups` clips of one cycle are pooled together."""
    d = tokens.shape[-1]
    mean = tokens.mean(1).reshape(-1, groups, d).mean(1)
    mx = tokens.max(1).values.reshape(-1, groups, d).max(1).values
    return mean, mx


@torch.no_grad()
def layer_features(name, model, x):
    """List of (layer name, mean feature, max feature) for one batch."""
    out = []
    if name == "hear":
        b = x.shape[0]
        h = model.vit.embeddings(x.reshape(b * 4, 1, 192, 128))
        for i, layer in enumerate(model.vit.layers, 1):
            h = layer(h)
            out.append((f"L{i}", *_pool(h[:, 1:], 4)))
        out.append(("final", *_pool(model.vit.layernorm(h)[:, 1:], 4)))
    else:  # Swin family (htsat, opera_ct) and CLAP: stage by stage
        e = model.enc if name == "clap" else model.htsat
        if name == "clap":
            x = e.batch_norm(x.transpose(1, 3)).transpose(1, 3)
            x = e.patch_embed(e.reshape_mel2img(x), None)
            for i, layer in enumerate(e.layers):
                x = layer(x, e.input_resolutions[i], False, False)[0]
                out.append((f"S{i + 1}", *_pool(x)))
        else:
            x = e.bn0(x.transpose(1, 3)).transpose(1, 3)
            x = e.patch_embed(e.reshape_wav2img(x))
            if e.ape:
                x = x + e.absolute_pos_embed
            x = e.pos_drop(x)
            for i, layer in enumerate(e.layers):
                x = layer(x)[0]
                out.append((f"S{i + 1}", *_pool(x)))
        out.append(("final", *_pool(e.norm(x))))
    return out


def extract(name, model, loader):
    feats, ys = {}, []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            res = layer_features(name, model, x.cuda(non_blocking=True))
        for layer, mean, mx in res:
            for pool, f in (("mean", mean), ("max", mx)):
                feats.setdefault((layer, pool), []).append(f.float().cpu().numpy())
        ys.append(y.numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(ys)


def probe(ftr, ytr, fva, yva):
    sc = StandardScaler().fit(ftr)
    clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(ftr), ytr)
    p = clf.predict_proba(sc.transform(fva))
    full = np.zeros((len(fva), 4))
    full[:, clf.classes_] = p
    pred = full.argmax(1)
    return {
        "auroc_abnormal": auroc(1 - full[:, 0], yva != 0),
        "sp": 100 * float((pred[yva == 0] == 0).mean()),
        "se": 100 * float((pred[yva > 0] == yva[yva > 0]).mean()),
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--encoder", required=True)
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--out", default=str(OUTPUTS / "layer_probe"))
    a = a.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.12)  # about 4 GB: shares the GPU with running jobs
    res = {}
    for seed in a.seeds:
        args = parse_args(
            ["--config", f"experiments/train/base.yaml,experiments/train/encoder_{a.encoder}.yaml,"
             "experiments/train/freeze_encoder.yaml", "--seed", str(seed), "--num_workers", "4"]
        )
        train_loader, val_loader, _, train_dataset, _, _, _ = build_loaders(args)
        eval_train = copy.copy(train_dataset)
        eval_train.transform, eval_train.train_flag = transforms.ToTensor(), False
        tr = DataLoader(Subset(eval_train, train_loader.dataset.indices), batch_size=a.batch, num_workers=4)
        va = DataLoader(val_loader.dataset, batch_size=a.batch, num_workers=4)
        model = get_encoder(args.encoder).build(args).cuda().eval()
        ftr, ytr = extract(args.encoder, model, tr)
        fva, yva = extract(args.encoder, model, va)
        del model
        torch.cuda.empty_cache()
        for k in ftr:
            res.setdefault(k, []).append(probe(ftr[k], ytr, fva[k], yva))
    rows = []
    print("| layer | pooling | AUROC abnormal | Sp | Se |\n|---|---|---|---|---|")
    for (layer, pool), rs in res.items():
        m = lambda key: "{:.3f}±{:.3f}".format(np.mean([r[key] for r in rs]), np.std([r[key] for r in rs]))
        print(f"| {layer} | {pool} | {m('auroc_abnormal')} | {m('sp')} | {m('se')} |")
        rows.append({"layer": layer, "pool": pool, "runs": rs})
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.encoder}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
