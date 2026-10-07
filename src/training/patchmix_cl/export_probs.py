"""Freeze a trained run and export softmax + embeddings for the conformal evaluation.

Writes `<unit outputs dir>/export/<split>.npz` for the official test set and, when the run had one, the held-out
validation patients (`probs` (n, K), `emb` (n, 768), `labels`, `device` (ids of `src.processing.icbhi.DEVICES`),
`patient`), and `export/model.sha256` with the hash of the weight file, so every downstream number is tied to one frozen
checkpoint. The weights are not modified.

    python -m src.training.patchmix_cl.export_probs outputs/train/<run_id>/<cell>/<unit> --ckpt report_epoch_<E>.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import runs
from .config import build_parser
from .dataset import build_loaders
from .modeling import build_model
from .reporting import per_device_report


@torch.no_grad()
def run(loader, model, classifier):
    model.eval(), classifier.eval()
    probs, emb, ys, devs = [], [], [], []
    for images, labels, devices in loader:
        feat = model(images.cuda(non_blocking=True))
        probs.append(torch.softmax(classifier(feat).float(), 1).cpu().numpy())
        emb.append(feat.float().cpu().numpy())
        ys.append(labels.numpy())
        devs.append(devices.numpy())
    return tuple(np.concatenate(x) for x in (probs, emb, ys, devs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('unit_dir', help='outputs/<exp>/<run_id>/<cell>/<unit> of the run (holds train_args.json)')
    ap.add_argument('--ckpt', required=True, help='weight file in the unit checkpoints dir (fixed refit: report_epoch_<E>.pth)')
    ap.add_argument('--num_workers', type=int, default=4)
    a = ap.parse_args()
    out_unit, ckpt_unit = Path(a.unit_dir), runs.twin_dir(a.unit_dir, 'checkpoints')
    args = build_parser().parse_args([])
    vars(args).update(json.load(open(out_unit / 'train_args.json')))
    args.num_workers = a.num_workers

    _, val_loader, test_loader, _, _ = build_loaders(args)
    model, classifier, _, _, _ = build_model(args)
    ckpt = ckpt_unit / a.ckpt
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if ck['model'] is not None:  # absent for a frozen encoder: the pretrained weights built by build_model are the encoder
        model.load_state_dict(ck['model'])
    classifier.load_state_dict(ck['classifier'])

    out = out_unit / 'export'
    out.mkdir(exist_ok=True)
    loaders = [('test', test_loader, np.asarray(test_loader.dataset.patients))]
    if val_loader is not None:  # a fixed refit trains on every training patient: no held-out val
        loaders.append(('val', val_loader, np.asarray(val_loader.dataset.dataset.patients)[val_loader.dataset.indices]))
    for name, loader, patients in loaders:
        P, E, Y, D = run(loader, model, classifier)
        np.savez_compressed(out / (name + '.npz'), probs=P, emb=E, labels=Y, device=D, patient=patients)
        print(name, P.shape, 'pooled Score {:.2f}'.format(per_device_report(Y, P.argmax(1), D)['all']['score']))
    (out / 'model.sha256').write_text('{}  {}\n'.format(runs.sha256_file(ckpt), ckpt.name))


if __name__ == '__main__':
    main()
