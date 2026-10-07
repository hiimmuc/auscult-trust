"""Stage 1 -> Stage 2 bridge: freeze a trained Patch-Mix CL run and export softmax + embeddings for the conformal pipeline.

Writes `<out>/<split>.npz` for the official test set and, if the run had one, the held-out validation patients (`probs` (n, K), `emb` (n, 768),
`labels`, `device` (0 Meditron, 1 LittC2SE, 2 Litt3200, 3 AKGC417L), `patient`), and `<out>/model.sha256` with the hash of
the weight file, so every Stage 2 number is tied to one frozen checkpoint. The weights are not modified.

Usage: .venv/bin/python export_probs.py save/<run>/ [--out DIR]    (run folder holds train_args.json and best.pth)
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

import main as M
from util import stage1


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def run(loader, model, classifier, args):
    model.eval(); classifier.eval()
    P, E, Y, D = [], [], [], []
    for images, labels, metadata in loader:
        feat = model(images.cuda(non_blocking=True))
        out = classifier(M.fuse(feat, metadata, args))
        P.append(torch.softmax(out.float(), 1).cpu().numpy()); E.append(feat.float().cpu().numpy())
        Y.append(labels.numpy()); D.append(metadata[:, stage1.META_DEVICE_IDX].long().numpy())
    return np.concatenate(P), np.concatenate(E), np.concatenate(Y), np.concatenate(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--ckpt', default='best.pth', help='checkpoint in run_dir (fixed refit: report_epoch_<E>.pth)')
    a = ap.parse_args()
    args = M.build_parser().parse_args([])
    vars(args).update(json.load(open(os.path.join(a.run_dir, 'train_args.json'))))
    args.num_workers = a.num_workers
    train_loader, val_loader, test_loader, args = M.set_loader(args)
    model, classifier, _, _, _ = M.set_model(args)
    ckpt = os.path.join(a.run_dir, a.ckpt)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model']); classifier.load_state_dict(ck['classifier'])
    out = a.out or os.path.join(a.run_dir, 'export')
    os.makedirs(out, exist_ok=True)
    patients = {'test': np.asarray(test_loader.dataset.patients)}
    loaders = [('test', test_loader)]
    if val_loader is not None:  # a fixed refit (--selection fixed) trained on every training patient: no held-out val
        patients['val'] = np.asarray(val_loader.dataset.dataset.patients)[val_loader.dataset.indices]
        loaders.append(('val', val_loader))
    for name, loader in loaders:
        P, E, Y, D = run(loader, model, classifier, args)
        np.savez_compressed(os.path.join(out, name + '.npz'), probs=P, emb=E, labels=Y, device=D, patient=patients[name])
        print(name, P.shape, 'pooled Score {:.2f}'.format(stage1.per_device_report(Y, P.argmax(1), D)['all']['score']))
    open(os.path.join(out, 'model.sha256'), 'w').write(sha256(ckpt) + '  ' + os.path.basename(ckpt) + '\n')


if __name__ == '__main__':
    main()
