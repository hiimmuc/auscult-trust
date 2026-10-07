"""Command line and YAML config of a training run (Patch-Mix CL recipe plus the device-shift variants).

`--config a.yaml,b.yaml` loads files in order, later files override earlier ones (`experiments/train/base.yaml`
plus one arm file); the command line wins over both. Unknown YAML keys are an error.
"""
import argparse
import math
from pathlib import Path

import yaml

from src.paths import DATA
from src.processing.icbhi import CLASSES

EXP = 'train'


def build_parser():
    p = argparse.ArgumentParser('Train Patch-Mix CL with optional device-shift variants')
    p.add_argument('--config', type=str, default=None, help='YAML file(s), comma-separated (experiments/train/*.yaml)')
    p.add_argument('--exp', type=str, default=EXP, help='experiment name: outputs/<exp>/<run_id>/<cell>/<unit>/')
    p.add_argument('--cell', type=str, default=None, help='variant name; default: config stems joined by "+", without "base"')
    p.add_argument('--unit', type=str, default=None, help='fit name; default: cv<fold> when screening, else seed<seed>')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print_freq', type=int, default=10)

    # epoch selection
    p.add_argument('--selection', type=str, default='cv', choices=['cv', 'test', 'fixed'],
                   help='cv: epoch chosen on held-out training patients; test: chosen on the test set (optimistic, '
                        'original Patch-Mix protocol); fixed: train on all training patients and keep --report_epochs')
    p.add_argument('--val_frac', type=float, default=0.2, help='fraction of training patients held out for validation')
    p.add_argument('--cv_folds', type=int, default=0,
                   help='screening: >0 uses device-stratified patient-grouped CV with this many folds (same folds for '
                        'every variant and seed) instead of --val_frac; the test set is not evaluated')
    p.add_argument('--cv_fold', type=int, default=0, help='screening: index of the validation fold')
    p.add_argument('--report_epochs', type=str, default='',
                   help='selection=fixed: comma list of epochs chosen by CV screening; each is saved as report_epoch_<E>.pth')

    # device-shift variants
    p.add_argument('--spectrum_correction', action='store_true', help='P1: spectrum correction (SC) on the waveform, device from file name')
    p.add_argument('--sc_reference', type=str, default='arithmetic', choices=['arithmetic', 'geometric'])
    p.add_argument('--sc_mode', type=str, default='dynamic', choices=['dynamic', 'static'],
                   help='SC coefficient bound (chosen by CV score): "dynamic" clips every bin to +-sc_limit_freq_diff dB; '
                        '"static" leaves [sc_limit_freq_low, sc_limit_freq_high] Hz unclipped and zeroes the correction outside')
    p.add_argument('--sc_limit_freq_low', type=float, default=50.0, help='Hz, "static" only')
    p.add_argument('--sc_limit_freq_high', type=float, default=2000.0, help='Hz, "static" only')
    p.add_argument('--sc_limit_freq_diff', type=float, default=20.0, help='dB, "dynamic" only')
    p.add_argument('--random_gain_db', type=float, default=0.0, help='P3: SD (dB) of the random per-bin gain augmentation')
    p.add_argument('--freq_mixstyle', type=float, default=0.0, help='P4: Freq-MixStyle probability per batch')

    # optimisation
    p.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--lr_decay_epochs', type=str, default='120,160')
    p.add_argument('--lr_decay_rate', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--cosine', action='store_true', help='cosine annealing')
    p.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    p.add_argument('--warm_epochs', type=int, default=0)
    p.add_argument('--mix_beta', type=float, default=1.0, help='patch-mix interpolation coefficient')
    p.add_argument('--time_domain', action='store_true', help='patch mix over the time axis only')

    # data
    p.add_argument('--data_folder', type=str, default=str(DATA / 'processed' / 'patchmix_icbhi'),
                   help='holds audio_test_data/, official_split.txt')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--n_cls', type=int, default=4, choices=[2, 4], help='lung-sound classes: 4 (normal, crackle, wheeze, both) or 2')
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--desired_length', type=int, default=8, help='seconds per cycle after cut/pad')
    p.add_argument('--n_mels', type=int, default=128)
    p.add_argument('--pad_types', type=str, default='repeat', choices=['zero', 'repeat'])

    # model (AST) and loss
    p.add_argument('--from_sl_official', action='store_true', help='start from the ImageNet-pretrained DeiT weights')
    p.add_argument('--audioset_pretrained', action='store_true', help='start from the AudioSet-pretrained AST weights')
    p.add_argument('--freeze_encoder', action='store_true', help='train only the classifier and projector, not the pretrained encoder')
    p.add_argument('--ma_update', action='store_true', help='moving-average update of the weights')
    p.add_argument('--ma_beta', type=float, default=0.0)
    p.add_argument('--method', type=str, default='patchmix_cl', choices=['ce', 'patchmix', 'patchmix_cl'],
                   help='ce: plain cross-entropy; patchmix: Patch-Mix augmentation; patchmix_cl: Patch-Mix + contrastive loss')
    p.add_argument('--proj_dim', type=int, default=768)
    p.add_argument('--temperature', type=float, default=0.06)
    p.add_argument('--alpha', type=float, default=1.0, help='weight of the contrastive loss')
    p.add_argument('--negative_pair', type=str, default='all', choices=['all', 'diff_label'])
    p.add_argument('--target_type', type=str, default='grad_block',
                   choices=['grad_block', 'grad_flow', 'project_block', 'project_flow'])
    return p


def load_config(parser, paths):
    """Apply YAML files in order as parser defaults. Unknown keys raise."""
    known = {a.dest for a in parser._actions}
    for path in paths:
        cfg = yaml.safe_load(open(path)) or {}
        unknown = set(cfg) - known
        if unknown:
            raise KeyError('{}: unknown keys {}'.format(path, sorted(unknown)))
        parser.set_defaults(**cfg)


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    files = [c for c in (pre.config or '').split(',') if c]
    load_config(parser, files)
    args = parser.parse_args(argv)

    args.config_files = files
    args.cell = args.cell or '+'.join(Path(f).stem for f in files if Path(f).stem != 'base') or 'run'
    args.unit = args.unit or ('cv{}'.format(args.cv_fold) if args.cv_folds else 'seed{}'.format(args.seed))
    args.report_epoch_list = [int(e) for e in str(args.report_epochs).split(',') if e.strip()]
    if args.selection == 'fixed':
        assert args.report_epoch_list, '--selection fixed needs --report_epochs'
        assert max(args.report_epoch_list) <= args.epochs, 'report epoch beyond --epochs'
        assert args.cv_folds == 0, 'fixed refit trains on all training patients; do not combine with --cv_folds'
    if args.cv_folds:
        assert args.selection == 'cv' and 0 <= args.cv_fold < args.cv_folds
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
    args.cls_list = CLASSES if args.n_cls == 4 else ['normal', 'abnormal']
    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args
