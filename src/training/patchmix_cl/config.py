"""Command line and YAML config of a training run (Patch-Mix CL recipe plus the device-shift variants).

`--config a.yaml,b.yaml` loads files in order, later files override earlier ones (`experiments/train/base.yaml`
plus one arm file); the command line wins over both. Unknown YAML keys are an error.
"""

import argparse
import math
from pathlib import Path

import yaml
from src.paths import DATA
from src.processing.icbhi import CLASSES, DEVICES

from .encoders import NAMES, get_encoder

EXP = "train"


def build_parser():
    p = argparse.ArgumentParser("Train Patch-Mix CL with optional device-shift variants")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file(s), comma-separated (experiments/train/*.yaml)",
    )
    p.add_argument(
        "--exp",
        type=str,
        default=EXP,
        help="experiment name: outputs/<exp>/<run_id>/<cell>/<unit>/",
    )
    p.add_argument(
        "--cell",
        type=str,
        default=None,
        help='variant name; default: config stems joined by "+", without "base"',
    )
    p.add_argument(
        "--unit",
        type=str,
        default=None,
        help="fit name; default: cv<fold> when screening, else seed<seed>",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=10)

    # epoch selection
    p.add_argument(
        "--selection",
        type=str,
        default="cv",
        choices=["cv", "test", "fixed"],
        help="cv: epoch chosen on held-out training patients; test: chosen on the test set (optimistic, "
        "original Patch-Mix protocol); fixed: train on all training patients and keep --report_epochs",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="fraction of training patients held out for validation",
    )
    p.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="screening: >0 uses device-stratified patient-grouped CV with this many folds (same folds for "
        "every variant and seed) instead of --val_frac; the test set is not evaluated",
    )
    p.add_argument(
        "--cv_fold", type=int, default=0, help="screening: index of the validation fold"
    )
    p.add_argument(
        "--stop_epoch",
        type=int,
        default=0,
        help="stop after this epoch (0: run all --epochs); the schedule is unchanged, so the weights equal the full run's",
    )
    p.add_argument(
        "--report_epochs",
        type=str,
        default="",
        help="selection=fixed: comma list of epochs chosen by CV screening; each is saved as report_epoch_<E>.pth",
    )

    # device-shift variants
    p.add_argument(
        "--spectrum_correction",
        action="store_true",
        help="P1: spectrum correction (SC) on the waveform, device from file name",
    )
    p.add_argument(
        "--sc_reference", type=str, default="arithmetic", choices=["arithmetic", "geometric"]
    )
    p.add_argument(
        "--sc_mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help='SC coefficient bound (chosen by CV score): "dynamic" clips every bin to +-sc_limit_freq_diff dB; '
        '"static" leaves [sc_limit_freq_low, sc_limit_freq_high] Hz unclipped and zeroes the correction outside',
    )
    p.add_argument("--sc_limit_freq_low", type=float, default=50.0, help='Hz, "static" only')
    p.add_argument("--sc_limit_freq_high", type=float, default=2000.0, help='Hz, "static" only')
    p.add_argument("--sc_limit_freq_diff", type=float, default=20.0, help='dB, "dynamic" only')
    p.add_argument(
        "--sc_only_device",
        type=str,
        default=None,
        choices=DEVICES,
        help="apply SC to this device only (others stay raw); inference-time SC on weights trained without SC",
    )
    p.add_argument(
        "--sc_test_profile",
        type=str,
        default="target",
        choices=["target", "source"],
        help="official split: test clips use coefficients from their own unlabelled test clips (target, transductive) "
        "or the training coefficient of the same device (source, inductive)",
    )
    p.add_argument(
        "--random_gain_db",
        type=float,
        default=0.0,
        help="P3: SD (dB) of the random per-bin gain augmentation",
    )
    p.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="weight of the Jensen-Shannon consistency loss between clean images and a random per-bin gain view (0: off)",
    )
    p.add_argument("--consistency_gain_db", type=float, default=6.0, help="SD (dB) of the consistency view's gain knots")
    p.add_argument("--consistency_ramp", type=int, default=5, help="epochs over which the consistency weight ramps up")
    p.add_argument(
        "--freq_mixstyle", type=float, default=0.0, help="P4: Freq-MixStyle probability per batch"
    )

    # optimisation
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--lr_decay_epochs", type=str, default="120,160")
    p.add_argument("--lr_decay_rate", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--cosine", action="store_true", help="cosine annealing")
    p.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    p.add_argument("--warm_epochs", type=int, default=0)
    p.add_argument(
        "--mix_beta", type=float, default=1.0, help="patch-mix interpolation coefficient"
    )
    p.add_argument("--time_domain", action="store_true", help="patch mix over the time axis only")

    # data
    p.add_argument(
        "--data_folder",
        type=str,
        default=str(DATA / "processed" / "patchmix_icbhi"),
        help="holds audio_test_data/, official_split.txt",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--n_cls",
        type=int,
        default=4,
        choices=[2, 4],
        help="lung-sound classes: 4 (normal, crackle, wheeze, both) or 2",
    )
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--desired_length", type=int, default=8, help="seconds per cycle after cut/pad")
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--spec_augment", type=int, default=1, help="1: SpecAugment on training images; 0: off")
    p.add_argument("--noise_train_prob", type=float, default=0.0,
                   help="probability of mixing a stationary ESC-50 noise (folds 1-3) into a training image (0: off)")
    p.add_argument("--noise_snr_low", type=float, default=5.0, help="lowest SNR (dB) of the training noise")
    p.add_argument("--noise_snr_high", type=float, default=30.0, help="highest SNR (dB) of the training noise")
    p.add_argument("--pad_types", type=str, default="repeat", choices=["zero", "repeat"])
    p.add_argument(
        "--test_fold",
        type=str,
        default="official",
        choices=["official", "0", "1", "2", "3", "4"],
        help="official: 60/40 challenge split; 0-4: patient fold held out as test (80/20 split of patient_list_foldwise.txt)",
    )

    # model (AST) and loss
    p.add_argument(
        "--from_sl_official",
        action="store_true",
        help="start from the ImageNet-pretrained DeiT weights",
    )
    p.add_argument(
        "--audioset_pretrained",
        action="store_true",
        help="start from the AudioSet-pretrained AST weights",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="ast",
        choices=NAMES,
        help="backbone; opera_ct, clap and hear are frozen-only",
    )
    p.add_argument(
        "--holdout_device",
        type=str,
        default=None,
        choices=DEVICES,
        help="leave-one-device-out: test = every cycle of this device, calibration and training "
        "patients from the other devices (needs --selection fixed)",
    )
    p.add_argument(
        "--lodo_cal_frac",
        type=float,
        default=0.2,
        help="share of each source device's patients used for calibration (leave-one-device-out)",
    )
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="train only the classifier and projector, not the pretrained encoder",
    )
    p.add_argument("--ma_update", action="store_true", help="moving-average update of the weights")
    p.add_argument("--ma_beta", type=float, default=0.0)
    p.add_argument(
        "--method",
        type=str,
        default="patchmix_cl",
        choices=["ce", "patchmix", "patchmix_cl", "ce_pafa", "sg_scl", "qlung", "add_rsc"],
        help="ce: plain cross-entropy; patchmix: Patch-Mix augmentation; patchmix_cl: Patch-Mix + contrastive loss; "
        "ce_pafa: cross-entropy + PAFA patient losses (PCSL, GPAL) on the projector output; sg_scl: cross-entropy + "
        "stethoscope-guided supervised contrastive loss on two SpecAugment views behind gradient reversal; qlung: angular "
        "classifier + quality-adaptive angular margin loss; add_rsc: adaptive differential denoising layer in front of the "
        "encoder + bias denoising loss",
    )
    p.add_argument("--quality_alpha", type=float, default=0.7, help="qlung: weight of the spectral entropy in the quality score")
    p.add_argument("--quality_beta", type=float, default=0.3, help="qlung: weight of the RMS energy in the quality score")
    p.add_argument("--qlung_gamma", type=float, default=0.5, help="qlung: share of the quality margin vs the class-prior margin")
    p.add_argument("--qlung_m_target", type=float, default=0.2, help="qlung: class-prior margin of a class with prior 1/4")
    p.add_argument("--denoise_heads", type=int, default=8, help="add_rsc: heads of the differential attention")
    p.add_argument("--denoise_depth", type=int, default=6, help="add_rsc: layer depth that sets the initial lambda")
    p.add_argument("--loss_beta", type=float, default=0.5, help="add_rsc: weight of the bias denoising loss")
    p.add_argument("--proj_dim", type=int, default=768)
    p.add_argument("--lambda_pcsl", type=float, default=5.0, help="ce_pafa: weight of the patient cohesion-separation loss")
    p.add_argument("--lambda_gpal", type=float, default=0.005, help="ce_pafa: weight of the global patient alignment loss")
    p.add_argument("--temperature", type=float, default=0.06)
    p.add_argument("--alpha", type=float, default=1.0, help="weight of the contrastive loss")
    p.add_argument("--grl_schedule", choices=["reference", "progress"], default="reference",
                   help="sg_scl gradient-reversal weight 2 / (1 + exp(-10 p)) - 1. reference: p = step / epochs as in the SG-SCL code, "
                        "which is >= 10 from the first step, so the weight is 1 throughout; progress: p = fraction of training done "
                        "(the DANN ramp from 0 to 1)")
    p.add_argument("--negative_pair", type=str, default="all", choices=["all", "diff_label"])
    p.add_argument(
        "--cl_mode",
        type=str,
        default="2d",
        choices=["2d", "dual"],
        help="2d: one contrastive loss on the 2-D Patch-Mix view; dual (DASS only, Lung-SRAD): one loss on a time-only and one "
        "on a frequency-only Patch-Mix view, each weighted by --alpha, with the mixed features detached",
    )
    p.add_argument(
        "--gaussian_blur",
        action="store_true",
        help="DASS only: Gaussian blur inside VMamba blocks 2 and 3 of stage 2 (kernel 5, sigma 3), as in Lung-SRAD",
    )
    p.add_argument(
        "--target_type",
        type=str,
        default="grad_block",
        choices=["grad_block", "grad_flow", "project_block", "project_flow"],
    )
    return p


def load_config(parser, paths):
    """Apply YAML files in order as parser defaults. Unknown keys raise."""
    known = {a.dest for a in parser._actions}
    for path in paths:
        cfg = yaml.safe_load(open(path)) or {}
        unknown = set(cfg) - known
        if unknown:
            raise KeyError("{}: unknown keys {}".format(path, sorted(unknown)))
        parser.set_defaults(**cfg)


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    files = [c for c in (pre.config or "").split(",") if c]
    load_config(parser, files)
    args = parser.parse_args(argv)

    args.config_files = files
    args.cell = (
        args.cell or "+".join(Path(f).stem for f in files if Path(f).stem != "base") or "run"
    )
    args.unit = args.unit or (
        "cv{}".format(args.cv_fold) if args.cv_folds else "seed{}".format(args.seed)
    )
    args.report_epoch_list = [int(e) for e in str(args.report_epochs).split(",") if e.strip()]
    if args.selection == "fixed":
        assert args.report_epoch_list, "--selection fixed needs --report_epochs"
        assert max(args.report_epoch_list) <= args.epochs, "report epoch beyond --epochs"
        assert (
            args.cv_folds == 0
        ), "fixed refit trains on all training patients; do not combine with --cv_folds"
    if args.cv_folds:
        assert args.selection == "cv" and 0 <= args.cv_fold < args.cv_folds
    assert (
        args.freeze_encoder or not get_encoder(args.encoder).frozen_only
    ), "{} is only trained with a frozen encoder (add freeze_encoder.yaml)".format(args.encoder)
    if args.holdout_device:
        assert (
            args.selection == "fixed" and args.cv_folds == 0
        ), "--holdout_device needs --selection fixed (epoch frozen from screening) and no --cv_folds"
    if args.cl_mode == "dual" or args.gaussian_blur:
        assert args.encoder == "dass", "dual-axis Patch-Mix and the Gaussian blur exist only in the Lung-SRAD DASS"
    if args.cl_mode == "dual":
        assert args.method == "patchmix_cl" and not args.time_domain
    if args.holdout_device:
        assert args.test_fold == "official", "leave-one-device-out builds its own split"
    if args.noise_train_prob > 0:
        assert get_encoder(args.encoder).db_scale is not None, "training noise is mixed in the log-mel power domain"
    if args.consistency_weight > 0:
        assert args.random_gain_db == 0, "consistency training replaces the random gain augmentation; set random_gain_db 0"
        assert args.method == "patchmix_cl", "consistency loss is wired into the patchmix_cl branch"
        assert get_encoder(args.encoder).db_scale is not None, "consistency gain needs a log-mel image"
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(",")]
    args.cls_list = CLASSES if args.n_cls == 4 else ["normal", "abnormal"]
    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate**3)
            args.warmup_to = (
                eta_min
                + (args.learning_rate - eta_min)
                * (1 + math.cos(math.pi * args.warm_epochs / args.epochs))
                / 2
            )
        else:
            args.warmup_to = args.learning_rate
    return args
