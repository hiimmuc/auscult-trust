from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import copy
import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util import stage1
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import PatchMixLoss, PatchMixConLoss


def build_parser():
    parser = argparse.ArgumentParser('argument for supervised training')
    parser.add_argument('--config', type=str, default=None,
                        help='json file(s) of argument defaults, comma-separated (configs/stage1_variants/*.json); command line wins')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='headline Se/Score from the 4-class predictions collapsed to normal/abnormal (not a separately trained model)')

    # Stage 1 of proposal v8: epoch selection and device-motivated variants (P1-P8; P5 dropped, no licensed stethoscope IRs)
    parser.add_argument('--selection', type=str, default='cv', choices=['cv', 'test', 'fixed'],
                        help="cv: epoch chosen on a patient-level validation set carved from the training patients; "
                             "test: epoch chosen on the test set (optimistic, original Patch-Mix protocol); "
                             "fixed: train on all training patients and keep the epoch(s) given by --report_epochs "
                             "(the registered refit of a screened variant)")
    parser.add_argument('--val_frac', type=float, default=0.2, help='fraction of training patients held out for validation')
    parser.add_argument('--cv_folds', type=int, default=0,
                        help='screening: >0 uses patient-grouped CV with this many folds (device-stratified, same folds for '
                             'every variant and seed) instead of the --val_frac split; the test set is not evaluated')
    parser.add_argument('--cv_fold', type=int, default=0, help='screening: index of the validation fold')
    parser.add_argument('--report_epochs', type=str, default='',
                        help='selection=fixed: comma list of epochs chosen by CV screening; the first is saved as best.pth, '
                             'every one as report_epoch_<E>.pth')
    parser.add_argument('--select_metric', type=str, default='score', choices=['score', 'worst_device'],
                        help='P8: worst_device selects by the minimum per-device validation Score')
    parser.add_argument('--a1_input', action='store_true', help='P1: A1 spectrum correction on the waveform, device from file name')
    parser.add_argument('--a1_reference', type=str, default='arithmetic', choices=['arithmetic', 'geometric'])
    parser.add_argument('--bin_norm', action='store_true', help='P2: per-device, per-mel-bin standardisation')
    parser.add_argument('--rand_bin_gain', type=float, default=0.0, help='P3: SD (dB) of the random per-bin gain augmentation')
    parser.add_argument('--freq_mixstyle', type=float, default=0.0, help='P4: Freq-MixStyle probability per batch')
    parser.add_argument('--bin_affine', action='store_true', help='P6: learnable per-mel-bin affine layer before the encoder')
    parser.add_argument('--hc_fuse', action='store_true', help='P7: fuse gain-invariant handcrafted descriptors with the deep feature')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=4,
                        help='set k-way classification problem')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--weighted_sampler', action='store_true',
                        help='weighted sampler inversly proportional to class ratio')
    parser.add_argument('--stetho_id', type=int, default=-1, 
                        help='stethoscope device id, use only when finetuning on each stethoscope data')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--butterworth_filter', type=int, default=None, 
                        help='apply specific order butterworth band-pass filter')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--nfft', type=int, default=1024,
                        help='the frequency size of fast fourier transform')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--concat_aug_scale', type=float,  default=0, 
                        help='to control the number (scale) of concatenation-based augmented samples')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--blank_region_clip', action='store_true', 
                        help='remove the blank region, high frequency region')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')
    # Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='grad_block',
                        help='how to make target representation', choices=['grad_block', 'grad_flow', 'project_block', 'project_flow'])

    return parser


def parse_args():
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    if pre.config:
        for c in pre.config.split(','):  # several files = a combination (P9): later files override earlier ones
            parser.set_defaults(**json.load(open(c)))
    args = parser.parse_args()
    args.val_available = args.selection == 'cv'
    args.report_epoch_list = [int(e) for e in str(args.report_epochs).split(',') if str(e).strip()]
    if args.selection == 'fixed':
        assert args.report_epoch_list, '--selection fixed needs --report_epochs'
        assert max(args.report_epoch_list) <= args.epochs, 'report epoch beyond --epochs'
        assert args.cv_folds == 0, 'fixed refit trains on all training patients; do not combine with --cv_folds'
    if args.cv_folds:
        assert args.selection == 'cv' and 0 <= args.cv_fold < args.cv_folds

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    if args.method in ['patchmix', 'patchmix_cl']:
        assert args.model in ['ast', 'ssast']
    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
    else:
        raise NotImplementedError

    return args


def set_loader(args):
    if args.dataset == 'icbhi':
        # get rawo information and calculate mean and std for normalization
        # dataset = ICBHIDataset(train_flag=True, transform=transforms.Compose([transforms.ToTensor()]), args=args, print_flag=False, mean_std=True)
        # mean, std = get_mean_and_std(dataset)
        # args.h, args.w = dataset.h, dataset.w

        # print('*' * 20)
        # print('[Raw dataset information]')
        # print('Stethoscope device number: {}, and patience number without overlap: {}'.format(len(dataset.device_to_id), len(set(sum(dataset.device_id_to_patient.values(), []))) ))
        # for device, id in dataset.device_to_id.items():
        #     print('Device {} ({}): {} number of patience'.format(id, device, len(dataset.device_id_to_patient[id])))
        # print('Spectrogram shpae on ICBHI dataset: {} (height) and {} (width)'.format(args.h, args.w))
        # print('Mean and std of ICBHI dataset: {} (mean) and {} (std)'.format(round(mean.item(), 2), round(std.item(), 2)))
        
        args.h, args.w = 798, 128
        train_transform = [transforms.ToTensor(),
                            SpecAugment(args),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]                        
        # train_transform.append(transforms.Normalize(mean=mean, std=std))
        # val_transform.append(transforms.Normalize(mean=mean, std=std))
        
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        test_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)

        # Patient-level validation split of the training set (same split for every variant of one seed).
        # Shallow copy with the eval transform: shares the cached spectrograms, no SpecAugment, no raw augmentation.
        if args.cv_folds:  # screening fold: same device-stratified patient folds for every variant and seed
            devs = np.array([int(m[stage1.META_DEVICE_IDX].item()) for m in train_dataset.metadata])
            tr_idx, va_idx = stage1.cv_split(train_dataset.patients, devs, args.cv_folds, args.cv_fold)
        else:
            tr_idx, va_idx = stage1.patient_val_split(train_dataset.patients, args.val_frac, args.seed)
        val_dataset = copy.copy(train_dataset)
        val_dataset.transform, val_dataset.train_flag = val_transform, False
        val_set = torch.utils.data.Subset(val_dataset, va_idx)
        if args.selection == 'cv':
            train_set = torch.utils.data.Subset(train_dataset, tr_idx)
            labels_tr = np.asarray(train_dataset.labels)[tr_idx]
        else:
            train_set, labels_tr = train_dataset, np.asarray(train_dataset.labels)
        args.class_nums = np.bincount(labels_tr, minlength=args.n_cls).astype(float)
        class_ratio = args.class_nums / args.class_nums.sum() * 100
        print('train/val patients: {}/{}  cycles {}/{}'.format(len(set(train_dataset.patients[tr_idx])),
              len(set(train_dataset.patients[va_idx])), len(tr_idx), len(va_idx)))
    else:
        raise NotImplemented    
    
    if args.weighted_sampler:
        weights = torch.Tensor(1 / class_ratio[labels_tr])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set))
    else:
        sampler = None

    mk = lambda ds, **kw: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, **kw)
    train_loader = mk(train_set, shuffle=sampler is None, sampler=sampler, drop_last=True)
    # fixed refit: the val patients are inside the training set, so there is no validation loader
    val_loader = mk(val_set, shuffle=False) if args.selection == 'cv' else None
    test_loader = mk(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader, args


def set_model(args):    
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        kwargs['bin_affine_dim'] = args.n_mels if args.bin_affine else None
    elif args.model == 'ssast':
        kwargs['label_dim'] = args.n_cls
        kwargs['fshape'], kwargs['tshape'] = args.fshape, args.tshape
        kwargs['fstride'], kwargs['tstride'] = 10, 10
        kwargs['input_tdim'] = 798
        kwargs['task'] = args.ssast_task
        kwargs['pretrain_stage'] = not args.audioset_pretrained
        kwargs['load_pretrained_mdl_path'] = args.ssast_pretrained_type
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL

    model = get_backbone_class(args.model)(**kwargs)    
    classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    if args.hc_fuse:  # P7: deep feature (768) + invariant handcrafted descriptors
        d = model.final_feat_dim + len(stage1.HC_INVARIANT)
        classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, args.n_cls))

    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)

    if args.model not in ['ast', 'ssast'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")

            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)

        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))

    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method == 'patchmix_cl' else nn.Identity()

    if args.method == 'ce':
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    classifier.cuda()
    projector.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)

    return model, classifier, projector, criterion, optimizer


def fuse(features, metadata, args):
    """P7: append the standardised invariant handcrafted descriptors (metadata[:, 7:]) to the deep feature."""
    if not args.hc_fuse:
        return features
    return torch.cat([features, metadata[:, stage1.META_DEVICE_IDX + 1:].to(features.device, features.dtype)], 1)


def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    classifier.train()
    projector.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, metadata) in enumerate(train_loader):
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]

        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        if args.freq_mixstyle > 0:  # P4
            images = stage1.freq_mixstyle(images, p=args.freq_mixstyle)

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features = model(images)
                output = classifier(fuse(features, metadata, args))
                loss = criterion[0](output, labels)

            elif args.method == 'patchmix':
                mix_images, labels_a, labels_b, lam, index = model(images, y=labels, patch_mix=True, time_domain=args.time_domain)
                output = classifier(fuse(mix_images, metadata, args))
                loss = criterion[1](output, labels_a, labels_b, lam)

            elif args.method == 'patchmix_cl':
                features = model(images)
                output = classifier(fuse(features, metadata, args))
                loss = criterion[0](output, labels)

                if args.target_type == 'grad_block':
                    proj1 = deepcopy(features.detach())
                elif args.target_type == 'grad_flow':
                    proj1 = features
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(projector(features).detach())
                elif args.target_type == 'project_flow':
                    proj1 = projector(features)

                # use 'patchmix_cl' for augmentation
                mix_images, labels_a, labels_b, lam, index = model(images, y=labels, patch_mix=True, time_domain=args.time_domain)
                proj2 = projector(mix_images)
                loss += args.alpha * criterion[1](proj1, proj2, labels, labels_b, lam, index, args)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def evaluate(loader, model, classifier, criterion, args):
    """Run the model on a loader. Returns (per-device report dict, preds, labels, device ids, mean loss)."""
    model.eval()
    classifier.eval()
    losses = AverageMeter()
    preds, probs, ys, devs = [], [], [], []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                output = classifier(fuse(model(images), metadata, args))
                loss = criterion[0](output, labels)
            losses.update(loss.item(), labels.shape[0])
            preds.append(output.argmax(1).cpu().numpy())
            probs.append(torch.softmax(output.float(), 1).cpu().numpy())
            ys.append(labels.cpu().numpy())
            devs.append(metadata[:, stage1.META_DEVICE_IDX].long().numpy())
    preds, ys, devs = np.concatenate(preds), np.concatenate(ys), np.concatenate(devs)
    evaluate.last_probs = np.concatenate(probs)  # softmax of the last call (seed ensembles offline)
    return stage1.per_device_report(ys, preds, devs, args.n_cls), (preds, ys, devs), losses.avg


def headline(report, args):
    """(Sp, Se, Score) of the pooled row; with --two_cls_eval Se is the collapsed normal/abnormal sensitivity."""
    r = report['all']
    return [r['sp'], r['two_cls_se'] if args.two_cls_eval else r['se'], r['two_cls_score'] if args.two_cls_eval else r['score']]


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None
    best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score of the selected epoch on the test set
    best_key = -1.0
    history = []

    train_loader, val_loader, test_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

    args.start_epoch = 1
    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()

    # repro: full resume from `last.pth` (written every epoch) so a reboot costs at most one epoch.
    # The original --resume restored only model + optimizer. RNG state is not restored.
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); classifier.load_state_dict(ck['classifier'])
        projector.load_state_dict(ck['projector']); optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        best_acc, best_model, args.start_epoch = ck['best_acc'], ck['best_model'], ck['epoch'] + 1
        history, best_key = ck.get('history', []), ck.get('best_key', best_acc[-1])
        print("=> resumed '{}' after epoch {}, best Score {:.2f}".format(args.resume, ck['epoch'], best_acc[-1]))

    test_preds = ck.get('test_preds', {}) if args.resume and os.path.isfile(args.resume) else {}  # epoch -> softmax
    screening = args.cv_folds > 0  # no test evaluation while screening: selection never sees test data

    def select_key(h):
        if args.selection == 'fixed':  # registered epoch from screening; any later report epoch does not replace it
            return 1.0 if h['epoch'] == args.report_epoch_list[0] else -1.0
        src = h['val'] if args.selection == 'cv' else h['test']
        return src['all']['score'] if args.select_metric == 'score' else stage1.worst_device_score(src)

    print('*' * 20)
    if not args.eval:
        print('Training for {} epochs on {} dataset (selection={}, metric={})'.format(args.epochs, args.dataset, args.selection, args.select_metric))
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2-time1, acc))

            # validation (held-out train patients) and test, every epoch; preds only kept for the three reported epochs
            val_rep = evaluate(val_loader, model, classifier, criterion, args)[0] if val_loader is not None else None
            test_rep, test_raw = (None, None) if screening else evaluate(test_loader, model, classifier, criterion, args)[:2]
            h = {'epoch': epoch, 'val': val_rep, 'test': test_rep}
            sp, se, sc = headline(test_rep, args) if test_rep is not None else (float('nan'),) * 3
            if test_raw is not None:
                test_preds[epoch] = (test_raw[0].astype(np.int8), evaluate.last_probs.astype(np.float16))  # preds, softmax
            print(' * test S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} | val Score {}'.format(
                sp, se, sc, '{:.2f}'.format(val_rep['all']['score']) if val_rep is not None else '-'))
            history.append(h)

            key = select_key(h)
            save_bool = key > best_key and (args.selection in ('cv', 'fixed') or se > 5)
            if save_bool:
                best_key, best_acc = key, [sp, se, sc]
                best_model = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in (model, classifier)]  # repro: keep on CPU, saves ~0.7 GB GPU
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with key = {:.2f} when Epoch = {}'.format(key, epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)

            if args.selection == 'fixed' and epoch in args.report_epoch_list:
                save_model(model, optimizer, args, epoch, os.path.join(args.save_folder, 'report_epoch_{}.pth'.format(epoch)), classifier)

            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)

            last = os.path.join(args.save_folder, 'last.pth')
            torch.save({'model': model.state_dict(), 'classifier': classifier.state_dict(),
                        'projector': projector.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), 'best_acc': best_acc, 'best_model': best_model,
                        'best_key': best_key, 'history': history, 'test_preds': test_preds, 'epoch': epoch}, last + '.tmp')
            os.replace(last + '.tmp', last)  # atomic: a crash mid-save keeps the previous epoch

        # three numbers: last epoch, validation-selected epoch, best on test (optimistic: chosen with test labels)
        three = stage1.pick_epochs(history, args.select_metric)
        if args.selection == 'fixed':
            byep = {x['epoch']: x for x in history}
            three['fixed'] = {str(e): byep[e] for e in args.report_epoch_list}
        sc_of = lambda r: None if r is None else r['all']['score']
        report = {'args': {k: v for k, v in vars(args).items() if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))},
                  'selection': args.selection, 'select_metric': args.select_metric, **three, 'history': [
                      {'epoch': x['epoch'], 'val_score': sc_of(x['val']), 'test_score': sc_of(x['test']),
                       'val_worst': None if x['val'] is None else stage1.worst_device_score(x['val']),
                       'val_devices': None if x['val'] is None else {k: v['score'] for k, v in x['val'].items() if k != 'all'}}
                      for x in history]}
        with open(os.path.join(args.save_folder, 'stage1_report.json'), 'w') as f:
            json.dump(report, f, indent=1)
        if test_preds:  # per-epoch test predictions for paired, patient-level bootstrap CIs offline
            ep = sorted(test_preds)
            np.savez_compressed(os.path.join(args.save_folder, 'test_preds.npz'), epochs=np.array(ep),
                                preds=np.stack([test_preds[e][0] for e in ep]), probs=np.stack([test_preds[e][1] for e in ep]),
                                labels=np.asarray(test_loader.dataset.labels),
                                device=np.array([int(m[stage1.META_DEVICE_IDX].item()) for m in test_loader.dataset.metadata]),
                                patient=np.asarray(test_loader.dataset.patients))
        named = [(n, three[n]) for n in ('last', 'cv_selected', 'test_best_optimistic') if n in three]
        named += [('fixed@' + e, x) for e, x in three.get('fixed', {}).items()]
        for name, entry in named:
            if entry['test'] is None:
                print('{:>22} (epoch {:>3}): val Score {:.2f} (screening, test not evaluated)'.format(name, entry['epoch'], entry['val']['all']['score']))
                continue
            r = entry['test']['all']
            print('{:>22} (epoch {:>3}): Sp {:.2f} Se {:.2f} Score {:.2f} HS {:.2f} macroF1 {:.3f}{}'.format(
                name, entry['epoch'], r['sp'], r['se'], r['score'], r['hs'], r['macro_f1'],
                '  [OPTIMISTIC: epoch chosen on test]' if name == 'test_best_optimistic' else ''))

        # save a checkpoint of classifier with the selected epoch
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        rep, _, _ = evaluate(test_loader, model, classifier, criterion, args)
        best_acc = headline(rep, args)
        print(json.dumps(rep, indent=1))

    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))

if __name__ == '__main__':
    main()
