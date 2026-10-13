"""One training epoch and one evaluation pass."""

import contextlib
import sys
import time
from copy import deepcopy

import numpy as np
import torch

from .augment import freq_mixstyle, js_divergence, random_bin_gain_batch
from .encoders import get_encoder
from .optim import AverageMeter, accuracy, update_moving_average, warmup_learning_rate
from .reporting import per_device_report


def _encoder_context(args):
    """A frozen encoder runs without gradients (saves memory); a trained one with."""
    return torch.no_grad() if args.freeze_encoder else contextlib.nullcontext()


def train_epoch(loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler):
    """One epoch of Patch-Mix CL (or CE) training. Returns (mean loss, mean top-1 accuracy)."""
    model.train(), classifier.train(), projector.train()
    if args.freeze_encoder:
        model.eval()  # frozen encoder: no batch-norm statistics update, no dropout
    batch_time, losses, top1 = AverageMeter(), AverageMeter(), AverageMeter()
    cons_loss, cons_agree = AverageMeter(), AverageMeter()
    cons_w = args.consistency_weight * min(1.0, epoch / max(args.consistency_ramp, 1))
    db_scale = get_encoder(args.encoder).db_scale
    end = time.time()
    for idx, (images, labels, _) in enumerate(loader):
        if (
            args.ma_update
        ):  # state before the step, for the moving average (a frozen encoder does not change)
            with torch.no_grad():
                ma_state = [
                    None if m is model and args.freeze_encoder else deepcopy(m.state_dict())
                    for m in (model, classifier, projector)
                ]
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        if args.freq_mixstyle > 0:
            images = freq_mixstyle(images, p=args.freq_mixstyle)
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == "patchmix":
                with _encoder_context(args):
                    mix_images, labels_a, labels_b, lam, _ = model(
                        images, y=labels, patch_mix=True, time_domain=args.time_domain
                    )
                output = classifier(mix_images)
                loss = criterion[1](output, labels_a, labels_b, lam)
            else:
                with _encoder_context(args):
                    features = model(images)
                output = classifier(features)
                loss = criterion[0](output, labels)
                if args.consistency_weight > 0:  # same images under a random smooth per-bin gain must give the same softmax
                    style = random_bin_gain_batch(images, args.consistency_gain_db, db_scale)
                    with _encoder_context(args):
                        out_style = classifier(model(style))
                    js = js_divergence(output, out_style)
                    loss = loss + cons_w * js
                    cons_loss.update(js.item(), bsz)
                    cons_agree.update((output.argmax(1) == out_style.argmax(1)).float().mean().item() * 100, bsz)
            if args.method == "patchmix_cl":
                if args.target_type == "grad_block":
                    proj1 = features.detach().clone()
                elif args.target_type == "grad_flow":
                    proj1 = features
                elif args.target_type == "project_block":
                    proj1 = projector(features).detach().clone()
                else:  # project_flow
                    proj1 = projector(features)
                with _encoder_context(args):
                    mix_images, _, labels_b, lam, index = model(
                        images, y=labels, patch_mix=True, time_domain=args.time_domain
                    )
                proj2 = projector(mix_images)
                loss += args.alpha * criterion[1](proj1, proj2, labels, labels_b, lam, index, args)

        losses.update(loss.item(), bsz)
        top1.update(accuracy(output[:bsz], labels), bsz)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                if ma_state[0] is not None:
                    update_moving_average(args.ma_beta, model, ma_state[0])
                update_moving_average(args.ma_beta, classifier, ma_state[1])
                update_moving_average(args.ma_beta, projector, ma_state[2])

        if (idx + 1) % args.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\tBT {bt.val:.3f} ({bt.avg:.3f})\tloss {l.val:.3f} ({l.avg:.3f})\tAcc@1 {a.val:.3f} ({a.avg:.3f})".format(
                    epoch, idx + 1, len(loader), bt=batch_time, l=losses, a=top1
                )
                + (
                    "\tloss_cons {c.val:.4f} ({c.avg:.4f})\tagree {g.avg:.1f} (w {w:.2f})".format(c=cons_loss, g=cons_agree, w=cons_w)
                    if args.consistency_weight > 0
                    else ""
                )
            )
            sys.stdout.flush()
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(loader, model, classifier, args):
    """Run model and classifier on a loader.

    Returns:
        Tuple (per-device report, preds, softmax probs, labels, device ids).
    """
    model.eval(), classifier.eval()
    preds, probs, ys, devs = [], [], [], []
    for images, labels, devices in loader:
        with torch.cuda.amp.autocast():
            output = classifier(model(images.cuda(non_blocking=True)))
        preds.append(output.argmax(1).cpu().numpy())
        probs.append(torch.softmax(output.float(), 1).cpu().numpy())
        ys.append(labels.numpy())
        devs.append(devices.numpy())
    preds, probs, ys, devs = (np.concatenate(x) for x in (preds, probs, ys, devs))
    return per_device_report(ys, preds, devs, args.n_cls), preds, probs, ys, devs
