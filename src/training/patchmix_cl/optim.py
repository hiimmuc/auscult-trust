"""Optimiser, learning-rate schedule, moving average and running meters."""

import math

import numpy as np
import torch
import torch.optim as optim


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def set_optimizer(args, params):
    if args.optimizer == "sgd":
        return optim.SGD(
            params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
        )
    if args.optimizer == "adam":
        return optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    raise ValueError(args.optimizer)


def update_moving_average(beta, current_model, ma_state):
    """Exponential moving average of `current_model` towards `ma_state` (the state before the step): beta * old + (1 - beta) * new."""
    new_state = {}
    for (k1, new), (k2, old) in zip(current_model.state_dict().items(), ma_state.items()):
        assert k1 == k2
        new_state[k1] = old.data * beta + (1 - beta) * new.data
    current_model.load_state_dict(new_state)
    return current_model


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Top-1 accuracy (%) of a batch."""
    with torch.no_grad():
        return output.argmax(1).eq(target).float().mean().mul_(100.0)
