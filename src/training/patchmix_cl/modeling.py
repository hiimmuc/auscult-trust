"""AST backbone, classifier head, projector, losses and optimiser of a Stage 1 run."""
from copy import deepcopy

import torch
import torch.nn as nn

from .dataset import IMG_MEL, IMG_TIME
from .losses import PatchMixConLoss
from .models import ASTModel, Projector
from .optim import set_optimizer


def build_model(args):
    """Returns (model, classifier, projector, criterion list, optimizer), all on the GPU."""
    model = ASTModel(input_fdim=IMG_TIME, input_tdim=IMG_MEL, label_dim=args.n_cls, imagenet_pretrain=args.from_sl_official,
                     audioset_pretrain=args.audioset_pretrained, mix_beta=args.mix_beta)
    classifier = deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method == 'patchmix_cl' else nn.Identity()
    criterion = [nn.CrossEntropyLoss().cuda()]
    if args.method == 'patchmix_cl':
        criterion.append(PatchMixConLoss(temperature=args.temperature).cuda())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda(), classifier.cuda(), projector.cuda()
    params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    return model, classifier, projector, criterion, set_optimizer(args, params)
