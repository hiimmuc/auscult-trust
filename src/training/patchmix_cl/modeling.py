"""AST backbone, classifier head, projector, losses and optimiser of one training run."""
from copy import deepcopy

import torch
import torch.nn as nn

from .encoders import get_encoder
from .optim import set_optimizer
from .upstream import PatchMixConLoss, PatchMixLoss, Projector


def build_model(args):
    """Returns (model, classifier, projector, criterion list, optimizer), all on the GPU."""
    model = get_encoder(args.encoder).build(args)
    classifier = deepcopy(model.mlp_head)
    args.proj_dim = model.final_feat_dim  # the contrastive target is the raw feature, so both must have its size
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method == 'patchmix_cl' else nn.Identity()
    criterion = [nn.CrossEntropyLoss().cuda()]
    if args.method == 'patchmix':
        criterion.append(PatchMixLoss(criterion=criterion[0]).cuda())
    if args.method == 'patchmix_cl':
        criterion.append(PatchMixConLoss(temperature=args.temperature).cuda())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda(), classifier.cuda(), projector.cuda()
    if args.freeze_encoder:
        for p in model.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad] + list(classifier.parameters()) + list(projector.parameters())
    return model, classifier, projector, criterion, set_optimizer(args, params)
