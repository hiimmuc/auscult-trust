"""Encoder, classifier head, projector, losses and optimiser of one training run."""

from copy import deepcopy

import torch
import torch.nn as nn

from .encoders import get_encoder, image_shape
from .optim import set_optimizer
from .upstream import (
    AngularClassifier,
    AngularMarginLoss,
    DiffTransformerLayer,
    LabelSmoothingLoss,
    MetaCL,
    PAFALoss,
    PafaProjectionHead,
    PatchMixConLoss,
    PatchMixLoss,
    Projector,
)


class DenoisedEncoder(nn.Module):
    """ADD-RSC: the adaptive differential denoising layer runs on the (time, freq) image, the encoder on its output.

    The layer's auxiliary logits (input of the bias denoising loss) are kept in `self.aux` after every forward.
    """

    def __init__(self, encoder, n_freq, args):
        super().__init__()
        self.encoder, self.final_feat_dim, self.mlp_head = encoder, encoder.final_feat_dim, encoder.mlp_head
        self.denoise = DiffTransformerLayer(d_model=n_freq, num_heads=args.denoise_heads, depth=args.denoise_depth)
        self.denoise.classifier = nn.Linear(n_freq, args.n_cls)  # the reference hard-codes 256 bins and 4 classes
        self.aux = None

    def forward(self, x, **kwargs):
        z, self.aux = self.denoise(x[:, 0])
        return self.encoder(z.unsqueeze(1), **kwargs)


def build_model(args):
    """Returns (model, classifier, projector, criterion list, optimizer), all on the GPU."""
    spec = get_encoder(args.encoder)
    model = spec.build(args)
    if args.method == "add_rsc":
        model = DenoisedEncoder(model, image_shape(spec, args)[1], args)
    classifier = deepcopy(model.mlp_head)
    if args.method == "qlung":  # cosine classifier on normalised features, scale 37 as in the reference
        classifier = AngularClassifier(model.final_feat_dim, args.n_cls, s=37)
    args.proj_dim = (
        model.final_feat_dim
    )  # the contrastive target is the raw feature, so both must have its size
    if args.method in ("patchmix_cl", "sg_scl"):
        projector = Projector(model.final_feat_dim, args.proj_dim)
    elif args.method == "ce_pafa":  # PAFA AST setting: LayerNorm MLP head, 768 -> 768 -> 768
        projector = PafaProjectionHead(model.final_feat_dim, None, model.final_feat_dim, attention=False, norm_type="ln")
    else:
        projector = nn.Identity()
    criterion = [nn.CrossEntropyLoss().cuda()]
    if args.method == "patchmix":
        criterion.append(PatchMixLoss(criterion=criterion[0]).cuda())
    if args.method == "patchmix_cl":
        criterion.append(PatchMixConLoss(temperature=args.temperature).cuda())
    if args.method == "ce_pafa":
        criterion.append(PAFALoss().cuda())
    if args.method == "sg_scl":
        criterion.append(MetaCL(temperature=args.temperature).cuda())
    if args.method == "qlung":  # margin scale 15 and base margin 0.5 as in the reference training script
        criterion.append(AngularMarginLoss(model.final_feat_dim, args.n_cls, s=15, m=0.5, gamma=args.qlung_gamma,
                                           m_target=args.qlung_m_target, class_nums=args.class_nums).cuda())
    if args.method == "add_rsc":
        criterion.append(LabelSmoothingLoss(size=args.n_cls).cuda())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda(), classifier.cuda(), projector.cuda()
    if args.freeze_encoder:  # ADD-RSC keeps its denoising layer trainable in front of the frozen encoder
        for p in (model.encoder if args.method == "add_rsc" else model).parameters():
            p.requires_grad = False
    params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(classifier.parameters())
        + list(projector.parameters())
        + [p for c in criterion[1:] for p in c.parameters()]  # QLung's margin-loss weights
    )
    return model, classifier, projector, criterion, set_optimizer(args, params)
