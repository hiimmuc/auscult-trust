"""AuscultTrust: frozen or partially tuned audio encoder, masked-pooling head, conformal layer.

Modules:
    data   `FeatureData`: cached features (`src.features`) of every ICBHI cycle, patient-grouped folds
    model  `Head` (layer mix, fusion, pooling over real frames), `Tail` (last k ViT blocks), `AuscultTrust`
    fit    training loop (resumable), prediction, metrics
    probe  L0 logistic-regression rung
    conformal  `ConformalLayer`
    __main__  CLI: probe | cv-head | cv-ft | select | final | final-conformal  (see the module docstring)
"""
