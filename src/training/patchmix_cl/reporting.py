"""Training reports: per-device metrics in percent, epoch selection and the CV epoch rule."""
import numpy as np

from src.evaluation.metrics import by_device
from src.processing.icbhi import DEVICES

_PERCENT_KEYS = ('sp', 'se', 'score', 'hs', 'two_cls_se', 'two_cls_score', 'two_cls_hs')  # `recall` (per class) too


def per_device_report(y, pred, device_ids, n_cls=4):
    """Sp, Se, Score, HS, per-class recall and the 2-class view in percent, macro-F1 as a fraction, per device name plus `all`."""
    names = np.array(DEVICES)[np.asarray(device_ids)]
    return {d: {k: (100 * np.asarray(v)).tolist() if k == 'recall' else 100 * v if k in _PERCENT_KEYS else v for k, v in m.items()}
            for d, m in by_device(y, pred, names, n_cls).items()}


def pick_epochs(history):
    """The numbers to log: last epoch, validation-selected epoch, best epoch on test (optimistic).

    Args:
        history: List of dicts with `epoch`, `val` (report dict of the held-out train patients) and `test` (report dict).

    Returns:
        Dict `last`, `cv_selected`, `test_best_optimistic`, each the history entry of that epoch (keys absent when the
        split was not evaluated).
    """
    out = {'last': history[-1]}
    if history[-1].get('val') is not None:
        out['cv_selected'] = max(history, key=lambda h: h['val']['all']['score'])
    if history[-1].get('test') is not None:
        out['test_best_optimistic'] = max(history, key=lambda h: h['test']['all']['score'])
    return out


def mean_cv_curve(fold_curves):
    """Mean over folds of per-epoch validation curves; truncated to the shortest curve.

    Args:
        fold_curves: List (one per fold) of per-epoch values (epoch 1 first).

    Returns:
        (n_epochs,) array.
    """
    n = min(len(c) for c in fold_curves)
    return np.mean([np.asarray(c[:n], float) for c in fold_curves], 0)


def cv_epoch(fold_curves):
    """Registered epoch rule: the epoch (1-based) that maximises the mean CV curve; earliest on ties."""
    return int(np.argmax(mean_cv_curve(fold_curves))) + 1
