"""Stage 1 helpers of proposal v8: patient-level validation split, three-epoch reporting, per-device metrics,
and the device-motivated variants P1-P8 (flags in main.py, configs in ../../configs/stage1_variants/).

Pure numpy/torch functions with no dataset dependency, so they are unit-tested from the repo's `tests/`.
Image layout used by the dataset: (time, mel, 1) numpy, or (B, 1, time, mel) tensors after `ToTensor`.
"""
import os
import sys

import numpy as np

DEVICES = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L']  # dataset device ids 0..3
META_DEVICE_IDX = 6  # position of the device id inside the metadata vector (age, sex, BMI, weight, height, site, device)
# Gain-invariant handcrafted dims of `src.features_handcrafted` (37 means then 37 SDs): the SD over frames of the 20 MFCC.
# A stationary per-frequency gain adds a constant to the log-mel, hence a constant to every MFCC, so the frame SD is
# unchanged while the MFCC means, centroid, bandwidth, chroma, RMS and rolloff all move. [Interpretation]
HC_INVARIANT = np.arange(37, 57)


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def import_src(module):
    """Import a module of the main repo (`src.*`) from the Patch-Mix venv."""
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib
    return importlib.import_module(module)


def patient_val_split(patients, frac=0.2, seed=0):
    """Hold out whole patients of the training set as a validation set.

    Args:
        patients: (n,) patient id per training cycle.
        frac: Fraction of training patients for validation.
        seed: RNG seed (fixed per run so that all variants of one seed use the same split).

    Returns:
        Tuple (train_idx, val_idx), disjoint by patient.
    """
    patients = np.asarray(patients)
    ids = sorted(set(patients.tolist()))
    rng = np.random.default_rng(10_000 + seed)
    rng.shuffle(ids)
    val = set(ids[:max(1, int(round(frac * len(ids))))])
    is_val = np.array([p in val for p in patients.tolist()])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def patient_folds(patients, devices, n_folds=3, seed=12345):
    """Assign every training patient to one of `n_folds` grouped-CV folds, stratified by the patient's main device.

    The assignment depends only on the patient list and `seed`, not on the run seed, so every variant and every seed
    sees the same folds (paired screening). Devices with few patients (Litt3200 in the official train split) are spread
    over the folds instead of landing in one.

    Args:
        patients: (n,) patient id per training cycle.
        devices: (n,) device id per training cycle.
        n_folds: Number of folds.
        seed: RNG seed of the assignment (fixed by default; do not tie it to the run seed).

    Returns:
        Dict patient id -> fold index in [0, n_folds).
    """
    patients, devices = np.asarray(patients), np.asarray(devices)
    main_dev = {}
    for p in sorted(set(patients.tolist())):
        vals, counts = np.unique(devices[patients == p], return_counts=True)
        main_dev[p] = int(vals[np.argmax(counts)])
    rng = np.random.default_rng(seed)
    fold, k = {}, 0
    for d in sorted(set(main_dev.values())):
        ids = [p for p in sorted(main_dev) if main_dev[p] == d]
        rng.shuffle(ids)
        for p in ids:  # round-robin across devices, so fold sizes stay balanced overall
            fold[p] = k % n_folds
            k += 1
    return fold


def cv_split(patients, devices, n_folds, fold):
    """(train_idx, val_idx) of grouped-CV fold `fold`; disjoint by patient. See `patient_folds`."""
    assign = patient_folds(patients, devices, n_folds)
    is_val = np.array([assign[p] == fold for p in np.asarray(patients).tolist()])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _f1(y, pred, k):
    tp = ((pred == k) & (y == k)).sum()
    p, r = tp / max((pred == k).sum(), 1), tp / max((y == k).sum(), 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def metrics_from_preds(y, pred, n_cls=4):
    """Sp, Se, Score, HS (all in %) and macro-F1 from 4-class predictions; plus the 2-class view of the same predictions.

    The 2-class numbers (`two_cls_*`) are the 4-class model's predictions collapsed to normal/abnormal, which is what
    `--two_cls_eval` does in validate(): it is not a separately trained 2-class model.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    n, a = y == 0, y != 0
    sp = 100 * (pred[n] == 0).mean() if n.any() else 0.0
    se = 100 * (pred[a] == y[a]).mean() if a.any() else 0.0
    sp2 = sp
    se2 = 100 * (pred[a] > 0).mean() if a.any() else 0.0
    hm = lambda s, e: 2 * s * e / (s + e) if s + e else 0.0
    return {'n': int(len(y)), 'sp': float(sp), 'se': float(se), 'score': float((sp + se) / 2), 'hs': float(hm(sp, se)),
            'macro_f1': float(np.mean([_f1(y, pred, k) for k in range(n_cls)])),
            'two_cls_se': float(se2), 'two_cls_score': float((sp2 + se2) / 2), 'two_cls_hs': float(hm(sp2, se2))}


def per_device_report(y, pred, device_ids, n_cls=4):
    """`metrics_from_preds` per device name, plus `all`."""
    y, pred, device_ids = np.asarray(y), np.asarray(pred), np.asarray(device_ids)
    out = {'all': metrics_from_preds(y, pred, n_cls)}
    for i, name in enumerate(DEVICES):
        m = device_ids == i
        if m.any():
            out[name] = metrics_from_preds(y[m], pred[m], n_cls)
    return out


def worst_device_score(report):
    """Minimum Score over devices that have data (P8 criterion). `all` is excluded."""
    return min(v['score'] for k, v in report.items() if k != 'all')


def pick_epochs(history, select_metric='score'):
    """The three numbers to log: last epoch, validation-selected epoch, best epoch on test (optimistic).

    Args:
        history: List of dicts with `epoch`, `val` (report dict of the held-out train patients) and `test` (report dict).
        select_metric: 'score' (pooled val Score) or 'worst_device' (P8, min per-device val Score).

    Returns:
        Dict `last`, `cv_selected`, `test_best_optimistic`, each the history entry of that epoch.
    """
    out = {'last': history[-1]}
    if history[-1].get('val') is not None:
        key = (lambda h: h['val']['all']['score']) if select_metric == 'score' else (lambda h: worst_device_score(h['val']))
        out['cv_selected'] = max(history, key=key)
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


def random_bin_gain_image(image, max_db, rng):
    """P3: smooth random per-mel-bin gain on a (time, mel, 1) dataset image, in normalised units.

    `max_db` is in dB of the raw log-mel; the dataset normalises fbank by 2 * std (std = 4.5689974), so the gain is rescaled.
    """
    rbg = import_src('src.shift.correction').random_bin_gain
    out = rbg(image[..., 0], max_db / (2 * 4.5689974), rng=rng)
    return out[..., None].astype(image.dtype)


def freq_mixstyle(x, p=0.5, alpha=0.1, eps=1e-6):
    """P4: Freq-MixStyle. Mix per-frequency-bin statistics (over time) between random samples of the batch.

    Args:
        x: (B, 1, time, mel) tensor.
        p: Probability of applying the mix to the batch.
        alpha: Beta(alpha, alpha) concentration.
        eps: Variance floor.

    Returns:
        Tensor of the same shape.
    """
    import torch
    if torch.rand(1).item() > p or x.shape[0] < 2:
        return x
    mu, var = x.mean(2, keepdim=True), x.var(2, keepdim=True)
    sig = (var + eps).sqrt()
    xn = (x - mu) / sig
    lam = torch.distributions.Beta(alpha, alpha).sample((x.shape[0], 1, 1, 1)).to(x.device, x.dtype)
    perm = torch.randperm(x.shape[0], device=x.device)
    return xn * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])


def device_mean_spectra(waves_by_device, n_fft=1024, hop=512):
    """Mean |STFT| per device (P1 source/target statistics). Delegates to `src.shift.correction`."""
    ms = import_src('src.shift.correction').mean_spectrum
    return {d: ms(w, n_fft, hop) for d, w in waves_by_device.items() if len(w)}


def a1_coefficients(device_spectra, reference_spectra=None, reference='arithmetic', sc_mode='dynamic',
                     limit_freq_low=50.0, limit_freq_high=2000.0, limit_freq_diff=20.0, sr=16000, n_fft=1024):
    """P1 coefficients c = s_ref / s_device per device in `device_spectra`.

    The reference comes from `reference_spectra` (the training devices; default `device_spectra` itself). Train clips use
    their own (source) device spectra, test clips their own (target) spectra, both against the same train reference.

    `sc_mode`, `limit_freq_low`, `limit_freq_high`, `limit_freq_diff`, `sr`, `n_fft`: registered SC bound
    (prereg App. B2), matching `src.shift.correction.limit_coefficients`. "dynamic" clips every bin to
    +-`limit_freq_diff` dB; "static" leaves the [`limit_freq_low`, `limit_freq_high`] Hz band unclipped and
    zeroes the correction (coefficient = 1) outside it.
    """
    assert reference in ('arithmetic', 'geometric'), reference
    assert sc_mode in ('dynamic', 'static'), sc_mode
    stack = np.array(list((reference_spectra or device_spectra).values()))
    s_ref = stack.mean(0) if reference == 'arithmetic' else np.exp(np.log(np.maximum(stack, 1e-8)).mean(0))
    raw = {d: s_ref / np.maximum(s, 1e-8) for d, s in device_spectra.items()}
    if sc_mode == 'dynamic':
        return {d: 10 ** (np.clip(20 * np.log10(np.maximum(c, 1e-8)), -limit_freq_diff, limit_freq_diff) / 20)
                for d, c in raw.items()}
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= limit_freq_low) & (freqs <= limit_freq_high)
    return {d: np.where(band, c, 1.0) for d, c in raw.items()}


def device_bin_norm(images, device_ids, ref_stats=None):
    """P2: per-device, per-mel-bin standardisation of (time, mel, 1) images, then rescale to the reference statistics.

    Args:
        images: List of (time, mel, 1) arrays (modified copy returned).
        device_ids: (n,) device id per image.
        ref_stats: (mean, sd), each (mel,), the reference (train, all devices). Computed from `images` when None.

    Returns:
        Tuple (new images, ref_stats). Each device's own mean/SD per mel bin is removed, so a per-bin offset (gain) is cancelled.
    """
    device_ids = np.asarray(device_ids)
    flat = lambda idx: np.concatenate([images[i][..., 0] for i in idx], 0)
    if ref_stats is None:
        allx = flat(range(len(images)))
        ref_stats = (allx.mean(0), allx.std(0) + 1e-6)
    out = [None] * len(images)
    for d in np.unique(device_ids):
        idx = np.flatnonzero(device_ids == d)
        x = flat(idx)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        for i in idx:
            out[i] = (((images[i][..., 0] - mu) / sd * ref_stats[1] + ref_stats[0])[..., None]).astype(images[i].dtype)
    return out, ref_stats
