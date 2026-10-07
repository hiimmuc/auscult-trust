"""Phase 0 functions moved out of the v9 core (see archive/README.md)."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def closed_fraction(delta_none, delta_corrected, min_delta=0.0):
    """Share of the coverage deficit removed by a correction.

    Args:
        delta_none: Deficit without correction.
        delta_corrected: Deficit with correction.
        min_delta: Deficit below this (e.g. the coverage CI half-width) gives NaN: nothing to close.

    Returns:
        phi, or NaN.
    """
    return 1 - delta_corrected / delta_none if delta_none > min_delta else float("nan")


def decodability(src_emb, tgt_emb, src_groups, tgt_groups, folds=5, seed=0, y_src=None, y_tgt=None):
    """AUC of a domain classifier (logistic regression) predicting source vs target from embeddings.

    Patient-disjoint folds: a patient's rows are never in train and test together. With `y_src` and
    `y_tgt` the AUC is computed within each class and averaged (within-class decodability).

    Args:
        src_emb: (n_s, d) source embeddings.
        tgt_emb: (n_t, d) target embeddings.
        src_groups: (n_s,) patient ids. Patients shared with the target stay in the same fold.
        tgt_groups: (n_t,) patient ids.
        folds: Number of folds.
        seed: Unused by the solver; fixes the order of groups for reproducibility.
        y_src: Optional (n_s,) labels for the within-class variant.
        y_tgt: Optional (n_t,) labels for the within-class variant.

    Returns:
        Float AUC (0.5 = device not decodable).
    """
    X = np.vstack([src_emb, tgt_emb])
    d = np.r_[np.zeros(len(src_emb)), np.ones(len(tgt_emb))]
    g = np.r_[np.asarray(src_groups), np.asarray(tgt_groups)]
    y = None if y_src is None else np.r_[y_src, y_tgt]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    score = np.zeros(len(d))
    for tr, te in GroupKFold(min(folds, len(np.unique(g)))).split(X, d, g):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], d[tr])
        score[te] = clf.decision_function(X[te])
    if y is None:
        return float(roc_auc_score(d, score))
    aucs = [roc_auc_score(d[y == k], score[y == k]) for k in np.unique(y) if len(np.unique(d[y == k])) == 2]
    return float(np.mean(aucs))


def permutation_pvalue(src_emb, tgt_emb, groups_src, groups_tgt, n_perm=200, seed=0, **kw):
    """One-sided permutation test of decodability AUC against 0.5.

    Domain labels are permuted across patients (not rows), so within-patient correlation is kept.

    Args:
        src_emb, tgt_emb, groups_src, groups_tgt: As in `decodability`.
        n_perm: Number of permutations.
        seed: RNG seed.
        **kw: Passed to `decodability`.

    Returns:
        Tuple (observed AUC, p-value).
    """
    obs = decodability(src_emb, tgt_emb, groups_src, groups_tgt, **kw)
    rng = np.random.default_rng(seed)
    X = np.vstack([src_emb, tgt_emb])
    g = np.r_[np.asarray(groups_src), np.asarray(groups_tgt)]
    pats = np.unique(g)
    null = []
    for _ in range(n_perm):
        flip = dict(zip(pats, rng.permutation(len(pats)) < len(pats) / 2))  # random patient -> domain
        dom = np.array([flip[p] for p in g])
        if dom.all() or not dom.any():
            continue
        a, b = np.flatnonzero(~dom), np.flatnonzero(dom)
        null.append(decodability(X[a], X[b], g[a], g[b], **kw))
    return obs, float((1 + sum(v >= obs for v in null)) / (1 + len(null)))


def spearman_block_bootstrap(x, y, blocks, n_boot=1000, seed=0):
    """Spearman rho between per-pair quantities with a bootstrap over blocks (e.g. target device).

    Args:
        x: (m,) e.g. decodability per shift pair.
        y: (m,) e.g. coverage deficit per shift pair.
        blocks: (m,) block id per pair; whole blocks are resampled.
        n_boot: Number of resamples.
        seed: RNG seed.

    Returns:
        Tuple (rho, ci_low, ci_high). The claim needs a CI that excludes 0.
    """
    x, y, blocks = np.asarray(x, float), np.asarray(y, float), np.asarray(blocks)
    rows = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    ids, rng, vals = list(rows), np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([rows[ids[i]] for i in rng.choice(len(ids), len(ids))])
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            vals.append(spearmanr(x[idx], y[idx]).statistic)
    return float(spearmanr(x, y).statistic), *np.percentile(vals, [2.5, 97.5])


def closed_fraction_checked(cov_none, cov_corrected, alpha, n):
    """`closed_fraction` that returns NaN unless the uncorrected deficit exceeds the binomial half-width at nominal.

    Args:
        cov_none: Empirical coverage without correction.
        cov_corrected: Empirical coverage with correction.
        alpha: Miscoverage level.
        n: Test size (rows) behind the coverage estimates.

    Returns:
        phi, or NaN when Delta <= half-width (no detectable deficit to close).
    """
    return closed_fraction(coverage_deficit(cov_none, alpha), coverage_deficit(cov_corrected, alpha),
                           min_delta=binom_ci(1 - alpha, n))


def _pca(X, k):
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:k].T


def decodability_curve(src_emb, tgt_emb, src_groups, tgt_groups, ns=(20, 50, 100, 200), k=8, seed=0, **kw):
    """Unsaturated decodability: AUC of a PCA-`k` probe on `n` rows per domain, for each `n` in `ns`.

    A full-dimensional AUC saturates near 1 on any pair of devices, so the learning curve (does AUC rise with n?)
    is the quantity to read. Rows are drawn per domain without replacement; PCA is fit on the pooled subsample.

    Args:
        src_emb, tgt_emb: (n, d) embeddings.
        src_groups, tgt_groups: Patient ids per row.
        ns: Rows per domain.
        k: PCA components.
        seed: RNG seed.
        **kw: Passed to `decodability`.

    Returns:
        Dict n -> AUC (NaN when a domain has fewer than `n` rows or fewer than 2 patients are drawn).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        if min(len(src_emb), len(tgt_emb)) < n:
            out[n] = float("nan")
            continue
        a, b = rng.choice(len(src_emb), n, replace=False), rng.choice(len(tgt_emb), n, replace=False)
        Z = _pca(np.vstack([src_emb[a], tgt_emb[b]]), k)
        ga, gb = np.asarray(src_groups)[a], np.asarray(tgt_groups)[b]
        out[n] = decodability(Z[:n], Z[n:], ga, gb, **kw) if len(np.unique(np.r_[ga, gb])) > 1 else float("nan")
    return out


def mmd2(X, Y, bandwidth=None):
    """Biased RBF MMD^2 with the median-heuristic bandwidth (pooled pairwise distance)."""
    Z = np.vstack([X, Y])
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    h = bandwidth or np.median(D[D > 0]) or 1.0
    K = np.exp(-D / h)
    n = len(X)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def mmd_permutation(src_emb, tgt_emb, groups_src, groups_tgt, n_perm=200, seed=0):
    """MMD^2 between domains with a patient-level permutation p-value (domain labels permuted across patients).

    Returns:
        Tuple (observed MMD^2, p-value).
    """
    X = np.vstack([src_emb, tgt_emb])
    g = np.r_[np.asarray(groups_src), np.asarray(groups_tgt)]
    obs = mmd2(src_emb, tgt_emb)
    pats, rng, null = np.unique(g), np.random.default_rng(seed), []
    for _ in range(n_perm):
        flip = dict(zip(pats, rng.permutation(len(pats)) < len(pats) / 2))
        dom = np.array([flip[p] for p in g])
        if dom.all() or not dom.any():
            continue
        null.append(mmd2(X[~dom], X[dom]))
    return obs, float((1 + sum(v >= obs for v in null)) / (1 + len(null)))
