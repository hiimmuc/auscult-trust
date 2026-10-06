"""Stage 1 summary of proposal v8 (pre-registration `docs/prereg-v8.md` section 3): screening table and final table.

Run folders follow `scripts/run_stage1.sh`:
    save/icbhi_ast_patchmix_cl_<V>_cv<k>/          screening, fold k (seed k), selection=cv, test not evaluated
    save/icbhi_ast_patchmix_cl_<V>_fix_seed<s>/    final refit on all training patients, selection=fixed

Usage:
    .venv/bin/python stage1_summary.py screen [--save save] [--folds 3]
        -> prints the screening table, writes save/stage1_screen.json (CV epoch per variant, ranking, finalists)
    .venv/bin/python stage1_summary.py final [--save save] [--boot 1000]
        -> prints the final table (primary = registered CV epoch; literature column = best epoch on test, labelled),
           paired differences against P0 and the G-base decision; writes save/stage1_final.md

P8 (worst-device selection) is a selection rule, not a training change: it reuses the P0 runs and only picks another
epoch (argmax of the mean worst-device CV curve).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from util import stage1

PREFIX = 'icbhi_ast_patchmix_cl_'
BASE = 'P0_baseline'
P8 = 'P8_worst_device'


def _runs(save, pattern):
    """{(variant, index): report dict} for finished runs whose tag matches `pattern` (groups: variant, index)."""
    out = {}
    for f in glob.glob(os.path.join(save, PREFIX + '*', 'stage1_report.json')):
        tag = os.path.basename(os.path.dirname(f))[len(PREFIX):]
        m = re.fullmatch(pattern, tag)
        if m:
            out[(m.group(1), int(m.group(2)))] = json.load(open(f))
    return out


def screen(a):
    runs = _runs(a.save, r'(.+)_cv(\d+)')
    variants = sorted({v for v, _ in runs})
    if BASE in variants and P8 not in variants:
        variants.append(P8)  # derived from the P0 folds
    rows, missing = [], []
    for v in variants:
        src = BASE if v == P8 else v
        folds = [runs.get((src, k)) for k in range(a.folds)]
        if any(f is None for f in folds):
            missing.append(v)
            continue
        score = [[h['val_score'] for h in f['history']] for f in folds]
        worst = [[h['val_worst'] for h in f['history']] for f in folds]
        ep = stage1.cv_epoch(worst if v == P8 else score)
        ms, mw = stage1.mean_cv_curve(score), stage1.mean_cv_curve(worst)
        rows.append({'variant': v, 'epoch': ep, 'cv_score': float(ms[ep - 1]), 'cv_worst': float(mw[ep - 1]),
                     'fold_scores': [float(c[ep - 1]) for c in score], 'n_epochs': len(ms)})
    rows.sort(key=lambda r: (-r['cv_score'], -r['cv_worst']))  # registered: mean CV Score, ties by worst device
    print('| rank | variant | CV epoch | mean CV Score | fold Scores | mean CV worst-device |')
    print('|---|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        print('| {} | {} | {} | {:.2f} | {} | {:.2f} |'.format(i, r['variant'], r['epoch'], r['cv_score'],
              ', '.join('{:.1f}'.format(x) for x in r['fold_scores']), r['cv_worst']))
    if missing:
        print('\nnot finished (all {} folds needed): {}'.format(a.folds, ', '.join(missing)))
    singles = [r['variant'] for r in rows if r['variant'] not in (BASE, P8) and '+' not in r['variant']]
    finalists = [r['variant'] for r in rows if r['variant'] != BASE][:3] + ([BASE] if any(r['variant'] == BASE for r in rows) else [])
    out = {'epochs': {r['variant']: r['epoch'] for r in rows}, 'ranking': [r['variant'] for r in rows],
           'finalists': finalists, 'p9_candidates': singles[:2], 'rows': rows}
    json.dump(out, open(os.path.join(a.save, 'stage1_screen.json'), 'w'), indent=1)
    print('\nfinalists (top 3 + P0): {}'.format(', '.join(finalists)))
    if len(singles) >= 2:
        print('P9 = {} + {} (screen it with the same folds before the finals)'.format(*singles[:2]))
    print('wrote {}'.format(os.path.join(a.save, 'stage1_screen.json')))


def _score_counts(y, pred, patients):
    """Per-patient counts for the ICBHI Score: (normal n, normal correct, abnormal n, abnormal correct, abn. any-abnormal)."""
    ids = np.unique(patients)
    idx = np.searchsorted(ids, patients)
    n, a = y == 0, y != 0
    c = np.zeros((len(ids), 5))
    np.add.at(c[:, 0], idx, n)
    np.add.at(c[:, 1], idx, n & (pred == 0))
    np.add.at(c[:, 2], idx, a)
    np.add.at(c[:, 3], idx, a & (pred == y))
    np.add.at(c[:, 4], idx, a & (pred > 0))
    return c


def _score_from_counts(c, two_cls=False):
    """ICBHI Score (%) from (summed) counts; works on (..., 5) arrays."""
    sp = c[..., 1] / np.maximum(c[..., 0], 1)
    se = (c[..., 4] if two_cls else c[..., 3]) / np.maximum(c[..., 2], 1)
    return 100 * (sp + se) / 2


def expected_max_normal(k):
    """E[max of k iid standard normals], by numerical integration (k = 5: 1.163; k = 10: 1.539)."""
    x = np.linspace(-8, 8, 20001)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = np.cumsum(phi) * (x[1] - x[0])
    return float(np.sum(x * k * phi * Phi ** (k - 1)) * (x[1] - x[0]))


def _t_ci(d, level=0.95):
    from scipy import stats
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float('nan'), float('nan')
    h = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def _ms(x):
    x = np.asarray(x, float)
    return '{:.2f} ± {:.2f}'.format(x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def final(a):
    scr = json.load(open(os.path.join(a.save, 'stage1_screen.json')))
    runs = _runs(a.save, r'(.+)_fix_seed(\d+)')
    variants = sorted({v for v, _ in runs})
    if BASE in variants and P8 in scr['epochs'] and P8 not in variants:
        variants.append(P8)
    lines, res = [], {}
    for v in variants:
        src = BASE if v == P8 else v
        seeds = sorted(s for (vv, s) in runs if vv == src)
        ep = scr['epochs'].get(v)
        if ep is None:
            print('skip {}: no CV epoch in stage1_screen.json'.format(v))
            continue
        per = []
        for s in seeds:
            rep = runs[(src, s)]
            hist = {h['epoch']: h for h in rep['history']}
            fixed = rep.get('fixed', {}).get(str(ep))  # P8 is read from the P0 runs at its own report epoch
            tb = rep['test_best_optimistic']
            npz = os.path.join(a.save, PREFIX + '{}_fix_seed{}'.format(src, s), 'test_preds.npz')
            z = np.load(npz) if os.path.exists(npz) else None
            pred = z['preds'][list(z['epochs']).index(ep)] if z is not None else None
            per.append({'seed': s, 'fixed': fixed, 'score_at_ep': hist[ep]['test_score'], 'best': tb,
                        'pred': pred, 'z': z})
        res[v] = {'epoch': ep, 'seeds': [p['seed'] for p in per], 'per': per}

    K = len([v for v in res if v != BASE]) + (1 if BASE in res else 0)
    lines.append('## Stage 1 final (official 60/40 split, 4-class, refit on all training patients)\n')
    lines.append('Primary = epoch registered by 3-fold CV screening (no test selection). Literature column = best epoch '
                 'on test, as in Patch-Mix CL / SG-SCL: **optimistic, for comparison only**.\n')
    lines.append('| variant | CV epoch | seeds | Sp | Se | Score | HS | macro-F1 | 2-cls Score | literature: best-on-test Score |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for v, r in res.items():
        by_seed = {}
        for p in r['per']:
            if p['fixed'] is not None:
                by_seed[p['seed']] = p['fixed']['test']
            elif p['pred'] is not None:
                z = p['z']
                by_seed[p['seed']] = stage1.per_device_report(z['labels'], p['pred'], z['device'])
        tests = [t['all'] for t in by_seed.values()]
        get = lambda k: [t[k] for t in tests]
        best = [p['best']['test']['all']['score'] for p in r['per']] if v != P8 else ['n/a']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            v, r['epoch'], len(tests), _ms(get('sp')), _ms(get('se')), _ms(get('score')), _ms(get('hs')),
            _ms(get('macro_f1')), _ms(get('two_cls_score')), _ms(best) if v != P8 else 'n/a (selection rule)'))
        r['scores'] = get('score')
        r['by_seed'] = by_seed

    if res:
        sds = [np.std(r['scores'], ddof=1) for r in res.values() if len(r['scores']) > 1]
        s = (np.mean(sds) / np.sqrt(5)) if sds else float('nan')
        lines.append('\nExpected optimism of choosing the best of K = {} variants on test: s·E[max] = {:.2f} · {:.3f} = '
                     '{:.2f} points (s = mean seed SD / √5).'.format(K, s, expected_max_normal(max(K, 1)), s * expected_max_normal(max(K, 1))))

    # R1: seed ensemble (mean softmax of the seeds at the registered epoch). A separate row, not a single model.
    ens = []
    for v, r in res.items():
        zs = [p['z'] for p in r['per'] if p['z'] is not None and 'probs' in p['z'].files]
        if len(zs) >= 2:
            pr = np.mean([z['probs'][list(z['epochs']).index(r['epoch'])].astype(float) for z in zs], 0)
            m = stage1.per_device_report(zs[0]['labels'], pr.argmax(1), zs[0]['device'])['all']
            ens.append('| {} (ensemble of {} seeds) | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
                v, len(zs), m['sp'], m['se'], m['score'], m['hs'], m['two_cls_score']))
    if ens:
        lines.append('\n### R1: seed ensembles (compare only with ensembles, e.g. Meta-Ensemble 66.49)\n')
        lines.append('| ensemble | Sp | Se | Score | HS | 2-cls Score |')
        lines.append('|---|---|---|---|---|---|')
        lines.extend(ens)

    lines.append('\nPublished anchors, same split, 4-class (`anchors.md`): AST fine-tuning 59.55 ± 0.88; Patch-Mix CL '
                 '62.37 ± 0.61; SG-SCL 61.71 ± 1.61 (all best epoch on test).')

    # Per-device table of the primary epoch (F2)
    lines.append('\n### Per device (primary epoch), Score mean ± SD over seeds\n')
    devs = ['Meditron', 'Litt3200', 'AKGC417L']
    lines.append('| variant | ' + ' | '.join(devs) + ' |')
    lines.append('|---|' + '---|' * len(devs))
    for v, r in res.items():
        cells = []
        for d in devs:
            vals = [t[d]['score'] for t in r['by_seed'].values() if d in t]
            cells.append(_ms(vals) if vals else '-')
        lines.append('| {} | {} |'.format(v, ' | '.join(cells)))

    # Paired differences against P0 (G-base)
    if BASE in res:
        lines.append('\n### Paired difference against P0 (same seeds)\n')
        lines.append('| variant | ΔScore mean | 95% t-CI (paired seeds) | 95% patient bootstrap CI | G-base |')
        lines.append('|---|---|---|---|---|')
        base = {p['seed']: p for p in res[BASE]['per']}
        rng = np.random.default_rng(0)
        chosen = []
        for v, r in res.items():
            if v == BASE:
                continue
            common = [p for p in r['per'] if p['seed'] in base]
            bs = res[BASE]['by_seed']
            d = [t['all']['score'] - bs[sd]['all']['score'] for sd, t in r['by_seed'].items() if sd in bs]
            m, lo, hi = _t_ci(d)
            boot = '-'
            if common and all(p['pred'] is not None and base[p['seed']]['pred'] is not None for p in common):
                z0 = common[0]['z']
                y, pat = z0['labels'], z0['patient']
                cv = np.stack([_score_counts(y, p['pred'], pat) for p in common])            # (seeds, patients, 5)
                cb = np.stack([_score_counts(y, base[p['seed']]['pred'], pat) for p in common])
                npat = cv.shape[1]
                w = rng.multinomial(npat, np.ones(npat) / npat, size=a.boot)                 # (boot, patients)
                sv = _score_from_counts(np.einsum('bp,spk->bsk', w, cv)).mean(1)
                sb = _score_from_counts(np.einsum('bp,spk->bsk', w, cb)).mean(1)
                q = np.percentile(sv - sb, [2.5, 97.5])
                boot = '[{:.2f}, {:.2f}]'.format(*q)
            passed = bool(np.isfinite(lo) and lo > 0)
            if passed:
                chosen.append((m, v))
            lines.append('| {} | {:+.2f} | [{:.2f}, {:.2f}] | {} | {} |'.format(v, m, lo, hi, boot, 'pass' if passed else 'no'))
        pick = max(chosen)[1] if chosen else BASE
        lines.append('\nG-base: Stage 2 model = **{}** at its CV epoch ({}). Rule: best finalist whose paired ΔScore CI '
                     'over P0 lies above 0, otherwise P0.'.format(pick, res[pick]['epoch']))
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(a.save, 'stage1_final.md'), 'w').write(txt + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['screen', 'final'])
    ap.add_argument('--save', default='save')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    screen(a) if a.mode == 'screen' else final(a)


if __name__ == '__main__':
    main()
