"""Stage 1 summary (proposal v9, `docs/prereg-v8.md` + v9 amendment): CV screening table and final table with gate G.

Reads one campaign, `outputs/stage1/<run_id>/<cell>/<unit>/stage1_report.json` (default run: `latest`):
    cell `P0_baseline`, `P1_a1_input`, `P1_a1_input+sc_static`, ...   units `cv<k>` (screening) or `seed<s>` (final refit)

    python -m src.training.patchmix_cl.summary screen [--run RUN_ID] [--folds 3]
        -> screening table; writes <run>/stage1_screen.json (CV epoch and sc_mode per arm, `auscult_trust`, final plan)
    python -m src.training.patchmix_cl.summary final [--run RUN_ID] [--boot 1000]
        -> final table (primary = registered CV epoch; literature column = best epoch on test, labelled optimistic),
           paired differences against P0 and gate G (AuscultTrust vs P0, 95% t-interval lower bound > -1.5 points);
           writes <run>/stage1_final.md

SC arms are screened with both `sc_mode` values (`<arm>` = dynamic, `<arm>+sc_static` = static); the higher mean CV Score
is kept, ties go to dynamic. AuscultTrust = the better of P1 / P1+P3 by mean CV Score (tie: P1).
"""
import argparse
import json
import re

import numpy as np
from scipy import stats

from src.paths import OUTPUTS
from .config import EXP
from .reporting import cv_epoch, mean_cv_curve, per_device_report

BASE, SC_ARMS = 'P0_baseline', ('P1_a1_input', 'P1P3_sc_gain')
STATIC = '+sc_static'
GATE_MARGIN = -1.5  # points; about half the published AST-FT to Patch-Mix CL gap
BENCHMARK_SEEDS = range(5)


def run_dir(run):
    return OUTPUTS / EXP / run


def load_units(root, unit_pattern):
    """{(cell, index): report dict} of finished units whose name matches `unit_pattern` (one group: the index)."""
    out = {}
    for f in root.glob('*/*/stage1_report.json'):
        m = re.fullmatch(unit_pattern, f.parent.name)
        if m:
            out[(f.parent.parent.name, int(m.group(1)))] = json.loads(f.read_text())
    return out


def screen(a):
    root = run_dir(a.run)
    units = load_units(root, r'cv(\d+)')
    cells, rows, missing = sorted({c for c, _ in units}), {}, []
    for c in cells:
        folds = [units.get((c, k)) for k in range(a.folds)]
        if any(f is None for f in folds):
            missing.append(c)
            continue
        curves = [[h['val_score'] for h in f['history']] for f in folds]
        ep = cv_epoch(curves)
        rows[c] = {'cell': c, 'epoch': ep, 'cv_score': float(mean_cv_curve(curves)[ep - 1]),
                   'fold_scores': [float(x[ep - 1]) for x in curves], 'sc_mode': 'static' if c.endswith(STATIC) else 'dynamic'}
    arms = {}
    for c, r in rows.items():  # per arm keep the better sc_mode; ties go to dynamic
        arm = c.removesuffix(STATIC)
        best = arms.get(arm)
        if best is None or r['cv_score'] > best['cv_score'] or (r['cv_score'] == best['cv_score'] and r['sc_mode'] == 'dynamic'):
            arms[arm] = r
    for arm, r in arms.items():
        r['sc_mode'] = r['sc_mode'] if arm in SC_ARMS else None
    ranking = sorted(arms, key=lambda x: -arms[x]['cv_score'])
    sc = [x for x in SC_ARMS if x in arms]
    trust = max(sc, key=lambda x: (arms[x]['cv_score'], x == SC_ARMS[0])) if sc else None  # tie: P1
    print('| rank | arm | cell | sc_mode | CV epoch | mean CV Score | fold Scores |')
    print('|---|---|---|---|---|---|---|')
    for i, x in enumerate(ranking, 1):
        r = arms[x]
        print('| {} | {} | {} | {} | {} | {:.2f} | {} |'.format(i, x, r['cell'], r['sc_mode'] or '-', r['epoch'], r['cv_score'],
                                                              ', '.join('{:.1f}'.format(s) for s in r['fold_scores'])))
    if missing:
        print('\nnot finished (all {} folds needed): {}'.format(a.folds, ', '.join(missing)))
    (root / 'stage1_screen.json').write_text(json.dumps({'arms': arms, 'ranking': ranking, 'auscult_trust': trust}, indent=1))
    print('\nAuscultTrust = {} (better mean CV Score of P1 / P1+P3, tie: P1)'.format(trust))


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


def _t_ci(d, level=0.95):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()), float('nan'), float('nan')
    h = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def _ms(x):
    x = np.asarray(x, float)
    return '{:.2f} ± {:.2f}'.format(x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def _collect(root, runs, arms):
    """Per arm: seeds, per-seed test reports at the registered epoch, test-best score, predictions."""
    res = {}
    for arm, r in arms.items():
        ep, per = r['epoch'], {}
        for s in sorted(s for (c, s) in runs if c == r['cell']):
            rep = runs[(r['cell'], s)]
            z = np.load(root / r['cell'] / 'seed{}'.format(s) / 'test_preds.npz')
            pred = z['preds'][list(z['epochs']).index(ep)]
            per[s] = {'test': rep['fixed'][str(ep)]['test'], 'best': rep['test_best_optimistic']['test']['all']['score'],
                      'pred': pred, 'labels': z['labels'], 'patient': z['patient'], 'device': z['device']}
        res[arm] = {'epoch': ep, 'cell': r['cell'], 'per': per}
    return res


def final(a):
    root = run_dir(a.run)
    scr = json.loads((root / 'stage1_screen.json').read_text())
    res = _collect(root, load_units(root, r'seed(\d+)'), scr['arms'])
    trust = scr['auscult_trust']
    L = ['## Stage 1 final (official 60/40 split, 4-class, refit on all training patients)\n',
         'Primary = epoch registered by 3-fold CV screening (no test selection), seeds 0-4 (benchmark report). Literature '
         'column = best epoch on test, as in Patch-Mix CL / SG-SCL: **optimistic, for comparison only**.\n',
         '| arm | cell | CV epoch | seeds | Sp | Se | Score | HS | macro-F1 | 2-cls Score | literature: best-on-test Score |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, r in res.items():
        per = {s: p for s, p in r['per'].items() if s in BENCHMARK_SEEDS}
        tests = [p['test']['all'] for p in per.values()]
        get = lambda k: [t[k] for t in tests]
        L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            arm + (' (AuscultTrust)' if arm == trust else ''), r['cell'], r['epoch'], len(tests), _ms(get('sp')), _ms(get('se')),
            _ms(get('score')), _ms(get('hs')), _ms(get('macro_f1')), _ms(get('two_cls_score')), _ms([p['best'] for p in per.values()])))
    L.append('\nPublished anchors, same split, 4-class: AST fine-tuning 59.55 ± 0.88; Patch-Mix CL 62.37 ± 0.61; SG-SCL 61.71 ± 1.61 '
             '(all best epoch on test).')

    devs = ['Meditron', 'Litt3200', 'AKGC417L']
    L += ['\n### Per device (primary epoch, seeds 0-4), Score mean ± SD\n', '| arm | ' + ' | '.join(devs) + ' |', '|---|' + '---|' * len(devs)]
    for arm, r in res.items():
        cells = [_ms([p['test'][d]['score'] for s, p in r['per'].items() if s in BENCHMARK_SEEDS and d in p['test']] or [np.nan]) for d in devs]
        L.append('| {} | {} |'.format(arm, ' | '.join(cells)))

    if BASE in res:
        L += ['\n### Paired difference against P0 (same seeds)\n',
              '| arm | seeds | ΔScore mean | 95% t-CI (paired seeds) | 95% patient bootstrap CI |', '|---|---|---|---|---|']
        base, rng = res[BASE]['per'], np.random.default_rng(0)
        gate = None
        for arm, r in res.items():
            if arm == BASE:
                continue
            seeds = sorted(set(r['per']) & set(base))
            m, lo, hi = _t_ci([r['per'][s]['test']['all']['score'] - base[s]['test']['all']['score'] for s in seeds])
            y, pat = r['per'][seeds[0]]['labels'], r['per'][seeds[0]]['patient']
            cv = np.stack([_score_counts(y, r['per'][s]['pred'], pat) for s in seeds])
            cb = np.stack([_score_counts(y, base[s]['pred'], pat) for s in seeds])
            w = rng.multinomial(cv.shape[1], np.ones(cv.shape[1]) / cv.shape[1], size=a.boot)  # (boot, patients)
            diff = (_score_from_counts(np.einsum('bp,spk->bsk', w, cv)) - _score_from_counts(np.einsum('bp,spk->bsk', w, cb))).mean(1)
            q = np.percentile(diff, [2.5, 97.5])
            L.append('| {} | {} | {:+.2f} | [{:.2f}, {:.2f}] | [{:.2f}, {:.2f}] |'.format(arm, len(seeds), m, lo, hi, *q))
            if arm == trust:
                gate = (len(seeds), m, lo)
        if gate:
            n, m, lo = gate
            verdict = 'pass' if np.isfinite(lo) and lo > GATE_MARGIN else 'FAIL (Stage 2 uses P0 + test-time SC)'
            L.append('\nGate G ({} vs P0, {} seeds): ΔScore {:+.2f}, lower bound {:.2f} vs margin {:.1f} -> **{}**.'.format(
                trust, n, m, lo, GATE_MARGIN, verdict))
    txt = '\n'.join(L)
    print(txt)
    (root / 'stage1_final.md').write_text(txt + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['screen', 'final'])
    ap.add_argument('--run', default='latest', help='run_id under outputs/stage1/')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    screen(a) if a.mode == 'screen' else final(a)


if __name__ == '__main__':
    main()
