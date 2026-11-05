"""Rank the factors behind the cross-domain performance drop (paper: Discussion).

Joins the per-frame errors written by eval_pairwise.py (CLR_DUMP_PERFRAME=1, mode CRDual) with the
per-frame factors from crossdomain_factors.py, then reports for each factor:

  * its Spearman rank correlation with the final translation error over the whole test split;
  * the factor's median inside the worst-K frames against its median over the rest, and the
    separation between the two expressed in robust (median absolute deviation) units.

The second view says whether the worst frames actually differ in
in the way the paper's Discussion describes. Spearman is used rather than Pearson because none of these
factors is expected to act linearly on the error.

    python3 crossdomain_worstk.py [K]
"""
import csv
import os
import sys

import numpy as np

ERRORS = os.environ.get('CLR_PERFRAME_CSV')
FACTORS = os.environ.get('CLR_FACTORS_CSV', '/home/appuser/data/output/CRDual/factors.csv')
FACTOR_NAMES = ('n_points', 'mean_range', 'frac_near', 'brightness', 'contrast')


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    rows = list(csv.DictReader(open(ERRORS)))
    last_it = max(int(r['iteration']) for r in rows)
    err = {r['frame']: (float(r['transl_cm']), float(r['rot_deg']))
           for r in rows if int(r['iteration']) == last_it}
    fac = {r['frame']: r for r in csv.DictReader(open(FACTORS))}

    frames = sorted(set(err) & set(fac))
    print('%d frames joined (final iteration %d)' % (len(frames), last_it))

    t = np.array([err[f][0] for f in frames])
    r = np.array([err[f][1] for f in frames])
    print('translation %.1f cm mean, %.1f median; rotation %.2f deg mean, %.2f median'
          % (t.mean(), np.median(t), r.mean(), np.median(r)))

    order = np.argsort(-t)
    worst = set(frames[i] for i in order[:K])
    print('\nworst %d frames: %.1f to %.1f cm' % (K, t[order[K - 1]], t[order[0]]))

    print('\n%-11s %8s %10s %10s %8s' % ('factor', 'spearman', 'worstK med', 'rest med', 'sep(MAD)'))
    for name in FACTOR_NAMES:
        v = np.array([float(fac[f][name]) for f in frames])
        w = np.array([float(fac[f][name]) for f in frames if f in worst])
        o = np.array([float(fac[f][name]) for f in frames if f not in worst])
        mad = np.median(np.abs(o - np.median(o))) or float('nan')
        print('%-11s %8.3f %10.1f %10.1f %8.2f'
              % (name, spearman(v, t), np.median(w), np.median(o), (np.median(w) - np.median(o)) / mad))

    print('\nworst %d frames, most-erroneous first:' % K)
    print('%-8s %9s %8s %9s %10s %10s' % ('frame', 'transl', 'rot', 'n_points', 'brightness', 'mean_rng'))
    for i in order[:K]:
        f = frames[i]
        print('%-8s %8.1f %8.2f %9s %10.1f %10.1f'
              % (f, t[i], r[i], fac[f]['n_points'], float(fac[f]['brightness']), float(fac[f]['mean_range'])))


if __name__ == '__main__':
    main()
