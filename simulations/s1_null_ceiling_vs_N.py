"""s1 null ceiling vs N

Null ceiling of the estimator on Gaussian i.i.d. data at six sample sizes, for the grid and likelihood estimators.

Reads  : teto_nulo_q_vs_N.csv
Writes : teto_nulo_q_vs_N.csv
Source : recovered from the execution record of the analysis session; the only edits are
         the path constants below and, where noted in README.md, the import of qstat.py.

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import io
import os
import re
import time
import zipfile
import numpy as np
import pandas as pd
from scipy import stats, signal as sg
from scipy import stats as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, RAW_ZIP

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

# skill:qstat kernel.py (auto-injected on skill load) - need qstat_fit_q
# This function must be available; reconstructing from usage in trace
# qstat_fit_q is a pre-loaded skill helper - we need its implementation
# From inspect.getsource shown in trace, we reconstruct minimal version

def qstat_lnq(p, q):
    """q-logarithm: ln_q(p) = (p^(1-q) - 1)/(1-q)"""
    if abs(q - 1.0) < 1e-9:
        return np.log(p)
    return (p ** (1.0 - q) - 1.0) / (1.0 - q)

def qstat_fit_q(x, qmin=1.0, qmax=2.59, step=0.01):
    """Fit q-Gaussian to x by maximizing R2 of the linearized fit."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    s = x.std()
    if s == 0:
        return dict(q=np.nan, beta=np.nan, r2=np.nan)
    x = x / s
    # build histogram
    n = x.size
    nbins = max(50, min(200, n // 50))
    counts, edges = np.histogram(x, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    centers = centers[ok]; counts = counts[ok]
    best = (-np.inf, np.nan, np.nan)
    for qv in np.arange(qmin, qmax + 1e-9, step):
        y = qstat_lnq(counts, qv)
        if not np.all(np.isfinite(y)):
            continue
        A = np.vstack([centers**2, np.ones_like(centers)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if coef[0] >= 0:
            continue
        pred = A @ coef
        ss = ((y - y.mean())**2).sum()
        r2 = 1 - ((y - pred)**2).sum() / ss if ss > 0 else -np.inf
        if r2 > best[0]:
            beta = -coef[0] * (2 - qv)
            best = (r2, qv, float(beta))
    return dict(q=best[1], beta=best[2], r2=best[0])

NS = [1000, 3000, 10000, 30000, 100000, 300000]
REPS = {1000: 200, 3000: 200, 10000: 200, 30000: 150, 100000: 80, 300000: 40}
rgn = np.random.default_rng(90210)
rows = []
import time
for N in NS:
    R = REPS[N]
    qg = []
    qm = []
    t1 = time.time()
    for _ in range(R):
        y = rgn.standard_normal(N)
        qg.append(qstat_fit_q(y)['q'])
        df, _, sc = tdist.fit(y, floc=0)
        qm.append((df + 3) / (df + 1))
    qg = np.array(qg)
    qm = np.array(qm)
    rows.append(dict(N=N, rep=R,
                     q50_grade=np.median(qg), q95_grade=np.percentile(qg, 95),
                     sd_grade=qg.std(ddof=1),
                     frac_no_piso=float((qg <= 1.0101).mean()),
                     q50_ML=np.median(qm), q95_ML=np.percentile(qm, 95),
                     sd_ML=qm.std(ddof=1),
                     c_ML=(np.percentile(qm, 95) - 1) * np.sqrt(N)))
    pd.DataFrame(rows).to_csv(DATA / 'teto_nulo_q_vs_N.csv', index=False, float_format='%.6g')
    print('  N=%-7d (%3d rep, %4.0f s)  grade q95 %.4f (%.0f%% no piso) | ML q95 %.4f  c=%.2f' % (
        N, R, time.time() - t1, rows[-1]['q95_grade'], 100 * rows[-1]['frac_no_piso'],
        rows[-1]['q95_ML'], rows[-1]['c_ML']))

print('\nconstante c do ajuste q95_ML = 1 + c/sqrt(N): mediana %.3f  (Cramer-Rao preve 2,534)' %
      np.median([r['c_ML'] for r in rows]))

rg2 = np.random.default_rng(13579)
rows2 = []
for N in NS:
    R = 200
    t1 = time.time()
    qg = np.array([qstat_fit_q(rg2.standard_normal(N), qmin=0.60)['q'] for _ in range(R)])
    rows2.append(dict(N=N, rep_grade_ext=R,
                      q50_grade_ext=np.median(qg), q95_grade_ext=np.percentile(qg, 95),
                      sd_grade_ext=qg.std(ddof=1),
                      frac_abaixo_1=float((qg < 1).mean()),
                      c_grade_ext=(np.percentile(qg, 95) - 1) * np.sqrt(N)))
    print('N=%-7d %4.0f s  mediana %.3f  q95 %.4f  sd %.4f  (%.0f%% abaixo de 1)  c=%.1f' % (
        N, time.time() - t1, rows2[-1]['q50_grade_ext'], rows2[-1]['q95_grade_ext'],
        rows2[-1]['sd_grade_ext'], 100 * rows2[-1]['frac_abaixo_1'], rows2[-1]['c_grade_ext']))

g = pd.DataFrame(rows2)
tt = pd.read_csv(DATA / 'teto_nulo_q_vs_N.csv').merge(g, on='N')
tt['q95_CramerRao'] = 1 + 2.534 / np.sqrt(tt.N)
tt.to_csv(DATA / 'teto_nulo_q_vs_N.csv', index=False, float_format='%.6g')
print()
print(tt[['N', 'q50_grade_ext', 'q95_grade_ext', 'q95_ML', 'q95_CramerRao']].to_string(index=False))
