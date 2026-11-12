"""14 marginal model selection

Maximum-likelihood comparison of five marginal families (Gaussian, Student-t, generalized Gaussian, alpha-stable, two-component Gaussian mixture) by AIC.

Reads  : stable_alphas.npy, stable_grid.npy, stable_pdfs.npy + RAW_ZIP
Writes : selecao_modelo_marginal.csv
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

import zipfile, io, time
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist, gennorm
from scipy.optimize import minimize_scalar

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

GRID = np.load(DATA / 'stable_grid.npy')
SP = np.load(DATA / 'stable_pdfs.npy')
ALPHAS = np.load(DATA / 'stable_alphas.npy')

LOGP = {a: np.log(SP[i]) for i, a in enumerate(ALPHAS)}

def ll_stable(y):
    best = (-np.inf, np.nan, np.nan)
    for a in ALPHAS:
        lp = LOGP[a]
        def nll(ls):
            s = np.exp(ls)
            return -(np.interp(y/s, GRID, lp, left=-700, right=-700).sum() - y.size * ls)
        r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(5.0)), method='bounded')
        if -r.fun > best[0]:
            best = (-r.fun, a, np.exp(r.x))
    return best

def ll_mix2(y, iters=200, tol=1e-8):
    w = 0.5; s1, s2 = 0.5 * y.std(), 2.0 * y.std(); prev = -np.inf
    for _ in range(iters):
        p1 = w * norm.pdf(y, 0, s1); p2 = (1 - w) * norm.pdf(y, 0, s2); tot = p1 + p2 + 1e-300
        r1 = p1 / tot; ll = np.log(tot).sum()
        if abs(ll - prev) < tol:
            break
        prev = ll; w = r1.mean()
        s1 = np.sqrt((r1 * y**2).sum() / max(r1.sum(), 1e-9)); s2 = np.sqrt(((1 - r1) * y**2).sum() / max((1 - r1).sum(), 1e-9))
        s1 = max(s1, 1e-6); s2 = max(s2, 1e-6)
    return ll, w, s1, s2

K = 23; rows = []; t0 = time.time()
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    y = x[::K]; y = y - y.mean(); n = y.size
    res = {'file': nm, 'n_sub': n}
    res['ll_gauss'] = norm.logpdf(y, 0, y.std()).sum(); res['k_gauss'] = 1
    df, _, sc = tdist.fit(y, floc=0); res['ll_t'] = tdist.logpdf(y, df, 0, sc).sum(); res['k_t'] = 2
    res['df_t'] = df; res['q_from_t'] = (df + 3) / (df + 1)
    b, _, sg_ = gennorm.fit(y, floc=0); res['ll_gn'] = gennorm.logpdf(y, b, 0, sg_).sum(); res['k_gn'] = 2; res['beta_gn'] = b
    ll, a_, s_ = ll_stable(y); res['ll_stable'] = ll; res['k_stable'] = 2; res['alpha_stable'] = a_
    llm, w_, s1_, s2_ = ll_mix2(y); res['ll_mix2'] = llm; res['k_mix2'] = 3; res['w_mix'] = w_; res['s_ratio'] = max(s1_, s2_) / min(s1_, s2_)
    rows.append(res); del x, y
MS = pd.DataFrame(rows)
for m in ['gauss', 't', 'gn', 'stable', 'mix2']:
    MS['AIC_' + m] = 2 * MS['k_' + m] - 2 * MS['ll_' + m]; MS['BIC_' + m] = MS['k_' + m] * np.log(MS.n_sub) - 2 * MS['ll_' + m]
aics = MS[['AIC_gauss', 'AIC_t', 'AIC_gn', 'AIC_stable', 'AIC_mix2']]
MS['vencedor'] = aics.idxmin(axis=1).str.replace('AIC_', '')
MS.to_csv(DATA / 'selecao_modelo_marginal.csv', index=False, float_format='%.8g')
print('%.0f s | n=%d' % (time.time() - t0, len(MS)))
print('\nvencedor por AIC:', MS.vencedor.value_counts().to_dict())
