"""15b bootstrap block q B60

Earlier moving-block bootstrap with 60 resamples, kept because Fig. S2(a) was first drawn from it.

Reads  : tripleto_qstat.csv + RAW_ZIP
Writes : bootstrap_blocos_q.csv
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

# skill:qstat kernel.py (auto-injected on skill load)
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma

def qstat_lnq(x, q):
    """Compute the q-logarithm: ln_q(x) = (x^(1-q) - 1) / (1-q)"""
    if abs(q - 1.0) < 1e-10:
        return np.log(x)
    return (np.power(x, 1.0 - q) - 1.0) / (1.0 - q)

def qstat_expq(x, q):
    """Compute the q-exponential: exp_q(x) = [1 + (1-q)*x]_+^(1/(1-q))"""
    if abs(q - 1.0) < 1e-10:
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    base = np.maximum(base, 0.0)
    return np.power(base, 1.0 / (1.0 - q))

def qstat_fit_q(x, qmin=1.0, qmax=2.59, step=0.01):
    """Fit a q-Gaussian to x by grid search over q, returning best q, beta, r2."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    x2 = x ** 2
    best = {'q': np.nan, 'beta': np.nan, 'r2': -np.inf}
    # build empirical CDF-based log-probability via histogram
    n = x.size
    nbins = min(200, max(50, n // 100))
    counts, edges = np.histogram(x, bins=nbins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    if ok.sum() < 8:
        return best
    yc = centers[ok]
    pc = counts[ok]
    log_pc = np.log(pc)
    for q in np.arange(qmin, qmax + 1e-9, step):
        # q-Gaussian: p(x) = A * [1-(1-q)*beta*x^2]^(1/(1-q))
        # log p(x) = log A + 1/(1-q) * log[1-(1-q)*beta*x^2]
        # fit beta by OLS on ln_q(p) vs x^2
        lnq_p = qstat_lnq(pc, q)  # shape (ok.sum(),)
        if not np.all(np.isfinite(lnq_p)):
            continue
        yc2 = yc ** 2
        # OLS: lnq_p = a - beta_eff * yc2
        A_mat = np.vstack([np.ones(yc2.size), yc2]).T
        coef, _, _, _ = np.linalg.lstsq(A_mat, lnq_p, rcond=None)
        if coef[1] >= 0:
            continue
        beta_eff = -coef[1]
        pred = coef[0] + coef[1] * yc2
        ss_res = np.sum((lnq_p - pred) ** 2)
        ss_tot = np.sum((lnq_p - lnq_p.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best['r2']:
            best = {'q': q, 'beta': beta_eff, 'r2': r2}
    return best

import zipfile, io, time
import numpy as np
import pandas as pd
from scipy import stats

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

FS = 24000.0

QS = pd.read_csv(DATA / 'tripleto_qstat.csv')
A_base = QS[['file', 'q_stat_raw', 'r2_raw']].copy()

BLK = 12000   # 500 ms, acima da escala de volatilidade medida
B = 60
rgb = np.random.default_rng(4242)
rows = []
t0 = time.time()
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    nb = x.size // BLK
    if nb < 8:
        rows.append(dict(file=nm, q_lo=np.nan, q_hi=np.nan, q_sd=np.nan, n_blocos=nb))
        del x
        continue
    blocks = x[:nb * BLK].reshape(nb, BLK)
    qs = []
    for _ in range(B):
        idx = rgb.integers(0, nb, nb)
        y = blocks[idx].ravel()
        f = qstat_fit_q(y)
        if np.isfinite(f['q']):
            qs.append(f['q'])
    qs = np.array(qs)
    rows.append(dict(file=nm, q_lo=np.percentile(qs, 2.5), q_hi=np.percentile(qs, 97.5),
                     q_sd=qs.std(ddof=1), n_blocos=nb, n_boot=len(qs)))
    del x, blocks

BB = pd.DataFrame(rows).merge(A_base, on='file', how='left')
BB['largura_IC'] = BB.q_hi - BB.q_lo
BB.to_csv(DATA / 'bootstrap_blocos_q.csv', index=False, float_format='%.6g')
print('%.0f s | blocos de %d ms, %d reamostragens' % (time.time() - t0, BLK / FS * 1000, B))
print('largura do IC95 por registro: mediana %.3f  IQR %.3f-%.3f' % (
    BB.largura_IC.median(), BB.largura_IC.quantile(.25), BB.largura_IC.quantile(.75)))
print('desvio padrao bootstrap de q: mediana %.4f' % BB.q_sd.median())
kp = (BB.q_stat_raw < 2.5999) & (BB.r2_raw >= 0.80)
print('registros cujo IC95 EXCLUI 5/3 por baixo: %d de %d' % (int((BB.loc[kp, 'q_hi'] < 5 / 3).sum()), int(kp.sum())))
print('registros cujo IC95 EXCLUI 1 (isto e, q>1 significativo): %d de %d' % (int((BB.loc[kp, 'q_lo'] > 1.0).sum()), int(kp.sum())))
