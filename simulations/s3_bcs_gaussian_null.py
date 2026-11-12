"""s3 bcs gaussian null

T on 60 Gaussian i.i.d. realizations: how often the kappa = 3 criterion returns a finite crossing by chance.

Reads  : (raw traces only)
Writes : bcs_nulo_gaussiano.csv
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

FS = 24000.0

def kappa_b(x, deltas, min_win=8):
    out = np.full(len(deltas), np.nan)
    N = x.size
    for i, n in enumerate(deltas):
        w = N // n
        if w < min_win or n < 5:
            continue
        y = x[:w*n].reshape(w, n)
        y = y - y.mean(1, keepdims=True)
        s2 = y * y
        m2 = s2.mean(1)
        m4 = (s2 * s2).mean(1)
        ok = m2 > 0
        out[i] = (m4[ok] / m2[ok]**2).mean()
    return out

def crossT(d, k, lev=3.0):
    m = np.isfinite(k)
    dd = np.asarray(d)[m]
    kk = np.asarray(k)[m]
    if kk[0] >= lev:
        return dd[0], 'censored_low'
    idx = np.where(kk >= lev)[0]
    if idx.size == 0:
        return np.nan, 'censored_high'
    j = idx[0]
    x0, x1 = np.log10(dd[j-1]), np.log10(dd[j])
    y0, y1 = kk[j-1], kk[j]
    return 10**(x0 + (lev - y0) * (x1 - x0) / (y1 - y0)), 'ok'

dms_raw = np.unique(np.round(np.logspace(np.log10(0.25), np.log10(500), 50) / 1000 * FS).astype(int))
dms = dms_raw[dms_raw >= 5]
deltas_ms = dms / FS * 1000

rg = np.random.default_rng(20261009)
nl = []
for rep in range(60):
    T, f = crossT(deltas_ms, kappa_b(rg.standard_normal(240000), dms))
    nl.append((T, f))

NL = pd.DataFrame(nl, columns=['T_ms', 'flag'])
NL.to_csv(DATA / 'bcs_nulo_gaussiano.csv', index=False, float_format='%.5g')
