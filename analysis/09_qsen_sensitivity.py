"""09 qsen sensitivity

Sensitivity of the median q_sen to sixteen defensible analysis choices (the identifiability check).

Reads  : (raw traces only) + RAW_ZIP
Writes : tripleto_qsen_sensibilidade.csv
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

import zipfile, io, os, time
import numpy as np
import pandas as pd
from scipy import stats

FS = 24000.0

RORD = np.round(np.arange(-5, 5.01, 0.5), 2)
RORD = RORD[RORD != 0]

def mfdfa_F2(x, scales, order=2):
    Y = np.cumsum(x - x.mean()); N = Y.size; out = []
    for s in scales:
        ns = N // s
        if ns < 10: out.append(None); continue
        segs = np.concatenate([Y[:ns*s].reshape(ns, s), Y[N-ns*s:].reshape(ns, s)], 0)
        t = np.arange(s); A = np.vander(t, order+1)
        coef, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)
        resid = segs.T - A @ coef
        out.append((resid**2).mean(0))
    return out

def mfdfa_diag(x, scales, rord=RORD, order=2):
    F2 = mfdfa_F2(x, scales, order); good = [i for i, f in enumerate(F2) if f is not None]
    ls = np.log(np.array(scales)[good]); h = np.empty(len(rord)); r2 = np.empty(len(rord))
    for j, r in enumerate(rord):
        Fr = np.array([(np.mean(F2[i]**(r/2.0)))**(1.0/r) for i in good]); y = np.log(Fr)
        c = np.polyfit(ls, y, 1); h[j] = c[0]
        ss = ((y - y.mean())**2).sum(); r2[j] = 1 - ((y - np.polyval(c, ls))**2).sum() / ss if ss > 0 else np.nan
    tau = rord * h - 1.0; alpha = np.gradient(tau, rord)
    return h, r2, alpha

def q_sen_lyra(alpha):
    amin, amax = float(np.min(alpha)), float(np.max(alpha))
    if amin <= 0 or amax <= 0: return np.nan, amin, amax
    return 1.0 - 1.0 / (1.0/amin - 1.0/amax), amin, amax

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

sub = np.random.default_rng(8).choice(len(npys), 30, replace=False)
grid = []
for smin_ms in [1, 5, 10, 20]:
    for smooth in ['none', 'quartic']:
        for rmax in [3, 5]:
            grid.append((smin_ms, smooth, rmax))

cache = {}
for k in sub:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(npys[k]))), dtype=np.float64)
    cache[k] = {}
    for smin_ms in [1, 5, 10, 20]:
        smin = int(smin_ms / 1000 * FS); smax = min(24000, x.size // 10)
        if smax <= smin * 2: continue
        sc = np.unique(np.logspace(np.log10(smin), np.log10(smax), 16).astype(int))
        h, r2, _ = mfdfa_diag(x, sc); cache[k][smin_ms] = h
    del x

rowsS = []
for (smin_ms, smooth, rmax) in grid:
    vals = []
    msk = np.abs(RORD) <= rmax
    for k in sub:
        if smin_ms not in cache[k]: continue
        h = cache[k][smin_ms][msk]; rr = RORD[msk]
        if smooth == 'none':
            tau = rr * h - 1.0; al = np.gradient(tau, rr)
        else:
            c = np.polyfit(rr, rr * h - 1.0, 4); al = np.polyval(np.polyder(c), rr)
        qs, _, _ = q_sen_lyra(al); vals.append(qs)
    v = np.array(vals, float)
    rowsS.append(dict(s_min_ms=smin_ms, tau_smoothing=smooth, r_max=rmax, n=np.isfinite(v).sum(),
                      q_sen_median=np.nanmedian(v), q_sen_q25=np.nanpercentile(v, 25), q_sen_q75=np.nanpercentile(v, 75)))

SENS = pd.DataFrame(rowsS)
SENS.to_csv(DATA / 'tripleto_qsen_sensibilidade.csv', index=False, float_format='%.5g')
