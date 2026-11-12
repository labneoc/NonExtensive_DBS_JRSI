"""08 mfdfa smin20

MFDFA restricted to scales above 20 ms, where the negative moments scale acceptably.

Reads  : (raw traces only) + RAW_ZIP
Writes : tripleto_mfdfa_smin20.csv
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

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

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

SMIN20 = int(20/1000*FS); rg20 = np.random.default_rng(2718); rows20 = []; t0 = time.time()
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    smax = min(24000, x.size//10)
    if smax <= SMIN20*2: del x; continue
    sc = np.unique(np.logspace(np.log10(SMIN20), np.log10(smax), 16).astype(int))
    rec = dict(file=nm, patient=pat, cls=cls)
    for tag, sig in [('real', x), ('shuf', rg20.permutation(x))]:
        h, r2, _ = mfdfa_diag(sig, sc); tau = RORD * h - 1.0; al = np.gradient(tau, RORD)
        qs, amin, amax = q_sen_lyra(al)
        rec['q_sen20_'+tag] = qs; rec['width20_'+tag] = amax - amin; rec['amin20_'+tag] = amin
        rec['r2neg_'+tag] = float(np.median(r2[RORD < 0]))
    rows20.append(rec); del x
M20 = pd.DataFrame(rows20)
M20.to_csv(DATA / 'tripleto_mfdfa_smin20.csv', index=False, float_format='%.5g')
print('%.0f s | n=%d | q_sen20 real median %.3f  shuf %.3f | R2neg real %.3f' % (
    time.time()-t0, len(M20), M20.q_sen20_real.median(), M20.q_sen20_shuf.median(), M20.r2neg_real.median()))
print('width20 real %.3f shuf %.3f' % (M20.width20_real.median(), M20.width20_shuf.median()))
