"""18 headtohead features

Assembles the descriptor table for the within-patient head-to-head comparison.

Reads  : espectro_184.csv, qresid_janela_proporcional_T.csv, tripleto_q_184.csv + RAW_ZIP
Writes : headtohead_features.csv
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
import zipfile
import io
import time
from scipy import stats, signal as sg
import statsmodels.api as sm

FS = 24000.0

# Load dependency artifacts
FQ = pd.read_csv(DATA / 'espectro_184.csv')
TRI = pd.read_csv(DATA / 'tripleto_q_184.csv')
WRk = pd.read_csv(DATA / 'qresid_janela_proporcional_T.csv')[['file', 'T_bcs_ms', 'win_kmed_ms']]

# Load zip file
ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

# BCS kappa function
dms_orig = np.unique(np.round(np.logspace(np.log10(0.25), np.log10(500), 50) / 1000 * FS).astype(int))
dms_orig = dms_orig[dms_orig >= 5]
dms = dms_orig
deltas_ms = dms / FS * 1000

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

def locz(x, win):
    win = max(int(win), 24)
    w = x.size // win
    if w < 4:
        return None
    y = x[:w*win].reshape(w, win)
    sd = y.std(1, keepdims=True)
    sd[sd == 0] = 1
    return ((y - y.mean(1, keepdims=True)) / sd).ravel()

SMIN20 = int(20 / 1000 * FS)
RORD = np.round(np.arange(-5, 5.01, 0.5), 2)
RORD = RORD[RORD != 0]

def mfdfa_F2(x, scales, order=2):
    Y = np.cumsum(x - x.mean())
    N = Y.size
    out = []
    for s in scales:
        ns = N // s
        if ns < 10:
            out.append(None)
            continue
        segs = np.concatenate([Y[:ns*s].reshape(ns, s), Y[N-ns*s:].reshape(ns, s)], 0)
        t = np.arange(s)
        A = np.vander(t, order + 1)
        coef, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)
        resid = segs.T - A @ coef
        out.append((resid**2).mean(0))
    return out

def mfdfa_diag(x, scales, rord=RORD, order=2):
    F2 = mfdfa_F2(x, scales, order)
    good = [i for i, f in enumerate(F2) if f is not None]
    ls = np.log(np.array(scales)[good])
    h = np.empty(len(rord))
    r2 = np.empty(len(rord))
    for j, r in enumerate(rord):
        Fr = np.array([(np.mean(F2[i]**(r/2.0)))**(1.0/r) for i in good])
        y = np.log(Fr)
        c = np.polyfit(ls, y, 1)
        h[j] = c[0]
        ss = ((y - y.mean())**2).sum()
        r2[j] = 1 - ((y - np.polyval(c, ls))**2).sum() / ss if ss > 0 else np.nan
    tau = rord * h - 1.0
    alpha = np.gradient(tau, rord)
    return h, r2, alpha

# Cost sampling
COST = {k: [] for k in ['welch', 'kurt_raw', 'q_raw', 'T_bcs', 'locz', 'q_resid', 'mfdfa']}
rows = []
t0 = time.time()
for i, nm in enumerate(npys):
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    r = WRk[WRk.file == nm]
    W = int(round(r.win_kmed_ms.iloc[0] / 1000 * FS)) if len(r) else int(0.05 * FS)
    import time as _time
    tt = _time.perf_counter(); k_raw = float(((x - x.mean())**4).mean() / ((x - x.mean())**2).mean()**2); COST['kurt_raw'].append(_time.perf_counter() - tt)
    tt = _time.perf_counter(); z = locz(x, W); COST['locz'].append(_time.perf_counter() - tt)
    k_res = float((z**4).mean() / (z**2).mean()**2) if z is not None else np.nan
    if i < 12:
        tt = _time.perf_counter(); sg.welch(x, fs=FS, nperseg=int(FS), noverlap=int(FS)//2); COST['welch'].append(_time.perf_counter() - tt)
        tt = _time.perf_counter(); kappa_b(x, dms); COST['T_bcs'].append(_time.perf_counter() - tt)
        sc = np.unique(np.logspace(np.log10(SMIN20), np.log10(min(24000, x.size//10)), 16).astype(int))
        tt = _time.perf_counter(); mfdfa_diag(x, sc); COST['mfdfa'].append(_time.perf_counter() - tt)
    rows.append(dict(file=nm, patient=pat, cls=cls, kurt_raw=k_raw, kurt_resid=k_res))
    del x

KR = pd.DataFrame(rows)
H = (FQ.merge(KR[['file', 'kurt_raw', 'kurt_resid']], on='file', how='left')
       .merge(TRI[['file', 'width20_real', 'q_sen20_real']], on='file', how='left'))

# Fix duplicate column if needed
if 'width20_real_x' in H.columns and 'width20_real_y' in H.columns:
    H = H.rename(columns={'width20_real_y': 'width20_real'}).drop(columns=['width20_real_x'])

H.to_csv(DATA / 'headtohead_features.csv', index=False, float_format='%.6g')
print('%.0f s | n=%d' % (time.time() - t0, len(H)))
