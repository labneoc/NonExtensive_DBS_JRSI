"""11 q by band

q of the band-limited signal (1-100, 100-500, 500-2000 Hz) against phase-randomized surrogates; writes both the per-record table and the summary.

Reads  : bcs_T_por_registro_184.csv + RAW_ZIP
Writes : q_por_banda.csv, q_por_banda_resumo.csv
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

import zipfile, io, os, re, time
import numpy as np
import pandas as pd
from scipy import stats, signal as sg

FS = 24000.0

# Load zip file
ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

# Load T values for window sizes
WK = pd.read_csv(DATA / 'bcs_T_por_registro_184.csv')[['file', 'T_bcs_ms', 'flag_bcs']]
# We need win_kmed_ms - load from qresid file if available, otherwise compute
# The trace uses WK from 'qresid_janela_proporcional_T.csv' for win_kmed_ms
# but here we just need WK for q_por_banda.csv which uses win_kmed_ms
# Actually q_por_banda only uses WK[['file','win_kmed_ms']] - we need to recreate this

# Recreate WK with win_kmed_ms from scratch using T_bcs_ms
T_df = pd.read_csv(DATA / 'bcs_T_por_registro_184.csv')
KS_kmed = 50.0 / 4.026  # k=12.4 approx
T_df['win_kmed_ms'] = T_df['T_bcs_ms'].apply(
    lambda T: max(int(round(KS_kmed * T / 1000 * FS)), 24) / FS * 1000
)
WK = T_df[['file', 'win_kmed_ms']]

def ft_surr(x, rng):
    F = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, F.size)
    ph[0] = 0
    if x.size % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size)

def zs(y):
    s = y.std()
    return (y - y.mean()) / s if s > 0 else None

def locz(x, win):
    win = max(int(win), 24)
    w = x.size // win
    if w < 4:
        return None
    y = x[:w * win].reshape(w, win)
    sd = y.std(1, keepdims=True)
    sd[sd == 0] = 1
    return ((y - y.mean(1, keepdims=True)) / sd).ravel()

def bp_sos(x, lo, hi, order=4):
    sos = sg.butter(order, [lo / (FS / 2), hi / (FS / 2)], btype='band', output='sos')
    return sg.sosfiltfilt(sos, x)

BANDS = [('1-100 Hz', 1, 100), ('100-500 Hz', 100, 500), ('500-2000 Hz', 500, 2000)]

# We need qstat_fit_q - this is a skill helper that must be defined
# From the trace it's used as: qstat_fit_q(x)['q'], qstat_fit_q(x)['r2'], qstat_fit_q(x)['beta']
# Reconstruct from context: fits q-Gaussian to amplitude distribution
def qstat_lnq(y, q):
    """Compute ln_q(y) = (y^(1-q) - 1)/(1-q), with ln_1(y) = ln(y)"""
    if abs(q - 1.0) < 1e-9:
        return np.log(y)
    return (y ** (1.0 - q) - 1.0) / (1.0 - q)

def qstat_expq(x, q):
    """q-exponential: [1 + (1-q)*x]^(1/(1-q)) for positive argument, else 0"""
    if abs(q - 1.0) < 1e-9:
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    result = np.where(base > 0, base ** (1.0 / (1.0 - q)), 0.0)
    return result

def qstat_fit_q(x, qmin=1.0, qmax=2.59, step=0.01, nbins=200):
    """Fit q-Gaussian to amplitude distribution of x. Returns dict with q, r2, beta."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return dict(q=np.nan, r2=np.nan, beta=np.nan)
    # Normalize
    xn = (x - x.mean()) / x.std()
    # Build histogram
    lo, hi = np.percentile(xn, 0.5), np.percentile(xn, 99.5)
    bins = np.linspace(lo, hi, nbins + 1)
    counts, edges = np.histogram(xn, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    prob = counts / (counts.sum() * dx)
    ok = prob > 0
    if ok.sum() < 10:
        return dict(q=np.nan, r2=np.nan, beta=np.nan)
    cc = centers[ok]
    pp = prob[ok]
    best = (-np.inf, np.nan, np.nan)
    for qv in np.arange(qmin, qmax + 1e-9, step):
        # ln_q(P) = ln_q(A) - beta * x^2
        y = qstat_lnq(pp, qv)
        if not np.all(np.isfinite(y)):
            continue
        A = np.vstack([cc ** 2, np.ones_like(cc)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if coef[0] >= 0:
            continue
        pred = A @ coef
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else -np.inf
        if r2 > best[0]:
            best = (r2, qv, -coef[0])
    return dict(q=best[1], r2=best[0], beta=best[2])

rg = np.random.default_rng(31337)
rows = []
t0 = time.time()
fq = None
PSDr = []
PSDz = []

for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    xs = ft_surr(x, rg)
    rec = dict(file=nm, patient=int(pat), cls=cls)
    f1 = qstat_fit_q(x)
    rec['q_full'] = f1['q']
    rec['r2_full'] = f1['r2']
    rec['q_full_ft'] = qstat_fit_q(zs(xs))['q']
    for lab, lo, hi in BANDS:
        yr = zs(bp_sos(x, lo, hi))
        ys = zs(bp_sos(xs, lo, hi))
        if yr is None or ys is None:
            continue
        fr = qstat_fit_q(yr)
        rec['q_' + lab] = fr['q']
        rec['r2_' + lab] = fr['r2']
        rec['qft_' + lab] = qstat_fit_q(ys)['q']
        rec['relpow_' + lab] = float(np.var(bp_sos(x, lo, hi)) / np.var(x))
        del yr, ys
    w = WK[WK.file == nm]
    W = int(round(w.win_kmed_ms.iloc[0] / 1000 * FS)) if len(w) else int(0.05 * FS)
    z = locz(x, W)
    if z is not None:
        fq, Pr = sg.welch(x, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2, detrend='constant')
        _, Pz = sg.welch(z, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2, detrend='constant')
        PSDr.append(Pr)
        PSDz.append(Pz)
    rows.append(rec)
    del x, xs, z

BQ = pd.DataFrame(rows)
BQ.to_csv(DATA / 'q_por_banda.csv', index=False, float_format='%.6g')

# Now compute summary
ordem = ['1-100 Hz', '100-500 Hz', '500-2000 Hz']
rows_s = []
for b in ordem + ['full']:
    qr = 'q_full' if b == 'full' else 'q_' + b
    qf = 'q_full_ft' if b == 'full' else 'qft_' + b
    d = BQ[[qr, qf]].dropna()
    dif = d[qr] - d[qf]
    rb = np.random.default_rng(7)
    bs = [np.mean(rb.choice(dif.values, dif.size)) for _ in range(4000)]
    bw = 240000 / FS * 2 * ((100 - 1) if b == '1-100 Hz' else (500 - 100) if b == '100-500 Hz' else (2000 - 500) if b == '500-2000 Hz' else 12000)
    rows_s.append(dict(
        banda=b,
        q_real=d[qr].median(),
        q_surrogate=d[qf].median(),
        excesso=dif.median(),
        lo=np.percentile(bs, 2.5),
        hi=np.percentile(bs, 97.5),
        p=stats.wilcoxon(d[qr], d[qf]).pvalue,
        pot_rel=1.0 if b == 'full' else BQ['relpow_' + b].median(),
        N_efetivo=int(bw),
        pct_acima_53=100 * (d[qr] > 5 / 3).mean()
    ))
S = pd.DataFrame(rows_s)
S.to_csv(DATA / 'q_por_banda_resumo.csv', index=False, float_format='%.5g')
print(S.to_string(index=False, float_format=lambda v: '%.4g' % v))
