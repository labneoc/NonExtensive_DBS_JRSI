"""s5 triplet reference processes

q_stat, q_sen, q_rel and multifractal width for the reference processes compared against the records in Fig. 7(b).

Reads  : (raw traces only) + RAW_ZIP
Writes : tripleto_bateria_nulos.csv, tripleto_bateria_resumo.csv
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

# ---- helpers that must be defined before use ----

# skill: qstat kernel helpers (assumed available as auto-injected)
# We need qstat_fit_q, qstat_lnq, qstat_expq — define them here

def qstat_lnq(x, q):
    """q-logarithm: ln_q(x) = (x^(1-q)-1)/(1-q), ln_1(x)=ln(x)"""
    x = np.asarray(x, float)
    if abs(q - 1.0) < 1e-9:
        return np.log(x)
    return (x**(1.0 - q) - 1.0) / (1.0 - q)

def qstat_expq(x, q):
    """q-exponential"""
    x = np.asarray(x, float)
    if abs(q - 1.0) < 1e-9:
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    base = np.where(base > 0, base, 0.0)
    return base ** (1.0 / (1.0 - q))

def qstat_fit_q(x, qmin=1.0, qmax=2.59, step=0.01):
    """Fit q-Gaussian to amplitude distribution. Returns dict with q, r2, beta."""
    x = np.asarray(x, float)
    x = (x - x.mean()) / x.std()
    bins = min(200, max(30, x.size // 200))
    counts, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    cc = centers[ok]; pp = counts[ok]
    best = (-np.inf, np.nan, np.nan)
    for qv in np.arange(qmin, qmax + 1e-9, step):
        y = qstat_lnq(pp, qv)
        if not np.all(np.isfinite(y)):
            continue
        A = np.vstack([cc**2, np.ones_like(cc)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if coef[0] >= 0:
            continue
        pred = A @ coef
        ss = ((y - y.mean())**2).sum()
        r2 = 1 - ((y - pred)**2).sum() / ss if ss > 0 else -np.inf
        if r2 > best[0]:
            beta = -coef[0] * (3.0 - qv)
            best = (r2, qv, float(beta))
    return dict(q=best[1], r2=best[0], beta=best[2])

FS = 24000.0
BIN = 24  # 1 ms

def envelope(x, binn=BIN):
    w = x.size // binn
    return np.sqrt((x[:w*binn].reshape(w, binn)**2).mean(1))

def acf(v, maxlag):
    v = v - v.mean(); n = v.size; nf = 1 << int(np.ceil(np.log2(2*n)))
    F = np.fft.rfft(v, nf); c = np.fft.irfft(F*np.conj(F), nf)[:maxlag+1]
    return c / c[0]

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
        ss = ((y - y.mean())**2).sum(); r2[j] = 1 - ((y - np.polyval(c, ls))**2).sum()/ss if ss > 0 else np.nan
    tau = rord*h - 1.0; alpha = np.gradient(tau, rord)
    return h, r2, alpha

def q_sen_lyra(alpha):
    amin, amax = float(np.min(alpha)), float(np.max(alpha))
    if amin <= 0 or amax <= 0: return np.nan, amin, amax
    return 1.0 - 1.0/(1.0/amin - 1.0/amax), amin, amax

tlag = np.arange(1001) * 1.0  # ms

def fit_from(start, C, qmin=0.0, qmax=6.0, cmin=0.02):
    ml = np.isfinite(C).sum() - 1; t = tlag[start:ml+1]; c = C[start:ml+1]
    ok = np.isfinite(c) & (c > cmin); t, c = t[ok], c[ok]
    if t.size < 8: return np.nan, np.nan, np.nan
    best = (-np.inf, np.nan, np.nan)
    for qv in np.arange(qmin, qmax + 1e-9, 0.01):
        y = qstat_lnq(c/c[0], qv)
        if not np.all(np.isfinite(y)): continue
        A = np.vstack([t - t[0], np.ones_like(t)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None); pred = A @ coef
        ss = ((y - y.mean())**2).sum(); r2 = 1 - ((y - pred)**2).sum()/ss if ss > 0 else -np.inf
        if r2 > best[0] and coef[0] < 0: best = (r2, qv, -1/coef[0])
    return best[1], best[2], best[0]

# ---- load data ----
ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

SMIN20 = int(20/1000*FS)
sc20 = np.unique(np.logspace(np.log10(SMIN20), np.log10(24000), 16).astype(int))

N = 240000

def gen(kind, rng):
    if kind == 'gaussian': return rng.standard_normal(N)
    if kind == 'ar1':
        a = np.empty(N); a[0] = 0; e = rng.standard_normal(N)
        for i in range(1, N): a[i] = 0.99*a[i-1] + e[i]
        return a
    if kind == 'pink':
        f = np.fft.rfftfreq(N, 1/FS); S = np.ones_like(f); S[1:] = f[1:]**(-0.8/2)
        ph = rng.uniform(0, 2*np.pi, f.size); ph[0] = 0
        return np.fft.irfft(S*np.exp(1j*ph), n=N)
    if kind == 'superstat':
        blk = int(0.030*FS); nb = N//blk
        b = rng.gamma(1.5, 1/1.5, nb)
        return (rng.standard_normal((nb, blk))/np.sqrt(b)[:,None]).ravel()
    if kind == 'student':
        return rng.standard_t(3, N)

KINDS = ['gaussian', 'ar1', 'pink', 'superstat', 'student']
rgn = np.random.default_rng(161803); NB = []; ACFn = {k: [] for k in KINDS}
t0 = time.time()
for kind in KINDS:
    for rep in range(20):
        y = gen(kind, rgn); y = (y - y.mean())/y.std()
        qst = qstat_fit_q(y)['q']
        e = envelope(y); C = acf(e, min(1000, e.size//4)); ACFn[kind].append(np.pad(C, (0, 1001-C.size), constant_values=np.nan))
        h, r2, _ = mfdfa_diag(y, sc20); tau = RORD*h - 1.0; al = np.gradient(tau, RORD)
        qsn, amin, amax = q_sen_lyra(al)
        NB.append(dict(kind=kind, rep=rep, q_stat=qst, q_sen=qsn, width=amax-amin, r2neg=float(np.median(r2[RORD<0]))))
NBd = pd.DataFrame(NB)
qrel_by = {}
for kind in KINDS:
    m = np.nanmedian(np.array(ACFn[kind]), 0); q_, tau_, r2_ = fit_from(5, m); qrel_by[kind] = (q_, tau_, r2_)
S = NBd.groupby('kind').agg(q_stat=('q_stat','median'), q_sen=('q_sen','median'), width=('width','median')).reindex(KINDS)
S['q_rel'] = [qrel_by[k][0] for k in KINDS]; S['tau_rel_ms'] = [qrel_by[k][1] for k in KINDS]
S['q_rel_req'] = 4*S.q_stat - 3; S['q_sen_req'] = 3 - 2*S.q_stat
NBd.to_csv(DATA / 'tripleto_bateria_nulos.csv', index=False, float_format='%.5g')
S.to_csv(DATA / 'tripleto_bateria_resumo.csv', float_format='%.5g')
print('%.0f s'%(time.time()-t0))
print(S[['q_stat','q_rel','q_rel_req','q_sen','q_sen_req','width','tau_rel_ms']].to_string(float_format=lambda v:'%.3f'%v))
