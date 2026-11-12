"""s2 generative process battery

Battery of thirteen generative processes at N = 1e5, 60 repetitions each.

Reads  : (raw traces only)
Writes : (see code)
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
from scipy import stats

rr_ = np.random.default_rng(2026)

QGW = np.round(np.arange(0.60, 2.5001, 0.005), 4)

def fitw(x, nb=40, xmax=4.0, qgrid=QGW):
    x = np.asarray(x, float); x = (x-x.mean())/x.std(); x = x[np.abs(x) <= xmax]
    h, e = np.histogram(x, bins=nb, range=(-xmax, xmax), density=True)
    c = 0.5*(e[:-1]+e[1:]); m = h > 0
    y = h[m]/h[m].max(); xx = c[m]**2
    with np.errstate(all="ignore"):
        Y = (np.power(y[None, :], 1-qgrid[:, None])-1)/(1-qgrid)[:, None]
    ok = np.isfinite(Y).all(1)
    Y = Y[ok]; qs = qgrid[ok]
    xm = xx.mean(); Ym = Y.mean(1, keepdims=True)
    sl = ((xx-xm)[None, :]*(Y-Ym)).sum(1)/((xx-xm)**2).sum()
    rr = Y-(sl[:, None]*xx[None, :]+(Ym[:,0]-sl*xm)[:, None])
    r2 = 1-(rr**2).sum(1)/((Y-Ym)**2).sum(1)
    k = int(np.nanargmax(r2)); return qs[k], -sl[k], r2[k]

def ar1(N, phi):
    e = rr_.standard_normal(N); x = np.empty(N); x[0] = e[0]/np.sqrt(1-phi**2)
    for i in range(1, N): x[i] = phi*x[i-1]+e[i]
    return x

def fgn(N, H):
    f = np.fft.rfftfreq(N)[1:]
    S = f**(1-2*H)
    ph = rr_.uniform(0, 2*np.pi, len(f))
    Z = np.concatenate([[0], np.sqrt(S)*np.exp(1j*ph)])
    return np.fft.irfft(Z, N)

def bandpass_gauss(N, lo=0.02, hi=0.45):
    X = np.fft.rfft(rr_.standard_normal(N)); f = np.fft.rfftfreq(N)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, N)

def garch(N, a0=0.05, a1=0.12, b1=0.85):
    e = rr_.standard_normal(N); s2 = np.empty(N); x = np.empty(N)
    s2[0] = a0/(1-a1-b1); x[0] = np.sqrt(s2[0])*e[0]
    for i in range(1, N):
        s2[i] = a0+a1*x[i-1]**2+b1*s2[i-1]; x[i] = np.sqrt(s2[i])*e[i]
    return x

def superstat(N, n_dof, T):
    nb = int(np.ceil(N/T))
    b = rr_.chisquare(n_dof, nb)/n_dof
    s = np.repeat(1/np.sqrt(b), T)[:N]
    return s*rr_.standard_normal(N)

def contam(N, eps, k):
    x = rr_.standard_normal(N)
    m = rr_.random(N) < eps
    x[m] *= k
    return x

def logistic(N, r=4.0):
    x = np.empty(N); x[0] = 0.4
    for i in range(1, N): x[i] = r*x[i-1]*(1-x[i-1])
    return x

# ---- Estudo A: calibração da nula (gaussiana i.i.d.) ----
NULL = {}
for N, R in [(1000, 400), (10000, 400), (100000, 200), (1000000, 60)]:
    NULL[N] = np.array([fitw(rr_.standard_normal(N))[0] for _ in range(R)])
np.savez("nula_q.npz", **{str(k): v for k, v in NULL.items()})
print("nula pronta")

# ---- Estudos B e C: bateria de processos ----
N = 100000; REP = 60
PROC = {
 "Gaussian i.i.d.":            lambda: rr_.standard_normal(N),
 "AR(1), phi=0.9":             lambda: ar1(N, 0.9),
 "AR(1), phi=0.99":            lambda: ar1(N, 0.99),
 "1/f noise (H=0.9)":          lambda: fgn(N, 0.9),
 "band-pass Gaussian":         lambda: bandpass_gauss(N),
 "logistic map (chaos)":       lambda: logistic(N),
 "GARCH(1,1)":                 lambda: garch(N),
 "superstat. n=2.5, T=10^4":   lambda: superstat(N, 2.5, 10000),
 "superstat. n=2.5, T=50":     lambda: superstat(N, 2.5, 50),
 "Student-t, nu=2.46":         lambda: stats.t.rvs(2.46, size=N, random_state=rr_),
 "Student-t, nu=10":           lambda: stats.t.rvs(10, size=N, random_state=rr_),
 "Gauss + 1% outliers (x5)":   lambda: contam(N, 0.01, 5),
 "Gauss + 0.1% outliers (x10)":lambda: contam(N, 0.001, 10),
}
BAT = []
for nome, gen in PROC.items():
    qs = np.array([fitw(gen())[0] for _ in range(REP)])
    BAT.append(dict(proc=nome, q_med=np.median(qs), q_lo=np.percentile(qs, 2.5),
                    q_hi=np.percentile(qs, 97.5), q_sd=qs.std()))
BATd = pd.DataFrame(BAT)
BATd.to_csv(DATA / 'bateria_processos.csv', index=False)
print("bateria pronta")
