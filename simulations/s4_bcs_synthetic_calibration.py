"""s4 bcs synthetic calibration

Calibration of T on synthetic chi-squared superstatistics of known input time scale T0.

Reads  : (raw traces only) + RAW_ZIP
Writes : bcs_calibracao_sintetica.csv
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
import io
import zipfile
import time

FS = 24000.0

def kappa_curve2(x, deltas, min_win=8):
    """Returns per-Delta: biased kappa (BCS), mean unbiased excess kurtosis G2, SE of that mean, n windows."""
    nd=len(deltas)
    kb=np.full(nd,np.nan); g2m=np.full(nd,np.nan); g2se=np.full(nd,np.nan); nw=np.zeros(nd,int)
    N=x.size
    for i,n in enumerate(deltas):
        w=N//n
        if w<min_win or n<5: continue
        y=x[:w*n].reshape(w,n); y=y-y.mean(1,keepdims=True)
        s2=y*y; m2=s2.mean(1); m4=(s2*s2).mean(1)
        ok=m2>0; r=m4[ok]/(m2[ok]**2)
        kb[i]=r.mean()
        G2=((n-1.0)/((n-2.0)*(n-3.0)))*((n+1.0)*(r-3.0)+6.0)
        g2m[i]=G2.mean(); g2se[i]=G2.std(ddof=1)/np.sqrt(G2.size); nw[i]=w
    return kb,g2m,g2se,nw

def cross_T2(dms_ms,y,thr):
    """First upward crossing of y over thr (elementwise), log-interpolated."""
    m=np.isfinite(y)&np.isfinite(thr)
    d=np.asarray(dms_ms,float)[m]; yy=np.asarray(y)[m]; tt=np.asarray(thr)[m]
    if d.size<2: return np.nan,'empty'
    dd=yy-tt
    if dd[0]>=0: return d[0],'censored_low'
    idx=np.where(dd>=0)[0]
    if idx.size==0: return np.nan,'censored_high'
    j=idx[0]; x0,x1=np.log10(d[j-1]),np.log10(d[j]); y0,y1=dd[j-1],dd[j]
    return 10**(x0+(0-y0)*(x1-x0)/(y1-y0)),'ok'

# Delta grid: 0.25 ms .. 500 ms, log-spaced
dms = np.unique(np.round(np.logspace(np.log10(0.25), np.log10(500), 50)/1000*FS).astype(int))
dms = dms[dms >= 5]
deltas_ms = dms/FS*1000

rng = np.random.default_rng(20261009)

# consume RNG state from earlier steps (permutation calls on 184 records)
z = RAW_ZIP
zf = zipfile.ZipFile(z)
names = zf.namelist()
npys = sorted(n for n in names if n.endswith('.npy'))

# Reproduce RNG state: first batch of calibration (cal)
for T0_ms in [2.0, 5.0, 20.0, 50.0, 200.0]:
    blk = int(round(T0_ms/1000*FS))
    for rep in range(20):
        nb = 240000//blk
        beta = rng.gamma(shape=1.5, scale=1/1.5, size=nb)
        y = (rng.standard_normal((nb, blk)) / np.sqrt(beta)[:,None]).ravel()
        y = (y - y.mean())/y.std()

# Reproduce RNG state: full re-run permutations on 184 records (first run)
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    xs = rng.permutation(x)
    del x, xs

# Reproduce RNG state: Gaussian null checks (60 reps)
Z = 2.0
for rep in range(60):
    y = rng.standard_normal(240000)

# Reproduce RNG state: second full re-run permutations on 184 records
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    xs = rng.permutation(x)
    del x, xs

# Now generate cal2
cal2 = []
for T0_ms in [2.0, 5.0, 20.0, 50.0, 200.0]:
    blk = int(round(T0_ms/1000*FS))
    for rep in range(20):
        nb = 240000//blk
        beta = rng.gamma(shape=1.5, scale=1/1.5, size=nb)
        y = (rng.standard_normal((nb, blk)) / np.sqrt(beta)[:,None]).ravel()
        y = (y - y.mean())/y.std()
        kb, g2m, g2se, _ = kappa_curve2(y, dms)
        Tb, fb = cross_T2(deltas_ms, kb, np.full(len(dms), 3.0))
        Tt, ft = cross_T2(deltas_ms, g2m, Z*g2se)
        cal2.append(dict(T0_ms=T0_ms, rep=rep, T_bcs_ms=Tb, T_test_ms=Tt, flag=ft))
cal2 = pd.DataFrame(cal2)

cal2.to_csv(DATA / 'bcs_calibracao_sintetica.csv', index=False, float_format='%.5g')
