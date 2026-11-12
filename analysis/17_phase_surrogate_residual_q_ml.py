"""17 phase surrogate residual q ml

Phase-randomized surrogate control for the residual q with the likelihood estimator at a fixed 50 ms window.

Reads  : (raw traces only) + RAW_ZIP
Writes : residual_q_ft_control.csv
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
from scipy import stats

Z = RAW_ZIP
zf = zipfile.ZipFile(Z)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

W = 1200  # 50 ms at 24 kHz, non-overlapping blocks

def local_z(x, w=W):
    nb = x.size // w
    b = x[:nb * w].reshape(nb, w)
    return ((b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)).ravel()

def ft_surr(x, rng):
    F = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, F.size)
    ph[0] = 0
    if x.size % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size)

def q_t(z, step=4):
    s = z[::step]
    nu, loc, sc = stats.t.fit(s, floc=0)
    nu = min(nu, 1e4)
    return (nu + 3) / (nu + 1)

def neff_ratio(x, maxlag=240):
    v = x - x.mean()
    n = v.size
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(v, nf)
    c = np.fft.irfft(F * np.conj(F), nf)[:maxlag + 1]
    r = c / c[0]
    return 1.0 + 2.0 * np.sum(r[1:] ** 2)

rng = np.random.default_rng(2026)
rows = []
t0 = time.time()
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    zr = local_z(x)
    xs = ft_surr(x, rng)
    zs = local_z(xs)
    rows.append(dict(patient=pat, cls=cls, file=nm,
        q_res_real=q_t(zr), q_res_ft=q_t(zs),
        kex_res_real=stats.kurtosis(zr), kex_res_ft=stats.kurtosis(zs),
        kex_ft_global=stats.kurtosis(xs), infl=neff_ratio(x)))
    del x, xs

R = pd.DataFrame(rows)
R.to_csv(DATA / 'residual_q_ft_control.csv', index=False)
print('%.0f s, n=%d' % (time.time() - t0, len(R)))
