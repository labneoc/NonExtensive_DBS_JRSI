"""07 mfdfa spectra

Multifractal detrended fluctuation analysis per record: alpha range, spectrum width and q_sen, for records and both surrogate types.

Reads  : (raw traces only) + RAW_ZIP
Writes : tripleto_mfdfa.csv
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

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

RORD = np.round(np.arange(-5, 5.01, 0.5), 2)
RORD = RORD[RORD != 0]

def mfdfa_F2(x, scales, order=2):
    """F^2(s,v) per scale, segments taken from both ends, polynomial detrending."""
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

def mfdfa_spectrum(x, scales, rord=RORD, order=2):
    F2 = mfdfa_F2(x, scales, order)
    good = [i for i, f in enumerate(F2) if f is not None]
    if len(good) < 6: return None
    ls = np.log(np.array(scales)[good])
    h = np.empty(len(rord))
    for j, r in enumerate(rord):
        Fr = np.array([(np.mean(F2[i]**(r/2.0)))**(1.0/r) for i in good])
        h[j] = np.polyfit(ls, np.log(Fr), 1)[0]
    tau = rord * h - 1.0
    alpha = np.gradient(tau, rord)
    fa = rord * alpha - tau
    return dict(h=h, tau=tau, alpha=alpha, f=fa)

def q_sen_lyra(alpha):
    amin, amax = float(np.min(alpha)), float(np.max(alpha))
    if amin <= 0 or amax <= 0: return np.nan, amin, amax
    return 1.0 - 1.0/(1.0/amin - 1.0/amax), amin, amax

def ft_surrogate(x, rng):
    F = np.fft.rfft(x); ph = rng.uniform(0, 2*np.pi, F.size); ph[0] = 0
    if x.size % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=x.size)

rgm = np.random.default_rng(1618); rows = []; ALPHA = {'real': [], 'shuf': [], 'ft': []}; FA = {'real': [], 'shuf': [], 'ft': []}
t0 = time.time()
for k, nm in enumerate(npys):
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    smax = min(24000, x.size//10)
    scales = np.unique(np.logspace(np.log10(24), np.log10(smax), 20).astype(int))
    rec = dict(file=nm, patient=pat, cls=cls, n_scales=len(scales), smax=smax)
    for tag, sig in [('real', x), ('shuf', rgm.permutation(x)), ('ft', ft_surrogate(x, rgm))]:
        sp_ = mfdfa_spectrum(sig, scales)
        if sp_ is None:
            rec['q_sen_'+tag] = np.nan; continue
        qs, amin, amax = q_sen_lyra(sp_['alpha'])
        rec['q_sen_'+tag] = qs; rec['amin_'+tag] = amin; rec['amax_'+tag] = amax; rec['width_'+tag] = amax - amin
        rec['h2_'+tag] = float(np.interp(2.0, RORD, sp_['h']))
        ALPHA[tag].append(sp_['alpha']); FA[tag].append(sp_['f'])
    rows.append(rec); del x
MF = pd.DataFrame(rows)
MF.to_csv(DATA / 'tripleto_mfdfa.csv', index=False, float_format='%.5g')
for t_ in ALPHA:
    np.save('mfdfa_alpha_%s.npy' % t_, np.array(ALPHA[t_]))
    np.save('mfdfa_f_%s.npy' % t_, np.array(FA[t_]))
print('%.0f s | n=%d' % (time.time()-t0, len(MF)))
print(MF[['width_real', 'width_shuf', 'width_ft', 'q_sen_real', 'q_sen_shuf', 'q_sen_ft', 'h2_real']].median().round(4).to_dict())
