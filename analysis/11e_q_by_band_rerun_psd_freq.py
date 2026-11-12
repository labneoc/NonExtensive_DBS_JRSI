"""11e q by band rerun psd freq

Further re-run of step 14; writes the same four outputs.

Reads  : qresid_janela_proporcional_T.csv + RAW_ZIP
Writes : psd_freq.npy
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

from qstat import qstat_fit_q

import zipfile, io, time
import numpy as np
import pandas as pd
from scipy import stats, signal as sg

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

FS = 24000.0

BANDS = [('1-100 Hz', 1, 100), ('100-500 Hz', 100, 500), ('500-2000 Hz', 500, 2000)]

WK = pd.read_csv(DATA / 'qresid_janela_proporcional_T.csv')[['file', 'win_kmed_ms']]

def ft_surr(x, rng):
    F = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, F.size)
    ph[0] = 0
    if x.size % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size)

def bp(x, lo, hi, order=4):
    b, a = sg.butter(order, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return sg.filtfilt(b, a, x)

def locz(x, win):
    win = max(int(win), 24)
    w = x.size // win
    if w < 4:
        return None
    y = x[:w * win].reshape(w, win)
    sd = y.std(1, keepdims=True)
    sd[sd == 0] = 1
    return ((y - y.mean(1, keepdims=True)) / sd).ravel()

def zs(y):
    s = y.std()
    return (y - y.mean()) / s if s > 0 else None

rg = np.random.default_rng(31337)
rows = []
PSDr = []
PSDz = []
t0 = time.time()
fq = None

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
        yr = zs(bp(x, lo, hi))
        ys = zs(bp(xs, lo, hi))
        if yr is None or ys is None:
            continue
        fr = qstat_fit_q(yr)
        rec['q_' + lab] = fr['q']
        rec['r2_' + lab] = fr['r2']
        rec['qft_' + lab] = qstat_fit_q(ys)['q']
        rec['relpow_' + lab] = float(np.var(bp(x, lo, hi)) / np.var(x))
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

np.save(DATA / 'psd_freq.npy', fq)
