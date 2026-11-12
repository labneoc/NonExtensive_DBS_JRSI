"""11b q by band rerun

Re-run of step 14 that also writes the median power spectra before and after local standardization.

Reads  : q_por_banda.csv, qresid_janela_proporcional_T.csv + RAW_ZIP
Writes : psd_freq.npy, psd_locz_todos.npy, psd_raw_todos.npy, q_por_banda.csv
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
import numpy as np, pandas as pd
from scipy import signal as sg

FS = 24000.0
ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
names = zf.namelist()
npys = sorted(n for n in names if n.endswith('.npy'))

WK = pd.read_csv(DATA / 'qresid_janela_proporcional_T.csv')[['file', 'win_kmed_ms']]

BANDS = [('1-100 Hz', 1, 100), ('100-500 Hz', 100, 500), ('500-2000 Hz', 500, 2000)]

def ft_surr(x, rng):
    F = np.fft.rfft(x); ph = rng.uniform(0, 2*np.pi, F.size); ph[0] = 0
    if x.size % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=x.size)

def bp(x, lo, hi, order=4):
    b, a = sg.butter(order, [lo/(FS/2), hi/(FS/2)], btype='band')
    return sg.filtfilt(b, a, x)

def bp_sos(x, lo, hi, order=4):
    sos = sg.butter(order, [lo/(FS/2), hi/(FS/2)], btype='band', output='sos')
    return sg.sosfiltfilt(sos, x)

def zs(y):
    s = y.std()
    return (y - y.mean()) / s if s > 0 else None

def locz(x, win):
    win = max(int(win), 24); w = x.size // win
    if w < 4: return None
    y = x[:w*win].reshape(w, win); sd = y.std(1, keepdims=True); sd[sd == 0] = 1
    return ((y - y.mean(1, keepdims=True)) / sd).ravel()

rg = np.random.default_rng(31337); rows = []; PSDr = []; PSDz = []; t0 = time.time(); fq = None
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    xs = ft_surr(x, rg)
    rec = dict(file=nm, patient=int(pat), cls=cls)
    f1 = qstat_fit_q(x); rec['q_full'] = f1['q']; rec['r2_full'] = f1['r2']
    rec['q_full_ft'] = qstat_fit_q(zs(xs))['q']
    for lab, lo, hi in BANDS:
        yr = zs(bp(x, lo, hi)); ys = zs(bp(xs, lo, hi))
        if yr is None or ys is None: continue
        fr = qstat_fit_q(yr); rec['q_'+lab] = fr['q']; rec['r2_'+lab] = fr['r2']
        rec['qft_'+lab] = qstat_fit_q(ys)['q']
        rec['relpow_'+lab] = float(np.var(bp(x, lo, hi)) / np.var(x))
        del yr, ys
    w = WK[WK.file == nm]
    W = int(round(w.win_kmed_ms.iloc[0]/1000*FS)) if len(w) else int(0.05*FS)
    z = locz(x, W)
    if z is not None:
        fq, Pr = sg.welch(x, fs=FS, nperseg=int(FS), noverlap=int(FS)//2, detrend='constant')
        _, Pz = sg.welch(z, fs=FS, nperseg=int(FS), noverlap=int(FS)//2, detrend='constant')
        PSDr.append(Pr); PSDz.append(Pz)
    rows.append(rec); del x, xs, z
BQ = pd.DataFrame(rows); BQ.to_csv(DATA / 'q_por_banda.csv', index=False, float_format='%.6g')
np.save(DATA / 'psd_raw_todos.npy', np.median(np.array(PSDr), 0)); np.save(DATA / 'psd_locz_todos.npy', np.median(np.array(PSDz), 0)); np.save(DATA / 'psd_freq.npy', fq)
print('%.0f s | n=%d' % (time.time()-t0, len(BQ)))
cols = ['q_full', 'q_full_ft'] + [f'{p}_{b}' for b, _, _ in BANDS for p in ('q', 'qft')]
print(BQ[cols].median().round(3).to_dict())
print('potencia relativa por banda:', {b: round(BQ['relpow_'+b].median(), 4) for b, _, _ in BANDS})

rg = np.random.default_rng(31337); add = []; t0 = time.time()
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    xs = ft_surr(x, rg)
    yr = bp_sos(x, 1, 100); ys = bp_sos(xs, 1, 100)
    zr = zs(yr); zq = zs(ys)
    d = dict(file=nm)
    d['q_1-100 Hz'] = qstat_fit_q(zr)['q'] if zr is not None else np.nan
    d['qft_1-100 Hz'] = qstat_fit_q(zq)['q'] if zq is not None else np.nan
    d['relpow_1-100 Hz'] = float(np.var(yr) / np.var(x))
    add.append(d); del x, xs, yr, ys, zr, zq
A = pd.DataFrame(add)
BQ = pd.read_csv(DATA / 'q_por_banda.csv').merge(A, on='file', how='left')
BQ.to_csv(DATA / 'q_por_banda.csv', index=False, float_format='%.6g')
print('%.0f s' % (time.time()-t0))
ordem = ['1-100 Hz', '100-500 Hz', '500-2000 Hz']
print('\nq mediano por banda (real | surrogate de fase | potencia relativa):')
print('  banda completa : %.3f | %.3f | 1.000' % (BQ.q_full.median(), BQ.q_full_ft.median()))
for b in ordem:
    print('  %-14s : %.3f | %.3f | %.4f' % (b, BQ['q_'+b].median(), BQ['qft_'+b].median(), BQ['relpow_'+b].median()))
