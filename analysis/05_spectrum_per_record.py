"""05 spectrum per record

Welch power spectrum per record and band powers from delta to 3-8 kHz.

Reads  : (raw traces only) + RAW_ZIP
Writes : espectro_184.csv
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
from scipy import signal as sg

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

FS = 24000.0

BANDS = [('delta', 1, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta', 13, 30), ('lowgamma', 30, 55),
         ('highgamma', 65, 200), ('f200_500', 200, 500), ('f500_3k', 500, 3000), ('f3k_8k', 3000, 8000)]
rows = []
t0 = time.time()
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    f, P = sg.welch(x, fs=FS, nperseg=int(FS), noverlap=int(FS)//2, detrend='constant')
    tot = np.trapezoid(P[(f >= 1) & (f <= 8000)], f[(f >= 1) & (f <= 8000)])
    rec = dict(file=nm, patient=pat, cls=cls, total_power_1_8k=tot)
    for b, lo, hi in BANDS:
        m = (f >= lo) & (f <= hi)
        rec['rel_' + b] = np.trapezoid(P[m], f[m]) / tot
    ms = (f >= 200) & (f <= 3000)
    rec['slope_200_3k'] = np.polyfit(np.log10(f[ms]), np.log10(P[ms]), 1)[0]
    ml = (f >= 1) & (f <= 55)
    rec['slope_1_55'] = np.polyfit(np.log10(f[ml]), np.log10(P[ml]), 1)[0]
    rows.append(rec)
    del x, P
FQ = pd.DataFrame(rows)
FQ.to_csv(DATA / 'espectro_184.csv', index=False, float_format='%.6g')
print('%.0f s | total power after global z-score: median %.4f, IQR %.4f-%.4f (CV=%.3f)' % (
    time.time() - t0, FQ.total_power_1_8k.median(), FQ.total_power_1_8k.quantile(.25), FQ.total_power_1_8k.quantile(.75),
    FQ.total_power_1_8k.std() / FQ.total_power_1_8k.mean()))
