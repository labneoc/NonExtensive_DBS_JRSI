"""13 firing rate adjustment

Firing-rate confound: record-level regression of q and T on relative high-frequency power, then the paired contrast on the residuals.

Reads  : tripleto_q_184.csv + RAW_ZIP
Writes : espectro_ajuste_confundidor.csv
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
from scipy import stats
import statsmodels.api as sm

FS = 24000.0

# Load tripleto data
TRI = pd.read_csv(DATA / 'tripleto_q_184.csv')

# Load zip file
ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

from scipy import signal as sg

BANDS = [('delta', 1, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta', 13, 30), ('lowgamma', 30, 55),
         ('highgamma', 65, 200), ('f200_500', 200, 500), ('f500_3k', 500, 3000), ('f3k_8k', 3000, 8000)]

rows = []
t0 = time.time()
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    f, P = sg.welch(x, fs=FS, nperseg=int(FS), noverlap=int(FS) // 2, detrend='constant')
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
FQ = FQ.merge(TRI[['file', 'q_stat_resid', 'q_stat_raw', 'T_bcs_ms', 'width20_real']], on='file', how='left')

D2 = FQ.dropna(subset=['q_stat_resid', 'rel_f500_3k']).copy()

out = []
for lab, tgt, cov in [('q_resid bruto', 'q_stat_resid', []),
                      ('q_resid aj 500-3k', 'q_stat_resid', ['rel_f500_3k']),
                      ('q_resid aj ambas', 'q_stat_resid', ['rel_f500_3k', 'rel_f3k_8k']),
                      ('rel500_3k bruto', 'rel_f500_3k', []),
                      ('rel500_3k aj q', 'rel_f500_3k', ['q_stat_resid'])]:
    d = D2.dropna(subset=[tgt] + cov).copy()
    d['adj'] = sm.OLS(d[tgt], sm.add_constant(d[cov])).fit().resid if cov else d[tgt] - d[tgt].mean()
    g = d.groupby(['patient', 'cls'])['adj'].mean().unstack().dropna()
    dd = g['inside'] - g['outside']
    out.append(dict(modelo=lab, delta=dd.mean(), p=stats.wilcoxon(g['inside'], g['outside']).pvalue,
                    dz=dd.mean() / dd.std(ddof=1),
                    consistencia=max((dd > 0).sum(), len(dd) - (dd > 0).sum()) / len(dd), n=len(g)))

AJ = pd.DataFrame(out)
AJ.to_csv(DATA / 'espectro_ajuste_confundidor.csv', index=False, float_format='%.5g')
