"""12 spectral contrast stn

Paired inside-outside contrast of the spectral descriptors.

Reads  : espectro_184.csv, tripleto_q_184.csv + RAW_ZIP
Writes : espectro_184.csv, espectro_contraste_stn.csv
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
from scipy import signal as sg
from scipy import stats

def apply_figure_style(*args, **kwargs):
    pass

def panel_letter(*args, **kwargs):
    pass

FS = 24000.0

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

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
FQ.to_csv(DATA / 'espectro_184.csv', index=False, float_format='%.6g')
print('%.0f s | total power after global z-score: median %.4f, IQR %.4f-%.4f (CV=%.3f)' % (
    time.time() - t0, FQ.total_power_1_8k.median(), FQ.total_power_1_8k.quantile(.25),
    FQ.total_power_1_8k.quantile(.75),
    FQ.total_power_1_8k.std() / FQ.total_power_1_8k.mean()))

TRI = pd.read_csv(DATA / 'tripleto_q_184.csv')

FQ = pd.read_csv(DATA / 'espectro_184.csv')
FQ = FQ.merge(TRI[['file', 'q_stat_resid', 'q_stat_raw', 'T_bcs_ms', 'width20_real']], on='file', how='left')

def icc1(df, col):
    d = df[['patient', col]].dropna()
    g = d.groupby('patient')[col]
    k = g.count()
    if (k < 2).all():
        return np.nan, np.nan
    grand = d[col].mean()
    n_ = k.mean()
    msb = (k * (g.mean() - grand) ** 2).sum() / (len(k) - 1)
    msw = ((g.apply(lambda v: ((v - v.mean()) ** 2).sum())).sum()) / (len(d) - len(k))
    return (msb - msw) / (msb + (n_ - 1) * msw), d[col].std()

feats = ['rel_' + b for b, _, _ in BANDS] + ['slope_200_3k', 'slope_1_55']
res = []
for c in feats + ['q_stat_resid', 'T_bcs_ms']:
    g = FQ.groupby(['patient', 'cls'])[c].mean().unstack().dropna()
    d = g['inside'] - g['outside']
    npos = int((d > 0).sum())
    p = stats.wilcoxon(g['inside'], g['outside']).pvalue
    icc = icc1(FQ, c)[0]
    res.append(dict(feature=c, n=len(g), delta=d.mean(), p=p, dz=d.mean() / d.std(ddof=1),
                    frac_consistent=max(npos, len(d) - npos) / len(d), ICC1=icc))
SP = pd.DataFrame(res).sort_values('p')
SP['p_bonf'] = np.minimum(SP.p * len(feats), 1.0)
SP.to_csv(DATA / 'espectro_contraste_stn.csv', index=False, float_format='%.5g')
print(SP.to_string(index=False, float_format=lambda v: '%.4g' % v))
