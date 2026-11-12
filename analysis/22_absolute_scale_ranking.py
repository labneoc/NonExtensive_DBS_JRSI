"""22 absolute scale ranking

Reconstructs the lost channel scale from the preprocessing metadata and ranks the absolute-scale descriptors.

Reads  : metadados_preproc.csv + RAW_ZIP
Writes : ranking_escala_absoluta.csv
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

import zipfile, io, time
import numpy as np, pandas as pd
from scipy import stats, signal as sg

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

M = pd.read_csv(DATA / 'metadados_preproc.csv')

rows = []; t0 = time.time()
for nm in npys:
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    mad = np.median(np.abs(x - np.median(x))) / 0.6745
    env = np.sqrt((x[:(x.size//24)*24].reshape(-1, 24)**2).mean(1))
    rows.append(dict(file=nm, sz_mad=mad, sz_envmode=np.percentile(env, 50), sd_z=x.std()))
    del x
SZ = pd.DataFrame(rows)

R = M.merge(SZ, on='file', how='left')
R['scale_mad'] = R.noise_sigma / R.sz_mad
R['scale_env'] = R.noise_sigma / R.sz_envmode

R['abs_p500_3k'] = R.scale_mad**2 * R.rel_f500_3k
R['abs_beta'] = R.scale_mad**2 * R.rel_beta
R['abs_rms'] = R.scale_mad
R['frac_fundo'] = R.sz_mad
R['pico_sobre_fundo'] = 1.0 / R.sz_mad

def auc_pp(df, col, sign):
    out = {}
    for p, g in df.groupby('patient'):
        I = g.loc[g.cls == 'inside', col].dropna().values * sign
        O = g.loc[g.cls == 'outside', col].dropna().values * sign
        if len(I) and len(O):
            out[p] = np.mean([(1. if a > b else .5 if a == b else 0.) for a in I for b in O])
    return pd.Series(out)

def lopo_auc2(df, col, robust=True):
    pm = df.groupby(['patient', 'cls'])[col].mean().unstack().dropna()
    per = {}
    for p in pm.index:
        dif = (pm.drop(index=p)['inside'] - pm.drop(index=p)['outside'])
        s = np.sign(dif.median() if robust else dif.mean()) or 1.0
        a = auc_pp(df[df.patient == p], col, s)
        if len(a):
            per[p] = a.iloc[0]
    return pd.Series(per)

cand = [('noise_sigma', 'amplitude de fundo (metadado)'),
        ('abs_rms', 'escala reconstruída'),
        ('abs_p500_3k', 'potência ABSOLUTA 500–3000 Hz'),
        ('abs_beta', 'potência ABSOLUTA beta'),
        ('frac_fundo', 'fundo / SD total (livre de escala)'),
        ('q_stat_resid', 'q residual'),
        ('T_bcs_ms', 'T (BCS)'),
        ('rel_f500_3k', 'potência relativa 500–3000 Hz'),
        ('q_stat_raw', 'q bruto')]

bad = R.nlargest(1, 'scale_mad').file.iloc[0]
Rc = R[R.file != bad].copy()

rb = np.random.default_rng(11)
out = []; AU = {}
for col, lab in cand:
    a = lopo_auc2(Rc, col)
    AU[col] = a
    bs = [np.mean(rb.choice(a.values, a.size)) for _ in range(4000)]
    out.append(dict(descritor=lab, coluna=col, AUC=a.mean(),
                    lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(a)))

RK = pd.DataFrame(out).sort_values('AUC', ascending=False)
RK.to_csv(DATA / 'ranking_escala_absoluta.csv', index=False, float_format='%.5g')
