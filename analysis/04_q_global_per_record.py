"""04 q global per record

Global q, beta and R^2 per record with the grid estimator, plus the same on a time-shuffled surrogate.

Reads  : qresid_janela_proporcional_T.csv + RAW_ZIP
Writes : tripleto_qstat.csv
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

import zipfile, io, os, time
import numpy as np
import pandas as pd

# We reproduce its logic here as a q-Gaussian fit to the histogram.
from scipy import stats, optimize

# Load WR
WR = pd.read_csv(DATA / 'qresid_janela_proporcional_T.csv')

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

rgq = np.random.default_rng(909); rows = []; t0 = time.time()
for nm in npys:
    pat, cls, fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    f1 = qstat_fit_q(x); f2 = qstat_fit_q(rgq.permutation(x))
    rows.append(dict(file=nm, patient=pat, cls=cls, n_samples=int(x.size),
                     q_stat_raw=f1['q'], r2_raw=f1['r2'], beta_raw=f1['beta'],
                     q_stat_shuf=f2['q'], r2_shuf=f2['r2']))
    del x
QS = pd.DataFrame(rows).merge(WR[['file', 'T_bcs_ms', 'q_kmed', 'q_kmed_shuf', 'win_kmed_ms']].rename(
        columns={'q_kmed': 'q_stat_resid', 'q_kmed_shuf': 'q_stat_resid_shuf'}), on='file', how='left')
QS['edge_raw'] = QS.q_stat_raw >= 2.59
QS.to_csv(DATA / 'tripleto_qstat.csv', index=False, float_format='%.5g')
print('%.0f s | n=%d | edge %d | r2 min %.3f' % (time.time()-t0, len(QS), QS.edge_raw.sum(), QS.r2_raw.min()))
print(QS[['q_stat_raw', 'q_stat_shuf', 'q_stat_resid', 'q_stat_resid_shuf']].median().round(4).to_dict())
print('raw IQR %.3f-%.3f | resid IQR %.3f-%.3f' % (QS.q_stat_raw.quantile(.25), QS.q_stat_raw.quantile(.75),
      QS.q_stat_resid.quantile(.25), QS.q_stat_resid.quantile(.75)))
