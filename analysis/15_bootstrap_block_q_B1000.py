"""15 bootstrap block q B1000

Moving-block bootstrap of the global q, 500 ms blocks, 1000 resamples (the version reported in the paper).

Reads  : (raw traces only) + RAW_ZIP
Writes : bootstrap_blocks_q_B1000.csv
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

from qstat import qstat_lnq

import numpy as np
import pandas as pd
import zipfile
import io
import time
import sys
from multiprocessing import Pool

Z = RAW_ZIP

QG = np.arange(1.01, 2.60, 0.01)

def fitq(s, nbins=120, clip=0.5):
    lo, hi = np.percentile(s, [clip, 100 - clip])
    h, e = np.histogram(s[(s > lo) & (s < hi)], bins=nbins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    m = h > 0
    xx = c[m] ** 2
    ratio = h[m] / h[m].max()
    best = (np.nan, -1.0)
    for q in QG:
        yy = qstat_lnq(ratio, q)
        r = np.corrcoef(xx, yy)[0, 1] ** 2
        if r > best[1]:
            best = (float(q), r)
    return best[0]

def work(nm, B=1000, blk=12000):
    zf = zipfile.ZipFile(Z)
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))), dtype=np.float64)
    N = x.size
    nb = N // blk
    rng = np.random.default_rng(abs(hash(nm)) % (2 ** 32))
    if nb < 2:
        return dict(file=nm, n_blocks=nb)
    qs = np.empty(B)
    for b in range(B):
        st = rng.integers(0, N - blk + 1, nb)
        y = np.concatenate([x[s:s + blk] for s in st])
        qs[b] = fitq(y)
    return dict(file=nm, n_blocks=nb, B=B, q_point=fitq(x),
                q_lo=np.percentile(qs, 2.5), q_hi=np.percentile(qs, 97.5),
                q_sd=qs.std(ddof=1))

if __name__ == '__main__':
    zf = zipfile.ZipFile(Z)
    npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))
    t0 = time.time()
    with Pool(6) as p:
        rows = p.map(work, npys, chunksize=4)
    B_df = pd.DataFrame(rows)
    B_df.to_csv(DATA / 'bootstrap_blocks_q_B1000.csv', index=False, float_format='%.6g')
    print('%.0f s' % (time.time() - t0))
