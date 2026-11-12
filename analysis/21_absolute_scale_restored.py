"""21 absolute scale restored

Per-record reconstructed amplitude scale (MAD and envelope based).

Reads  : metadados_preproc.csv + RAW_ZIP
Writes : escala_restaurada.csv
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

ZP=RAW_ZIP
zf=zipfile.ZipFile(ZP); npys=sorted(n for n in zf.namelist() if n.endswith('.npy'))
M=pd.read_csv(DATA / 'metadados_preproc.csv')
rows=[]; t0=time.time()
for nm in npys:
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    mad=np.median(np.abs(x-np.median(x)))/0.6745          # nivel de fundo robusto, em unidades z
    env=np.sqrt((x[:(x.size//24)*24].reshape(-1,24)**2).mean(1))
    rows.append(dict(file=nm,sz_mad=mad,sz_envmode=np.percentile(env,50),sd_z=x.std())); del x
SZ=pd.DataFrame(rows)
R=M.merge(SZ,on='file',how='left')
R['scale_mad']=R.noise_sigma/R.sz_mad                      # unidades brutas por unidade z
R['scale_env']=R.noise_sigma/R.sz_envmode
print('%.0f s'%(time.time()-t0))
print('sz_mad (nivel de fundo no traco z): mediana %.4f  IQR %.4f-%.4f  CV=%.3f'%(
    R.sz_mad.median(),R.sz_mad.quantile(.25),R.sz_mad.quantile(.75),R.sz_mad.std()/R.sz_mad.mean()))
print('sd do traco z (deve ser 1): mediana %.4f'%R.sd_z.median())
print('fator de escala recuperado: mediana %.2f  IQR %.2f-%.2f  faixa %.2f-%.2f'%(
    R.scale_mad.median(),R.scale_mad.quantile(.25),R.scale_mad.quantile(.75),R.scale_mad.min(),R.scale_mad.max()))
print('rho(scale_mad, noise_sigma) = %+.3f  -> se ~1, a escala nao acrescenta nada alem do noise_sigma'%
      stats.spearmanr(R.scale_mad,R.noise_sigma)[0])
print('rho(sz_mad, noise_sigma)    = %+.3f'%stats.spearmanr(R.sz_mad,R.noise_sigma)[0])
R.to_csv(DATA / 'escala_restaurada.csv',index=False,float_format='%.6g')
