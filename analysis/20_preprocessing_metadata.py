"""20 preprocessing metadata

Per-record preprocessing metadata (interpolated fraction, noise level, trim limits) read from the selected_files.csv files inside the deposit.

Reads  : headtohead_features.csv + RAW_ZIP
Writes : metadados_preproc.csv
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

import zipfile, io
import numpy as np
import pandas as pd
from scipy import stats

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
names = zf.namelist()

csvs = [n for n in names if n.endswith('selected_files.csv')]

rows = []
for c in csvs:
    df = pd.read_csv(io.BytesIO(zf.read(c)))
    rows.append(df)
SEL = pd.concat(rows, ignore_index=True)
SEL['file'] = SEL.output_npy.str.replace(r'.*/dbs_mendeley/', '', regex=True)

H = pd.read_csv(DATA / 'headtohead_features.csv')
M = H.merge(SEL[['file', 'artifact_pct', 'noise_sigma', 'trim_start', 'trim_end']], on='file', how='left')

M.to_csv(DATA / 'metadados_preproc.csv', index=False, float_format='%.6g')
