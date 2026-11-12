"""23 artifact robustness merge

Merges q, T, artifact fraction and band values into the table behind Fig. 4(a), Fig. 5(b) and Figs. S2-S3.

Reads  : bcs_T_por_registro_184.csv, metadados_preproc.csv, q_por_banda.csv, qresid_janela_proporcional_T.csv, tripleto_qstat.csv
Writes : robustez_artefato.csv
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

import pandas as pd
import numpy as np
from scipy import stats

QS = pd.read_csv(DATA / 'tripleto_qstat.csv')
M = pd.read_csv(DATA / 'metadados_preproc.csv')
T = pd.read_csv(DATA / 'bcs_T_por_registro_184.csv')
WR = pd.read_csv(DATA / 'qresid_janela_proporcional_T.csv')

BQ = pd.read_csv(DATA / 'q_por_banda.csv')

A = (QS[['file', 'patient', 'cls', 'q_stat_raw', 'q_stat_resid', 'beta_raw', 'r2_raw', 'n_samples']]
     .merge(M[['file', 'artifact_pct', 'noise_sigma']], on='file', how='left')
     .merge(T[['file', 'T_bcs_ms', 'flag_bcs']], on='file', how='left')
     .merge(BQ[['file', 'q_1-100 Hz', 'q_500-2000 Hz']], on='file', how='left'))

A.to_csv(DATA / 'robustez_artefato.csv', index=False, float_format='%.6g')
