"""Figure S4 - within-patient head-to-head comparison of descriptors.

Inputs : ranking_escala_absoluta.csv
Output : figures_out/FigS4_head_to_head.png
Run    : python figures/figS4_head_to_head.py

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import P, FIGOUT, NAV, RED, GRY, AMB, DGR, apply_figure_style, plet

apply_figure_style()

RK = pd.read_csv(P['ranking_escala_absoluta.csv'])

NAV = '#1F4E79'
AMB = '#D99A3E'
GRY = '#9AA3AC'

tr = {
    'potência ABSOLUTA 500–3000 Hz': 'absolute power 500–3000 Hz',
    'amplitude de fundo (metadado)': 'background amplitude (metadata)',
    'escala reconstruída': 'reconstructed scale',
    'q residual': r'$q$ residual (50 ms)',
    'T (BCS)': '$T$ (Beck–Cohen–Swinney)',
    'potência relativa 500–3000 Hz': 'relative power 500–3000 Hz',
    'potência ABSOLUTA beta': 'absolute beta power',
    'q bruto': r'$q$, whole record',
    'fundo / SD total (livre de escala)': 'background / total SD (scale-free)'
}

RK['lab'] = RK.descritor.map(tr)
RK = RK.sort_values('AUC').reset_index(drop=True)
cl = [AMB if 'q$' in l or '$T$' in l else (NAV if a >= 0.75 else GRY) for l, a in zip(RK.lab, RK.AUC)]

fig2, ax = plt.subplots(figsize=(8.6, 5.0))
yb = np.arange(len(RK))
ax.hlines(yb, RK.lo, RK.hi, color=cl, lw=3.2, alpha=.45)
for yi, x0, c in zip(yb, RK.AUC, cl):
    ax.plot(x0, yi, 'o', ms=8, color=c)
for yi, x0, c in zip(yb, RK.AUC, cl):
    ax.annotate('%.3f' % x0, (x0 + 0.018, yi), va='center', fontsize=9.5, color='#2E3B48')
ax.axvline(0.5, color=GRY, lw=1.2)
ax.set_yticks(yb)
ax.set_yticklabels(RK.lab, fontsize=10)
ax.set_xlim(0.35, 1.03)
ax.set_xlabel('Within-patient AUC, inside vs outside STN (95% CI)')

fig2.tight_layout()
fig2.savefig(FIGOUT / 'FigS4_head_to_head.png',dpi=300, bbox_inches='tight')
print('ok')