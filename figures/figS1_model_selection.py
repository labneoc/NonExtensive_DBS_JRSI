"""Figure S1 - marginal model selection by AIC.

Inputs : selecao_modelo_marginal.csv
Output : figures_out/FigS1_model_selection.png
Run    : python figures/figS1_model_selection.py

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

sm = pd.read_csv(P['selecao_modelo_marginal.csv'])

NAV = '#1F4E79'
RED = '#B3352B'
GRY = '#9AA3AC'
AMB = '#D99A3E'

mods = ['gauss', 't', 'gn', 'stable', 'mix2']
nm = {
    'gauss': 'Gaussian',
    't': '$q$-Gaussian (Student-$t$)',
    'gn': 'generalized Gaussian',
    'stable': r'$\alpha$-stable',
    'mix2': 'two-component\nGaussian mixture'
}

S = sm.copy()
A = S[['AIC_%s' % m for m in mods]].values
best = A.min(axis=1)
exc = pd.DataFrame((A - best[:, None]) / S.n_sub.values[:, None], columns=mods)
em = exc.median().sort_values()
win = S.vencedor.value_counts()

print(em.to_string())
print(win.to_string())

fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axs[0]
yb = np.arange(len(em))[::-1]
ax.barh(yb, em.values,
        color=[NAV if k == 'mix2' else (AMB if k == 't' else GRY) for k in em.index],
        height=.6)
for yi, v in zip(yb, em.values):
    ax.annotate('%.4f' % v, (v + 0.003, yi), va='center', fontsize=9.5, color='#2E3B48')
ax.set_yticks(yb)
ax.set_yticklabels([nm[k] for k in em.index], fontsize=10)
ax.set_xlim(0, 0.172)
ax.set_xlabel(r'Excess AIC per sample over the best model (median)')

plet(ax,'(a)',-0.30)

order = [m for m in ['mix2', 't', 'stable', 'gauss', 'gn'] if m in win.index]
vals = [int(win.get(m, 0)) for m in order]

ax = axs[1]
lbl2 = {
    'mix2': 'two-component\nmixture',
    't': '$q$-Gaussian\n(Student-$t$)',
    'stable': r'$\alpha$-stable',
    'gauss': 'Gaussian',
    'gn': 'generalized\nGaussian'
}
ax.bar(np.arange(len(order)), vals,
       color=[NAV if m == 'mix2' else (AMB if m == 't' else GRY) for m in order],
       width=.6)
for i, v in enumerate(vals):
    ax.annotate(str(v), (i, v + 3), ha='center', fontsize=10.5, color='#2E3B48')
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([lbl2[m] for m in order], fontsize=9.5)
ax.set_ylabel('Records won (of 184)')
ax.set_ylim(0, 180)

plet(ax,'(b)',-0.30)

fig.tight_layout()
fig.savefig(FIGOUT / 'FigS1_model_selection.png',dpi=300, bbox_inches='tight')
print('ok')