"""Figure 3 - the clock of the variance fluctuation: kappa curves and calibration of T.

Inputs : bcs_kappa_curvas_184.csv, bcs_T_por_registro_184.csv, bcs_nulo_gaussiano.csv, bcs_calibracao_sintetica.csv
Output : figures_out/Fig3_clock.png
Run    : python figures/fig3_clock.py

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

T2 = pd.read_csv(P['bcs_T_por_registro_184.csv'])
K = pd.read_csv(P['bcs_kappa_curvas_184.csv'])
nb = pd.read_csv(P['bcs_nulo_gaussiano.csv'])
cal_raw = pd.read_csv(P['bcs_calibracao_sintetica.csv'])

deltas = np.array([float(c) for c in K.columns[:-1]])
lo, med, hi = np.nanpercentile(K[K.columns[:-1]].values, [25, 50, 75], axis=0)
cal = cal_raw.groupby('T0_ms').agg(T=('T_bcs_ms', 'median')).reset_index()
cal['razao'] = cal.T0_ms / cal['T']
Tmed = float(T2.loc[T2.flag_bcs == 'ok', 'T_bcs_ms'].median())

fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6))
ax = axs[0]
ax.fill_between(deltas, lo, hi, color=NAV, alpha=.20, lw=0)
ax.plot(deltas, med, color=NAV, lw=2.4, label='records (median, IQR)')
ax.axhline(3.0, color=AMB, lw=2.0, ls='--', label=r'criterion $\kappa(\Delta)=3$')
ax.axvline(Tmed, color=RED, lw=1.6, ls=':')
ax.text(Tmed * 1.15, 0.14, '$T=%.2f$ ms' % Tmed, transform=ax.get_xaxis_transform(), color=RED, fontsize=10.5)
ax.set_xscale('log')
ax.set_xlabel(r'Window length $\Delta$ (ms)')
ax.set_ylabel(r'Window-averaged kurtosis $\kappa(\Delta)$')
ax.legend(loc='lower right')
plet(ax, '(a)', -0.14)

ax = axs[1]
ax.plot(cal.T0_ms, cal['T'], 'o-', color=NAV, lw=2, ms=7, label='synthetic superstatistics')
ax.plot([1.2, 400], [1.2, 400], ':', color='k', lw=1.4, label='identity')
for _, r in cal.iterrows():
    ax.annotate(r'$\times$%.1f' % r.razao, (r.T0_ms, r['T']), textcoords='offset points',
                xytext=(5, -14), ha='left', fontsize=9, color=DGR)
ax.axhline(Tmed, color=RED, lw=1.6, ls='--', label='records, median $T=%.2f$ ms' % Tmed)
ax.axhline(nb.loc[nb.flag == 'ok', 'T_ms'].median(), color=GRY, lw=1.6, ls='--',
           label='Gaussian i.i.d. (31 of 60 finite)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.2, 400)
ax.set_ylim(0.6, 500)
ax.set_xlabel('Input time scale $T_0$ (ms)')
ax.set_ylabel('Recovered $T$ (ms)')
ax.legend(loc='upper left', fontsize=9)
plet(ax, '(b)', -0.14)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig3_clock.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig3')
