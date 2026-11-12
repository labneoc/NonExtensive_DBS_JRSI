"""Figure 6 - q resolved by frequency band, and the transfer function of local standardization.

Inputs : q_por_banda_resumo.csv, psd_freq.npy, psd_raw_todos.npy, psd_locz_todos.npy
Output : figures_out/Fig6_q_across_bands.png
Run    : python figures/fig6_q_across_bands.py

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

pf = np.load(P['psd_freq.npy'])
pr = np.load(P['psd_raw_todos.npy'])
pl = np.load(P['psd_locz_todos.npy'])
QB = pd.read_csv(P['q_por_banda_resumo.csv'])

m = pf > 0
ratio = pl[m] / pr[m]
f = pf[m]
i3 = np.argmin(np.abs(ratio[(f > 20) & (f < 2000)] - 0.5))
f3 = f[(f > 20) & (f < 2000)][i3]

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))

ax = axs[0]
yv = np.arange(3)[::-1]
lab = ['1-100 Hz', '100-500 Hz', '500-2000 Hz']
sub = QB[QB.banda.isin(['1-100 Hz', '100-500 Hz', '500-2000 Hz'])].reset_index(drop=True)
ax.hlines(yv, sub.q_surrogate, sub.q_real, color=NAV, lw=2.0, alpha=.35)
ax.plot(sub.q_real, yv, 'o', ms=9, color=NAV, label='records')
ax.plot(sub.q_surrogate, yv, 's', ms=7, color=GRY, label='phase-randomized surrogate')
ax.axvline(1.0, color=GRY, lw=1.0)
ax.axvline(5 / 3, color=AMB, lw=2.0, ls='--')
ax.text(5 / 3 + 0.02, 2.35, '$q=5/3$', fontsize=9.5, color=AMB)
ax.axvline(7 / 5, color=AMB, lw=1.2, ls=':')
ax.text(7 / 5 + 0.015, 2.35, '$q=7/5$', fontsize=9.5, color=AMB)
for yi, (_, r) in zip(yv, sub.iterrows()):
    ax.text(1.02, yi + 0.22, '%.1f%% of power\n%.0f%% beyond 5/3' % (100 * r.pot_rel, r.pct_acima_53),
            fontsize=9, color=DGR)
ax.set_yticks(yv)
ax.set_yticklabels(lab)
ax.set_xlim(0.93, 1.92)
ax.set_ylim(-0.75, 2.6)
ax.set_xlabel('$q$ of the band-limited signal (median over 184 records)')
ax.legend(frameon=False, loc='lower right')
plet(ax, '(a)', -0.20)

ax = axs[1]
mm = f <= 2000
ax.semilogx(f[mm], ratio[mm], color=NAV, lw=2.0)
ax.axhline(1.0, color=GRY, lw=1.0)
ax.axhline(0.5, color=AMB, lw=1.4, ls='--')
ax.axvline(f3, color=RED, lw=1.6, ls=':')
ax.text(f3 * 1.15, 0.42, '$-3$ dB at %.0f Hz' % f3, fontsize=10, color=RED)
ax.axvspan(1, 100, color=AMB, alpha=.12, lw=0)
ax.text(1.6, 0.80, '1-100 Hz\n(highest $q$)', fontsize=9.5, color='#8a6a28')
ax.set_xlim(0.8, 2000)
ax.set_ylim(0, max(1.15, ratio[mm].max() * 1.08))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power after / before local standardization')
plet(ax, '(b)', -0.12)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig6_q_across_bands.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig6')
