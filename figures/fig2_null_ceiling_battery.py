"""Figure 2 - null ceiling of the estimator and the battery of generative processes.

Inputs : teto_nulo_q_vs_N.csv, bateria_processos.csv
Output : figures_out/Fig2_null_ceiling_battery.png
Run    : python figures/fig2_null_ceiling_battery.py

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

b = pd.read_csv(P['bateria_processos.csv'])
tt = pd.read_csv(P['teto_nulo_q_vs_N.csv'])

ren = {
    'AR(1), phi=0.9': r'AR(1), $\phi=0.9$',
    'AR(1), phi=0.99': r'AR(1), $\phi=0.99$',
    'Student-t, nu=2.46': r'Student-$t$, $\nu=2.46$',
    'Student-t, nu=10': r'Student-$t$, $\nu=10$',
    'superstat. n=2.5, T=50': r'superstatistics, $T=50$',
    'superstat. n=2.5, T=10^4': r'superstatistics, $T=10^4$',
    '1/f noise (H=0.9)': r'$1/f$ noise ($H=0.9$)',
    'Gauss + 1% outliers (x5)': r'Gaussian + 1% outliers ($\times5$)',
    'Gauss + 0.1% outliers (x10)': r'Gaussian + 0.1% outliers ($\times10$)',
}
bb = b.sort_values('q_med').reset_index(drop=True)
bb['lab'] = bb.proc.map(lambda s: ren.get(s, s))

qg = 1.480        # median grid-estimator q over the 184 records
Nd = 240000       # typical record length
xx = np.logspace(2.9, 5.7, 200)

fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))

ax = axs[0]
ax.plot(tt.N, tt.q95_grade_ext, 'o-', color=NAV, lw=2, ms=6, label='grid estimator, 95th percentile')
ax.plot(tt.N, tt.q50_grade_ext, 'o--', color=NAV, lw=1.2, ms=4, alpha=.55, label='grid estimator, median')
ax.plot(tt.N, tt.q95_ML, 's-', color=RED, lw=2, ms=6, label='maximum likelihood, 95th percentile')
ax.plot(xx, 1 + 2.558 / np.sqrt(xx), ':', color='k', lw=1.6, label=r'$1+2.558/\sqrt{N}$ (simulated)')
ax.plot(xx, 1 + 2.09 / np.sqrt(xx), '-.', color=DGR, lw=1.1, label=r'$1+2.09/\sqrt{N}$ (asymptotic)')
ax.axhline(1.0, color=GRY, lw=1.0)
ax.axhline(qg, color=AMB, lw=2.2)
ax.text(1.1e3, qg + 0.012, 'median of the records, $q=1.480$', color=AMB, fontsize=10.5)
ax.axvline(Nd, color=GRY, lw=1.0, ls='--')
ax.text(Nd * 0.92, 0.845, 'length of the records\n($N=240\,000$)', ha='right', va='bottom', fontsize=9.5, color=DGR)
ax.set_xscale('log')
ax.set_xlabel('Sample size $N$')
ax.set_ylabel('Fitted $q$ on Gaussian i.i.d. data')
ax.set_ylim(0.80, 1.56)
ax.legend(loc='center right', frameon=False)
plet(ax, '(a)')

ax = axs[1]


def col(r):
    if r.q_med < 0.9:
        return DGR
    return RED if r.q_med >= 1.10 else NAV


cols = [col(r) for _, r in bb.iterrows()]
y = np.arange(len(bb))
ax.hlines(y, bb.q_lo, bb.q_hi, color=cols, lw=3.0, alpha=.45)
for yi, xi, ci, mk in zip(y, bb.q_med, cols, ['D' if c == DGR else 'o' for c in cols]):
    ax.plot(xi, yi, mk, ms=7, color=ci)
ax.set_yticks(y)
ax.set_yticklabels(bb.lab)
ax.axvline(1.0, color=GRY, lw=1.2)
ax.axvline(qg, color=AMB, lw=2.2)
ax.text(qg - 0.02, 1.1, 'median of\nthe records', color=AMB, fontsize=10, ha='right', va='center')
il = int(np.where(np.array(cols) == DGR)[0][0])
ax.text(bb.q_med[il] + 0.03, il, 'below unity: bounded,\nU-shaped density', fontsize=9, color=DGR, va='center')
ax.set_xlabel(r'Fitted $q$ of the generative process ($N=10^5$)')
ax.set_xlim(0.55, 1.68)
ax.set_ylim(-0.7, len(bb) - 0.3)
plet(ax, '(b)', -0.42)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig2_null_ceiling_battery.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig2')
