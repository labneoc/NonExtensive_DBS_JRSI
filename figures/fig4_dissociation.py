"""Figure 4 - the border of the nucleus through the three quantities: global q, time scale T, residual q.

Inputs : robustez_artefato.csv, bcs_T_por_registro_184.csv, qresid_janela_proporcional_T.csv
Output : figures_out/Fig4_dissociation.png
Run    : python figures/fig4_dissociation.py

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

Rb = pd.read_csv(P['robustez_artefato.csv'])
T2 = pd.read_csv(P['bcs_T_por_registro_184.csv'])
W = pd.read_csv(P['qresid_janela_proporcional_T.csv'])


def paired(df, col, log=False):
    """One value per patient per side (median of its two records); paired Wilcoxon and Cohen's d_z."""
    d_ = df.copy()
    d_['v'] = np.log10(d_[col]) if log else d_[col]
    p = d_.pivot_table(index='patient', columns='cls', values='v', aggfunc='median').dropna()
    d = p['inside'] - p['outside']
    s = dict(md=d.mean(), p=st.wilcoxon(p['inside'], p['outside']).pvalue, dz=d.mean() / d.std(ddof=1))
    if log:
        p = 10 ** p
    s.update(mi=p['inside'].median(), mo=p['outside'].median())
    return p, s


def pfmt(p_):
    a_, e_ = ('%.1e' % p_).split('e')
    return r'%s\times10^{%d}' % (a_, int(e_))


def panel(ax, p, ylabel, txt, title, letter, log=False):
    for i in p.index:
        ax.plot([0, 1], [p.loc[i, 'inside'], p.loc[i, 'outside']], '-', color=GRY, lw=.8, alpha=.65)
    ax.plot(np.zeros(len(p)), p['inside'], 'o', color=NAV, ms=5.5, alpha=.85)
    ax.plot(np.ones(len(p)), p['outside'], 'o', color=RED, ms=5.5, alpha=.85)
    ax.plot([-.14, .14], [p['inside'].median()] * 2, color=NAV, lw=3.2)
    ax.plot([.86, 1.14], [p['outside'].median()] * 2, color=RED, lw=3.2)
    if log:
        ax.set_yscale('log')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['inside STN', 'outside STN'])
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(ylabel)
    ax.text(.5, .985, txt, transform=ax.transAxes, ha='center', va='top', fontsize=9.5, color='#2E3B48')
    ax.set_title(title, loc='left', fontsize=10.5, color=DGR, pad=14)
    plet(ax, letter, -0.17)


pg, sg = paired(Rb, 'q_stat_raw')
pT, sT = paired(T2[T2.flag_bcs == 'ok'], 'T_bcs_ms', log=True)
pq, sq = paired(W, 'q_kmed')

fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.9))
panel(axs[0], pg, 'Global index $q$ (whole record)',
      '$\\Delta=%+.3f$,  $p=%.3f$,  $d_z=%.2f$' % (sg['md'], sg['p'], sg['dz']), 'Size: no difference', '(a)')
axs[0].set_ylim(0.95, 2.3)
panel(axs[1], pT, 'Time scale $T$ (ms)',
      '%.2f vs %.2f ms (geom. ratio %.2f)\n$p=%s$,  $d_z=%.2f$' % (sT['mi'], sT['mo'], 10 ** sT['md'], pfmt(sT['p']), sT['dz']),
      'Clock: faster inside', '(b)', log=True)
axs[1].set_yticks([2, 3, 4, 6, 10, 20, 40])
axs[1].set_yticklabels(['2', '3', '4', '6', '10', '20', '40'])
axs[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
axs[1].set_ylim(1.2, 60)
panel(axs[2], pq, 'Residual index $q_{\\mathrm{res}}$ (own clock, $k=12.4$)',
      '%.3f vs %.3f ($\\Delta=%+.3f$)\n$p=%s$,  $d_z=%.2f$' % (sq['mi'], sq['mo'], sq['md'], pfmt(sq['p']), sq['dz']),
      'Residual: higher inside', '(c)')
axs[2].set_ylim(0.98, 1.5)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig4_dissociation.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig4: global', sg, '| T', sT, '| q_res', sq)
