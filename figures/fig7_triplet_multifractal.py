"""Figure 7 - the q-triplet against its two one-parameter predictions, and multifractal width.

Inputs : tripleto_q_184.csv, tripleto_mfdfa.csv, tripleto_bateria_resumo.csv, tripleto_qrel.csv
Output : figures_out/Fig7_triplet_multifractal.png
Run    : python figures/fig7_triplet_multifractal.py

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

TR = pd.read_csv(P['tripleto_q_184.csv'])
MF = pd.read_csv(P['tripleto_mfdfa.csv'])
NU = pd.read_csv(P['tripleto_bateria_resumo.csv'], index_col=0)
QR = pd.read_csv(P['tripleto_qrel.csv'])


def med_ci(v, B=5000, seed=3):
    v = np.asarray(pd.Series(v).dropna())
    bs = [np.median(np.random.default_rng(seed + i).choice(v, len(v), True)) for i in range(B)]
    return np.median(v), *np.percentile(bs, [2.5, 97.5])


qst = float(TR.q_stat_raw.median())
pred_rel = 4 * qst - 3          # mean relations, Eq. (3) of the paper
pred_dual = 1 / (2 - qst)       # duality relations q -> 2 - 1/q
qrw = med_ci(QR.q_rel_wide)
qsen = med_ci(TR.q_sen20_real)
prot = [QR[c].median() for c in ['q_rel', 'q_rel_wide', 'q_rel5', 'q_rel_ar1']]
span_rel = (min(prot), max(prot))
span_sen = (0.455, 0.982)       # span of the median over sixteen analysis choices

items = [('records', MF.width_real.median(), NAV),
         (r'$\chi^2$ superstatistics', float(NU.loc['superstat', 'width']), RED),
         (r'AR(1), $\phi=0.99$', float(NU.loc['ar1', 'width']), GRY),
         ('phase-randomized\nsurrogate', MF.width_ft.median(), GRY),
         ('Student-$t$, same $q$', float(NU.loc['student', 'width']), GRY),
         (r'$1/f$ noise ($H=0.9$)', float(NU.loc['pink', 'width']), GRY),
         ('time-shuffled surrogate', MF.width_shuf.median(), GRY),
         ('Gaussian i.i.d.', float(NU.loc['gaussian', 'width']), GRY)]
items = sorted(items, key=lambda t: t[1])

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax = axs[0]
for yi, sp in [(1, span_rel), (0, span_sen)]:
    ax.hlines(yi, sp[0], sp[1], color=NAV, lw=10, alpha=.16)
ax.plot(pred_rel, 1, 'D', ms=9, color=AMB)
ax.plot(pred_dual, 1, 'v', ms=10, color=AMB, mfc='white', mew=2)
ax.hlines(1, qrw[1], qrw[2], color=NAV, lw=3.0, alpha=.6)
ax.plot(qrw[0], 1, 'o', ms=9, color=NAV)
ax.hlines(0, qsen[1], qsen[2], color=NAV, lw=3.0, alpha=.6)
ax.plot(qsen[0], 0, 'o', ms=9, color=NAV)
ax.plot([], [], 'D', ms=9, color=AMB, label=r'mean relations, from $q_{\mathrm{stat}}=1.480$')
ax.plot([], [], 'v', ms=10, color=AMB, mfc='white', mew=2, label=r'duality relations, from $q_{\mathrm{stat}}=1.480$')
ax.plot([], [], 'o', ms=9, color=NAV, label='measured (median, 95% CI)')
ax.plot([], [], '-', lw=10, alpha=.16, color=NAV, label='span over analysis protocols')
ax.axvline(1.0, color=GRY, lw=1.2)
ax.text(1.04, 1.45, 'Boltzmann-Gibbs', fontsize=9.5, color=DGR)
ax.set_yticks([1, 0])
ax.set_yticklabels([r'$q_{\mathrm{rel}}$', r'$q_{\mathrm{sen}}$'], fontsize=14)
ax.set_xlim(-0.1, 3.25)
ax.set_ylim(-0.55, 1.80)
ax.set_xlabel('Index value')
ax.legend(frameon=False, loc='lower right', fontsize=9)
plet(ax, '(a)', -0.10)

ax = axs[1]
yb = np.arange(len(items))
ax.barh(yb, [t[1] for t in items], color=[t[2] for t in items], height=.62)
for yi, t_ in zip(yb, items):
    ax.text(t_[1] + 0.012, yi, '%.3f' % t_[1], va='center', fontsize=9.5, color='#2E3B48')
ax.set_yticks(yb)
ax.set_yticklabels([t_[0] for t_ in items])
ax.set_xlim(0, 0.70)
ax.set_xlabel(r'Multifractal spectrum width $\alpha_{\max}-\alpha_{\min}$')
plet(ax, '(b)', -0.36)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig7_triplet_multifractal.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig7')
