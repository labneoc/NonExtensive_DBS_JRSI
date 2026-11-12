"""Figure 5 - residual q on each record's own clock, and why a fixed window reads T.

Inputs : robustez_artefato.csv, qresid_janela_proporcional_T.csv, bcs_T_por_registro_184.csv
Output : figures_out/Fig5_residual_own_clock.png
Run    : python figures/fig5_residual_own_clock.py

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

D_rob = pd.read_csv(P['robustez_artefato.csv'])
W = pd.read_csv(P['qresid_janela_proporcional_T.csv'])
T2 = pd.read_csv(P['bcs_T_por_registro_184.csv'])


def paired(col):
    """Paired inside-outside difference, one value per patient per side (median of its two records)."""
    p_ = W.pivot_table(index='patient', columns='cls', values=col, aggfunc='median').dropna()
    dd = p_['inside'] - p_['outside']
    bs = np.array([np.mean(np.random.default_rng(2000 + i).choice(dd.values, len(dd), replace=True))
                   for i in range(5000)])
    return (len(dd), dd.mean(), *np.percentile(bs, [2.5, 97.5]),
            st.wilcoxon(p_['inside'], p_['outside']).pvalue, dd.mean() / dd.std(ddof=1))


sets = [('fixed 50 ms', 'q_fix50', 'q_fix50_shuf', None),
        (r'$k=2$', 'q_k2', 'q_k2_shuf', 'win_k2_ms'),
        (r'$k=5$', 'q_k5', 'q_k5_shuf', 'win_k5_ms'),
        (r'$k=12.4$', 'q_kmed', 'q_kmed_shuf', 'win_kmed_ms'),
        (r'$k=25$', 'q_k25', 'q_k25_shuf', 'win_k25_ms')]

RR = []
for lab, cr, cs, cw in sets:
    n, m_, l, h, p_, dz = paired(cr)
    n2, m2, l2, h2, p2, dz2 = paired(cs)
    RR.append(dict(lab=lab, win=W[cw].median() if cw else 50.0, n=n, d=m_, lo=l, hi=h, p=p_, dz=dz,
                   d_s=m2, lo_s=l2, hi_s=h2, p_s=p2))
RR = pd.DataFrame(RR)

Mc = D_rob.merge(T2[['file', 'T_bcs_ms', 'flag_bcs']].rename(columns={'flag_bcs': 'fl', 'T_bcs_ms': 'Tb'}), on='file')
Mc = Mc[Mc.fl == 'ok'].copy()
Mc['T_bcs_ms'] = Mc.Tb
r1 = st.spearmanr(np.log10(Mc.T_bcs_ms), Mc.q_stat_raw)
r2 = st.spearmanr(np.log10(Mc.T_bcs_ms), Mc.q_stat_resid)


def pfmt(p_):
    s = '%.1e' % p_
    a_, e_ = s.split('e')
    return r'%s\times10^{%d}' % (a_, int(e_))


fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax = axs[0]
yv = np.arange(len(RR))[::-1]
ax.hlines(yv, RR.lo, RR.hi, color=NAV, lw=3.2, alpha=.5)
ax.plot(RR.d, yv, 'o', ms=8, color=NAV, label='records')
ax.hlines(yv - 0.26, RR.lo_s, RR.hi_s, color=GRY, lw=3.2, alpha=.5)
ax.plot(RR.d_s, yv - 0.26, 's', ms=6, color=GRY, label='time-shuffled surrogate')
ax.axvline(0, color='k', lw=1.0)
ax.set_yticks(yv)
ax.set_yticklabels(['%s\n(%.0f ms)' % (l, w) for l, w in zip(RR.lab, RR.win)])
for yi, p_, dz in zip(yv, RR.p, RR.dz):
    ax.text(0.16, yi, '$p=%s$,  $d_z=%.2f$' % (pfmt(p_), dz), fontsize=9.5, va='center', color='#2E3B48')
ax.set_xlim(-0.075, 0.265)
ax.set_ylim(-0.75, len(RR) - 0.4)
ax.set_xlabel(r'$\Delta q_{\mathrm{res}}$ (inside $-$ outside STN, 46 patients)')
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
plet(ax, '(a)', -0.22)

ax = axs[1]
ax.plot(Mc.T_bcs_ms, Mc.q_stat_raw, 'o', ms=5.5, color=NAV, alpha=.75,
        label=r'global index ($\rho=%.3f$)' % r1.statistic)
ax.plot(Mc.T_bcs_ms, Mc.q_stat_resid, '^', ms=5.5, color=AMB, alpha=.8,
        label=r'residual index, fixed 50 ms window ($\rho=%.3f$)' % r2.statistic)
ax.set_ylim(0.95, 2.68)
ax.set_xscale('log')
ax.set_xlabel('Time scale $T$ (ms)')
ax.set_ylabel('$q$')
ax.legend(frameon=False, loc='upper right')
plet(ax, '(b)', -0.12)

fig.tight_layout()
fig.savefig(FIGOUT / 'Fig5_residual_own_clock.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig5')
