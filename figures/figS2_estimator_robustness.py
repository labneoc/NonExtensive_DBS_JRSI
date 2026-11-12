"""Figure S2 - per-record bootstrap uncertainty and artifact sensitivity.

Inputs : bootstrap_blocks_q_B1000.csv, robustez_artefato.csv
Output : figures_out/FigS2_estimator_robustness.png
Run    : python figures/figS2_estimator_robustness.py

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

# Load data
B_raw = pd.read_csv(P['bootstrap_blocks_q_B1000.csv'])
rob = pd.read_csv(P['robustez_artefato.csv'])

# Merge bootstrap with robustness data
B = B_raw.merge(rob[['file', 'artifact_pct', 'n_samples']], on='file')

# Prepare sorted data for panel A
Bs = B.dropna(subset=['q_lo', 'q_hi']).sort_values(['q_point', 'q_lo']).reset_index(drop=True)

# Compute correlations for panel B
rb_ = rob.dropna(subset=['artifact_pct'])
rb2 = rob.dropna(subset=['artifact_pct', 'q_stat_resid'])
ra = st.spearmanr(rb_.artifact_pct, rb_.q_stat_raw)
rbv = st.spearmanr(rb2.artifact_pct, rb2.q_stat_resid)

NAV = '#1F4E79'
AMB = '#D99A3E'
GRY = '#9AA3AC'


fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axs[0]
ax.fill_between(np.arange(len(Bs)), Bs.q_lo, Bs.q_hi, color=NAV, alpha=.25, lw=0, label='95% moving-block bootstrap interval (1000 resamples)')
ax.plot(np.arange(len(Bs)), Bs.q_point, color=NAV, lw=1.6, label='point estimate')
ax.axhline(5/3, color=AMB, lw=2.0, ls='--'); ax.annotate('$q=5/3$', xy=(4, 5/3+0.03), fontsize=10, color=AMB)
ax.axhline(7/5, color=AMB, lw=1.0, ls=':'); ax.axhline(1.0, color=GRY, lw=1.0)
ax.set_ylim(0.95, 2.85); ax.set_xlabel('Records, ordered by $q$ ($n=%d$)' % len(Bs)); ax.set_ylabel('$q$')
ax.legend(fontsize=9.5, frameon=False, loc='upper left')

plet(ax,'(a)',-0.12)

ax = axs[1]
ax.plot(rb_.artifact_pct, rb_.q_stat_raw, 'o', ms=5, color=NAV, alpha=.7, label=r'$q$, whole record')
ax.plot(rb2.artifact_pct, rb2.q_stat_resid, '^', ms=5, color=AMB, alpha=.75, label=r'$q$ residual (50 ms)')
ax.annotate(r'$\rho=%.3f$' % ra.statistic, xy=(0.62, 0.90), xycoords='axes fraction', fontsize=11.5, color=NAV)
ax.annotate(r'$\rho=%.3f$' % rbv.statistic, xy=(0.62, 0.82), xycoords='axes fraction', fontsize=11.5, color=AMB)
ax.set_xlabel('Samples flagged as artifact (%)'); ax.set_ylabel('$q$')
ax.legend(fontsize=9.5, frameon=False, loc='upper left')

plet(ax,'(b)',-0.12)

fig.tight_layout()
fig.savefig(FIGOUT / 'FigS2_estimator_robustness.png',dpi=300, bbox_inches='tight')