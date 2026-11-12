"""Figure S3 - the q-beta relation and the median power spectrum.

Inputs : robustez_artefato.csv, psd_mediana_f.npy, psd_mediana_P.npy
Output : figures_out/FigS3_qbeta_and_spectrum.png
Run    : python figures/figS3_qbeta_and_spectrum.py

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

NAV='#1F4E79'; RED='#B3352B'; GRY='#9AA3AC'; AMB='#D99A3E'

R3 = pd.read_csv(P['robustez_artefato.csv'])
R3 = R3.dropna(subset=['beta_raw','q_stat_raw'])

fm = np.load(P['psd_mediana_f.npy'])
Pm = np.load(P['psd_mediana_P.npy'])

s = R3[(R3.q_stat_raw>1.011)&(R3.q_stat_raw<2.599)]
bg = np.linspace(s.beta_raw.min(), s.beta_raw.max(), 200)

mk = (fm>=5)&(fm<=120)
sl = st.linregress(np.log10(fm[mk]), np.log10(Pm[mk]))

fig,axs=plt.subplots(1,2,figsize=(12.6,4.6))
ax=axs[0]
ax.plot(s.beta_raw,s.q_stat_raw,'o',ms=5.5,color=NAV,alpha=.75,label='records ($n=%d$)'%len(s))
ax.plot(bg,5/3-1/(3*bg),'-',color=AMB,lw=2.4,label=r'unit-variance constraint  $q=5/3-1/(3\beta)$')
ax.annotate(r'Spearman $\rho=0.985$'+'\n'+r'constraint alone: $R^2=0.573$',xy=(0.04,0.84),xycoords='axes fraction',
            fontsize=10.5,color='#2E3B48')
ax.set_xlabel(r'$\beta$'); ax.set_ylabel('$q$'); ax.legend(fontsize=9.5,frameon=False,loc='lower right')

ax=axs[1]
ax.loglog(fm[fm>0],Pm[fm>0],color=NAV,lw=1.4)
ax.axvspan(13,30,color=AMB,alpha=.20,lw=0)
ax.plot(fm[mk],10**(sl.intercept+sl.slope*np.log10(fm[mk])),'--',color=RED,lw=2.2)
ax.annotate('beta\nband',xy=(16,2.0e-3),fontsize=9.5,color='#8a6a28',ha='center')
ax.set_ylim(1e-8,6e-3)
ax.annotate('5–120 Hz slope $=%.2f$\n(physiological LFP: $-1$ to $-3$)'%sl.slope,xy=(0.05,0.10),xycoords='axes fraction',
            fontsize=10.5,color=RED)
ax.set_xlim(1,12000); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power spectral density (median over 184)')

fig.tight_layout(); fig.savefig(FIGOUT / 'FigS3_qbeta_and_spectrum.png',dpi=300,bbox_inches='tight')