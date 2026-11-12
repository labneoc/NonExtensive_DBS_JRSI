"""Figure 1 - the three properties of a variance fluctuation (size, clock, residual), synthetic illustration.

Inputs : none (synthetic data, fixed seed)
Output : figures_out/Fig1_concept.png
Run    : python figures/fig1_concept.py

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
from scipy.stats import kurtosis, norm

rng = np.random.default_rng(7)
fs, N, blk, ndof = 24000, 240000, 60, 5          # 10 s at 24 kHz; 2.5 ms variance blocks
nb = N // blk
sig = 1 / np.sqrt(rng.chisquare(ndof, nb) / ndof)  # local standard deviation, chi-squared inverse variance
xf = rng.standard_normal((nb, blk)) * sig[:, None]  # fast clock: independent blocks
# slow clock: the SAME blocks, re-ordered so that the variance drifts slowly
g = np.convolve(rng.standard_normal(nb + 400), np.hanning(13), mode='same')[200:200 + nb]
order = np.empty(nb, int)
order[np.argsort(g)] = np.argsort(sig)
xs = xf[order]
xf1, xs1 = xf.ravel(), xs.ravel()
sd = xf1.std()
xf1, xs1 = xf1 / sd, xs1 / sd
assert np.allclose(np.sort(xf1), np.sort(xs1))    # identical samples, hence identical marginal


def kappa(x, deltas_ms):
    out = []
    for d in deltas_ms:
        w = max(4, int(round(d * fs / 1000)))
        n = len(x) // w
        out.append(np.mean(kurtosis(x[:n * w].reshape(n, w), axis=1, fisher=False, bias=True)))
    return np.array(out)


dl = np.logspace(np.log10(0.25), np.log10(500), 50)
kf, ks = kappa(xf1, dl), kappa(xs1, dl)


def cross(k):
    i = np.argmax(k >= 3)
    return np.exp(np.interp(3, k[i - 1:i + 1], np.log(dl[i - 1:i + 1])))


Tf, Ts = cross(kf), cross(ks)
n_ = len(xf1) // blk
Z = xf1[:n_ * blk].reshape(n_, blk)
zf = ((Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)).ravel()

t = np.arange(N) / fs
bins = np.linspace(-8, 8, 121)
c = .5 * (bins[1:] + bins[:-1])
hf = np.histogram(xf1, bins, density=True)[0]
hs = np.histogram(xs1, bins, density=True)[0]
hz = np.histogram(zf, bins, density=True)[0]

fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.8))
ax = axs[0, 0]
m = t < 0.5
ax.plot(t[m], xf1[m] + 9, color=NAV, lw=.35)
ax.plot(t[m], xs1[m] - 9, color=RED, lw=.35)
ax.text(0.005, 18.0, 'fast clock', color=NAV, fontsize=10.5)
ax.text(0.005, 0.0, 'slow clock (the same samples, re-ordered)', color=RED, fontsize=10.5)
ax.set_ylim(-19, 22)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Time (s)')
ax.set_title('Two signals whose variance fluctuates', loc='left', fontsize=10.5, color=DGR)
plet(ax, '(a)', -0.12)

ax = axs[0, 1]
ax.plot(c, norm.pdf(c), color=GRY, lw=1.6, ls='--', label='Gaussian')
ax.plot(c, hs, color=RED, lw=4.5, alpha=.45, label='slow clock')
ax.plot(c, hf, color=NAV, lw=1.6, label='fast clock')
ax.set_yscale('log')
ax.set_ylim(2e-5, 1)
ax.set_xlabel('Amplitude (s.d.)')
ax.set_ylabel('Probability density')
ax.legend(loc='lower center', fontsize=9)
ax.set_title('Size: identical marginal, identical global $q$', loc='left', fontsize=10.5, color=DGR)
plet(ax, '(b)', -0.16)

ax = axs[1, 0]
ax.plot(dl, kf, color=NAV, lw=2.2, label='fast clock')
ax.plot(dl, ks, color=RED, lw=2.2, label='slow clock')
ax.axhline(3, color=AMB, lw=2, ls='--')
ax.text(150, 2.72, r'$\kappa(\Delta)=3$', color=AMB, fontsize=10)
for T_, col in [(Tf, NAV), (Ts, RED)]:
    ax.plot([T_, T_], [1.5, 3], color=col, lw=1.4, ls=':')
    ax.text(T_ * 1.1, 1.62, '$T$', color=col, fontsize=11)
ax.set_xscale('log')
ax.set_ylim(1.5, max(kf.max(), ks.max()) * 1.05)
ax.set_xlabel(r'Window length $\Delta$ (ms)')
ax.set_ylabel(r'Window-averaged kurtosis $\kappa(\Delta)$')
ax.legend(loc='upper left')
ax.set_title('Clock: the window at which the variance stops being constant', loc='left', fontsize=10.5, color=DGR)
plet(ax, '(c)', -0.12)

ax = axs[1, 1]
ax.plot(c, norm.pdf(c), color=GRY, lw=1.6, ls='--', label='Gaussian')
ax.plot(c, hf, color=NAV, lw=1.6, label='whole record')
ax.plot(c[hz > 0], hz[hz > 0], color=AMB, lw=2.4, label='after dividing each window\nby its own s.d.')
ax.set_yscale('log')
ax.set_ylim(2e-5, 1)
ax.set_xlabel('Amplitude (s.d.)')
ax.set_ylabel('Probability density')
ax.legend(loc='lower center', fontsize=9)
ax.set_title('Residual: what is left once the fluctuation is removed', loc='left', fontsize=10.5, color=DGR)
plet(ax, '(d)', -0.16)

fig.tight_layout(h_pad=2.0)
fig.savefig(FIGOUT / 'Fig1_concept.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('wrote Fig1')
