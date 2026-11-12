"""The q-Gaussian estimator used throughout the paper.

`qstat_fit_q` is the grid search on the linearized q-logarithmic plane described in
Sec. II of the manuscript: histogram of the sample trimmed by `clip` percent per tail,
`nbins` bins, grid of q from `qmin` to `qmax` in steps of `step`, keeping the q that
maximizes the coefficient of determination of ln_q(rho/rho_max) against x^2.

Known limitations, stated in the paper: no confidence interval, tail weight that depends
on the binning, and the possibility of pinning at a grid edge (five records do, four at
1.01 and one at 2.59; they are retained in every median and paired test).

Verified on 2026-10-29 against the stored per-record table: for records
01/inside/step_+0000_hemi_L, 13/outside, 26/inside, 38/outside and 46/outside this
implementation reproduces q, beta and R^2 of q_global_per_record exactly.

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import numpy as np
from scipy import stats


def qstat_lnq(y, q):
    """q-logaritmo: (y^(1-q)-1)/(1-q); q=1 devolve ln(y)."""
    import numpy as np
    y = np.asarray(y, float)
    if abs(q - 1.0) < 1e-12:
        return np.log(y)
    return (y ** (1.0 - q) - 1.0) / (1.0 - q)


def qstat_expq(z, q):
    """q-exponencial: [1+(1-q)z]^(1/(1-q)); q=1 devolve exp(z)."""
    import numpy as np
    z = np.asarray(z, float)
    if abs(q - 1.0) < 1e-12:
        return np.exp(z)
    base = 1.0 + (1.0 - q) * z
    return np.where(base > 0, np.maximum(base, 0) ** (1.0 / (1.0 - q)), 0.0)


def qstat_fit_q(sample, qmin=1.01, qmax=2.60, step=0.01, nbins=120, clip=0.5):
    """Estimador do artigo: busca em grade maximizando R^2 de ln_q(rho/rho0) vs x^2.

    Devolve {'q','beta','r2'}. Limitacao conhecida: sem intervalo de confianca,
    peso de cauda dependente do binning, e pode travar na borda da grade.
    """
    import numpy as np
    from scipy import stats
    s = np.asarray(sample, float)
    lo, hi = np.percentile(s, [clip, 100.0 - clip])
    h, e = np.histogram(s[(s > lo) & (s < hi)], bins=nbins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    m = h > 0
    xx = c[m] ** 2
    ratio = h[m] / h[m].max()
    best = {"q": float("nan"), "beta": float("nan"), "r2": -1.0}
    for q in np.arange(qmin, qmax, step):
        yy = qstat_lnq(ratio, q)
        sl, ic, rr = stats.linregress(xx, yy)[:3]
        if rr ** 2 > best["r2"]:
            best = {"q": float(q), "beta": float(-sl), "r2": float(rr ** 2)}
    return best
