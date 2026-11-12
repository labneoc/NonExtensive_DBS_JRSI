# NonExtensive_DBS_JRSI

Analysis code for

> A. P. Martins Neto, H. S. Lima, A. L. S. Tavares, L. S. Mangrich, B. D. Gomes,
> *What the Tsallis index hides: a superstatistical clock and its residual distinguish the
> human subthalamic nucleus from its surroundings* (submitted to the Journal of the Royal
> Society Interface, 2026).

Code author: **Prof. Bruno Duarte Gomes** — Laboratório de Neurofisiologia Eduardo Oswaldo
Cruz, ICB-UFPA / Laboratório de Simulação e Biologia Computacional, CCAD-UFPA.

The repository contains the scripts that turned 184 intraoperative microelectrode
recordings into the tables, the numbers and the eleven figures of the manuscript and its
electronic supplementary material, with two exceptions: one table and one set of lookup arrays were
inherited from an earlier exploratory round whose code is not part of this set. They are
shipped in `data/` and named in Sec. 8 under "Not recovered".

---

## Contents

1. [Why Tsallis statistics for microelectrode recordings](#1-why-tsallis-statistics-for-microelectrode-recordings)
2. [What the analysis found](#2-what-the-analysis-found)
3. [Data](#3-data)
4. [Repository layout](#4-repository-layout)
5. [Quick start](#5-quick-start)
6. [Methods, step by step](#6-methods-step-by-step)
7. [Script reference](#7-script-reference)
8. [Provenance of this code](#8-provenance-of-this-code)
9. [Caveats that travel with the numbers](#9-caveats-that-travel-with-the-numbers)
10. [Citation and contact](#10-citation-and-contact)

---

## 1. Why Tsallis statistics for microelectrode recordings

### 1.1 Additivity is not extensivity

The Boltzmann-Gibbs entropy S_BG = -k Σ p_i ln p_i is *additive*: for two independent
systems, S_BG(A+B) = S_BG(A) + S_BG(B), always. What thermodynamics actually requires is
something else — *extensivity*, that the entropy of a system of N elements grow
proportionally to N. For quasi-independent systems the two coincide, because the number of
accessible states grows exponentially, W ~ mu^N, and S_BG ~ N ln mu.

That coincidence breaks when the elements are strongly correlated. If correlations restrict
the occupancy of phase space so that W grows as a power, W ~ N^rho, then S_BG ~ rho ln N: the
entropy per element vanishes as the system grows, which thermodynamics does not admit. The
nonextensive entropy

    S_q = k (1 - Σ p_i^q) / (q - 1),        S_1 = S_BG

is *non-additive* — S_q(A+B) = S_q(A) + S_q(B) + (1-q)/k · S_q(A) S_q(B) — and for a suitable
q it is again *extensive* in exactly those strongly correlated systems, growing linearly with
N (Tsallis, Gell-Mann & Sato, PNAS **102**, 15377 (2005)). The index q is therefore not a
free-floating fit parameter: it is the exponent that repairs extensivity, and its distance
from 1 measures how far the system is from independence.

Among the entropies in this family, S_q has two properties its competitors lack: it is
concave for every q > 0, and it is Lesche-stable. Rényi and normalized escort entropies are
not (Tsallis, *Introduction to Nonextensive Statistical Mechanics*, Springer 2009, Sec. 3.2).

### 1.2 Why the q-Gaussian, and what q reads in a voltage trace

Maximizing S_q under a variance constraint (in the escort sense) gives the q-Gaussian,

    rho_q(x) = A_q [1 - (1-q) beta x^2]^(1/(1-q)),

which reduces to the Gaussian at q = 1 and develops power-law tails
rho ~ |x|^(-2/(q-1)) for q > 1. Two structural results make it more than a convenient
heavy-tailed shape. First, q-Gaussians are attractors of a q-generalized central limit
theorem for sums of strongly correlated variables, as the Gaussian is for independent ones
(Umarov, Tsallis & Steinberg, Milan J. Math. **76**, 307 (2008)). Second, the q that appears
in the attractor is tied to the q that restores extensivity. A measured q is thus a statement
about the correlation structure of the generating process, if the measurement is sound.

For extracellular recordings this is attractive for practical reasons too. The amplitude
distribution of a microelectrode trace is markedly non-Gaussian; q is a single dimensionless
number, invariant under rescaling of the signal, estimable from a few seconds of data, and
requires no spike sorting, no threshold, and no channel calibration — which matters when the
deposited traces have already been standardized and the physical gain is lost.

### 1.3 The rulers that make a q meaningful

A q is interpretable against *moment boundaries*, not against unity:

| boundary | meaning |
|---|---|
| q = 7/5 | the kurtosis of the q-Gaussian diverges |
| q = 5/3 | the variance diverges; the attractor basin changes from Gaussian to Lévy |
| q = 2 | the mean absolute value diverges |

These are N-independent and physically meaningful, where "q differs from 1" is neither. The
paper reports every index against them.

### 1.4 The four traps, and what this repository does about each

Fitting a q-Gaussian to a neural recording is easy; interpreting the result is not. Four
distinct objections must be answered before a number such as q = 1.5 means anything, and the
code here is organized around them.

**(i) Every estimator has a null distribution.** A finite sample of Gaussian noise does not
return q = 1; it returns a distribution of q whose upper tail depends on N and on the
estimator. A fitted q is interpretable only against the ceiling of its own estimator at its
own sample size. `simulations/s1` computes that ceiling; at the length of these records it is
q95 = 1.0052, far below the measured median.

**(ii) A marginal shape does not identify a mechanism.** Many unrelated processes produce
heavy tails. `simulations/s2` runs thirteen generative processes through the identical
estimator: 6 of the thirteen reach q >= 1.16 without being critical in any sense.
Crucially, a linear filter of Gaussian noise stays Gaussian however long its memory, so long
memory alone cannot inflate q — q reads the marginal, not the correlation. What does inflate
it is a fluctuating variance or a heavy-tailed innovation.

**(iii) Superstatistics is the alternative explanation that must be excluded first.** A
Gaussian whose variance fluctuates slowly produces a q-Gaussian marginal exactly: a chi-squared
mixture gives q = (nu+3)/(nu+1) (Beck & Cohen, Physica A **322**, 267 (2003)). This is not a
rival to the q-Gaussian form, it is a derivation of it — but it threatens any reading of the
*global* q as evidence of critical dynamics. The whole time-domain half of this repository
exists to measure the superstatistical component rather than assume it away: the
Beck-Cohen-Swinney time scale T of the variance fluctuation (`analysis/01`), and the residual
index that survives standardization on each record's own clock (`analysis/03`).

**(iv) Normalization manufactures relations.** The q-Gaussian family is closed under
rescaling: x -> x/s leaves q unchanged and takes beta -> beta s². Fitting q-Gaussians to
standardized data therefore *imposes* a one-parameter relation between q and beta, which is
not evidence of anything. The paper treats beta as a nuisance parameter of the normalization
convention; Fig. S3(a) shows how much of the observed q-beta tie the unit-variance constraint
alone accounts for.

To these the paper adds a fifth test, internal to the formalism: a q-stationary state is
characterized not by one index but by three — the triplet (q_sen, q_rel, q_stat) of
sensitivity to initial conditions, relaxation in phase space, and the stationary
distribution — tied by one-parameter relations. If the triplet does not close, the state is
not the one-parameter q-stationary state the relation describes, whatever the marginal looks
like. `analysis/06`, `analysis/07` and `figures/fig7` carry out that test.

---

## 2. What the analysis found

Every number below is recomputed by the scripts in this repository from the tables in
`data/`; the commands that produce them are named in Sec. 6.

**The magnitude of q is real but uninformative.** The median fitted index over the
184 records is q = 1.48, against a Gaussian-null ceiling of 1.0052 at the same
sample size — the elevation is not an estimator artifact. But 6 of thirteen
noncritical processes reach q >= 1.16 with the same estimator, so the value alone identifies
no mechanism.

**The elevation is superstatistical.** Standardizing each record locally collapses q towards
unity, while the same operation applied to a time-shuffled surrogate of the same record
leaves it unchanged. The Beck-Cohen-Swinney time scale of the variance fluctuation is
T = 4.03 ms in the median and falls ninefold under shuffling.

**The dissociation that is the main result.** Paired within patient across all 46
patients:

| quantity | inside STN | outside STN | contrast | p | d_z |
|---|---|---|---|---|---|
| global index q | 1.53 | 1.47 | +0.031 | 0.435 | +0.131 |
| time scale T (ms) | 3.39 | 6.0 | geometric ratio 0.62 | 1.5e-07 | -0.92 |
| residual index, k = 12.4 | 1.195 | 1.098 | +0.101 | 4.1e-08 | 1.21 |

The globally fitted index carries no anatomical information. The clock of the fluctuation
does, running faster inside the nucleus, and so does the index that remains after
standardization on each record's own clock. In the time-shuffled surrogate the residual
contrast vanishes (+0.025, p = 0.45).

**The contrast is not a linear-spectral effect.** On phase-randomized surrogates, which
preserve each record's power spectrum exactly and destroy everything else, 105 of
184 records never reach the kappa = 3 criterion within 500 ms and the rest cross at a
median of 94.8 ms, against 4.03 ms for the records; the paired contrast disappears
(ratio 1.06, p = 0.81, 24 pairs). The residual index on the same surrogates sits
at the floor of the grid on both sides (1.01, p = 0.65).

**Resolved by frequency, the non-Gaussianity lives where the power does not.**

| band | q of the records | q of the phase-randomized surrogate | share of total power | records beyond 5/3 |
|---|---|---|---|---|
| 1-100 Hz | 1.755 | 1.110 | 4.4 %  | 61 % |
| 100-500 Hz | 1.670 | 1.010 | 12.4 %  | 50 % |
| 500-2000 Hz | 1.420 | 1.010 | 53.4 %  | 10 % |

Local standardization is itself a high-pass filter with its half-power point at 276 Hz, so it
attenuates precisely the band in which q is highest — which is why the time-domain and
frequency-domain results explain each other.

**The q-triplet does not close.** From q_stat = 1.48 the mean relations predict
q_rel = 2.92 and the duality relations predict q_rel = 1.92; the measured value is
0.855, near the Boltzmann-Gibbs value. The sensitivity index is not identifiable in these
records (median 0.976, but it moves from 0.455 to 0.982 across sixteen defensible analysis
choices). On the strength of this internal test the paper withdraws the quasi-critical
reading of the global index.

**What survives is multifractality.** The multifractal spectrum width of the records
(0.568) exceeds every surrogate of themselves (phase-randomized 0.138, time-shuffled
0.046) and exceeds a Student-t series of identical q (0.097) by a factor of
5.9.

**Uncertainty, honestly reported.** The moving-block bootstrap gives a median 95 % interval
width of 0.39 per record and a bootstrap standard deviation of 0.102. The lower bound lies
above the grid floor in 172 of 184 records, but the interval lies entirely below 5/3
in only 98. Anyone planning to use q as a per-record readout should start from these two
numbers.

**Two dependencies that must be declared.** The global index tracks the fraction of samples
replaced at the interpolation stage (Spearman rho = 0.871, p = 2e-57) while the
residual index does not (0.053, p = 0.48); in the 55 records with less
than 0.5 % of samples interpolated the median global index is 1.18 rather than
1.48. The global index also tracks T weakly (rho = -0.219).

---

## 3. Data

The traces are **not** in this repository. They are a public deposit:

> K. A. Ciecierski and T. Mandat, *RAW Deep Brain Stmulation Recordings*,
> Mendeley Data, Version 1 (2023). doi:[10.17632/tr93krswn2.1](https://doi.org/10.17632/tr93krswn2.1)

The scripts expect the downloaded archive, which contains
`<patient>/<inside|outside>/step_<depth>_hemi_<L|R>_<tag>_<position>_processed.npy`
together with one `selected_files.csv` per patient. Point `NEDBS_RAW_ZIP` at it:

```bash
export NEDBS_RAW_ZIP=/path/to/drive-download-20261002T191830Z-1-001.zip      # Linux/macOS
$env:NEDBS_RAW_ZIP = "D:\path\to\drive-download-...-001.zip"                # PowerShell
```

Without that variable, `paths.py` looks for the archive in `raw/`.

**Design.** 184 records from 46 patients, 92 inside and 92 outside the
subthalamic nucleus, two records per patient per side. Sampling 24 kHz; the typical record is
240 000 samples (10.0 s) and the shortest 79 623 (3.32 s). Depths are distances from the
planned target along the trajectory, negative above it.

**State of the traces.** They arrive edge-trimmed, artifact-interpolated and globally
standardized. Two properties were measured rather than assumed: the low-pass corner sits at
2050 Hz (-3 dB) and there is no high-pass stage. The 300-5000 Hz band-pass described in the
companion manuscript is **not** present in these files. The 5-120 Hz spectral slope is -0.12
against -1 to -3 for physiological local field potential, so these traces contain no usable
LFP and nothing here should be read as a statement about the beta rhythm.

**Shipped tables.** `data/` holds about 2 MB of intermediate results — one row per record for
most of them — so that every figure can be redrawn without downloading or reprocessing the
raw traces. Column meanings are documented in the header of the script that writes each table
(Sec. 7).

---

## 4. Repository layout

```
paths.py          ROOT, DATA, FIGOUT, RAW_ZIP (override with NEDBS_RAW_ZIP)
qstat.py          the q-Gaussian grid estimator used throughout the paper
analysis/         raw traces -> per-record tables (needs RAW_ZIP), 23 numbered steps
simulations/      nulls, calibrations, the battery of generative processes
figures/          the eleven figures of the paper and its supplementary material
data/             intermediate tables (shipped) and figure inputs
figures_out/      figures are written here (git-ignored)
```

## 5. Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # or conda create -n nedbs python=3.11
pip install -r requirements.txt

# redraw all eleven figures from the shipped tables (about a minute, no raw data needed)
for f in figures/fig*.py; do python "$f"; done
```

To rebuild the tables from the raw traces, set `NEDBS_RAW_ZIP` and run `analysis/` in
**numerical order** — the numbering is a topological order of the dependency graph, so every
input exists before it is read — then `simulations/` in any order, then `figures/`. Steps that
loop over all 184 records and refit q many times per record (the bootstrap, MFDFA, the
q_rel protocols, the band decomposition) take a long time: budget hours, not minutes, and run
them one at a time. Every script writes into `data/` and is safe to re-run.

---

## 6. Methods, step by step

### 6.1 The estimator (`qstat.py`)

The index is fitted by the linearization that the companion work uses and that is common in
applied q-statistics. For a sample x:

1. trim 0.5 % from each tail and histogram the remainder into 120 bins, normalized as a
   density rho;
2. form the q-logarithm ln_q(rho/rho_max) = [(rho/rho_max)^(1-q) - 1] / (1-q);
3. scan q from 1.01 to 2.59 in steps of 0.01 and keep the q whose ln_q transform is most
   linear in x², judged by the coefficient of determination. The slope of that regression
   gives beta, the R² is retained as a quality measure.

For a true q-Gaussian, ln_q(rho/rho_max) is exactly -beta x², so the correct q is the one
that straightens the curve. The function returns `{'q', 'beta', 'r2'}`.

This estimator is biased low (q_fitted - q_true = -0.056 ± 0.106 in simulation, regression
slope 0.68 against the ideal 1), gives no confidence interval, weights the tail in a way that
depends on the binning, and can pin at a grid edge — 4 records sit at 1.01 and
1 at 2.59. They are retained in every median and paired test, all of which are
rank-based. A maximum-likelihood fit in the Student-t family with q = (nu+3)/(nu+1), on the
untrimmed series subsampled every 23 samples to quasi-independence, is used as a cross-check
and returns a median of 1.390 against 1.48 for the grid.

### 6.2 Null calibration of the estimator (`simulations/s1`)

Gaussian i.i.d. data are passed through the identical estimator at six sample sizes with
40-200 repetitions each. The 95th percentile of the fitted q is the ceiling that any measured
q must clear:

| N | maximum likelihood, q95 | grid estimator (extended below 1), q95 |
|---|---|---|
| 1000 | 1.0930 | 1.1205 |
| 3000 | 1.0518 | 0.8905 |
| 10000 | 1.0240 | 0.9900 |
| 30000 | 1.0151 | 1.0100 |
| 100000 | 1.0072 | 1.0105 |
| 300000 | 1.0046 | 1.0100 |

Two points matter. The grid estimator with its usual floor at 1.01 *censors* the null — it
cannot return a value below the floor — so the null must be recomputed with the grid extended
below unity, which is what the third column does. And the ceiling scales as 1 + c/sqrt(N):
the simulated c has median 2.558 over the six sample sizes, giving q95 = 1.0052 at
N = 240 000, while the asymptotic Cramér-Rao argument gives c = 2.09. Both place the ceiling
between 1.004 and 1.006 at the length of these records, nearly two orders of magnitude below
the measured (q - 1).

### 6.3 The battery of generative processes (`simulations/s2`)

Thirteen processes at N = 10^5, 60 repetitions each, through the same estimator:

| process | median fitted q |
|---|---|
| logistic map (chaos) | 0.600 |
| AR(1), phi=0.99 | 0.940 |
| 1/f noise (H=0.9) | 0.980 |
| AR(1), phi=0.9 | 0.985 |
| Gaussian i.i.d. | 0.990 |
| band-pass Gaussian | 0.990 |
| Gauss + 0.1% outliers (x10) | 0.995 |
| Gauss + 1% outliers (x5) | 1.160 |
| Student-t, nu=10 | 1.167 |
| GARCH(1,1) | 1.250 |
| superstat. n=2.5, T=10^4 | 1.413 |
| Student-t, nu=2.46 | 1.538 |
| superstat. n=2.5, T=50 | 1.540 |

The ordering is the argument. Linear memory of any strength leaves q at unity, because a
linear filter of Gaussian noise is Gaussian however long its memory. The logistic map at the
Ulam point falls *below* unity, as its bounded U-shaped invariant density implies. What lifts
q is a fluctuating variance or a heavy-tailed innovation.

### 6.4 The time scale of the variance fluctuation (`analysis/01`, `analysis/02`)

Under the Beck-Cohen-Swinney construction, if the signal is locally Gaussian with a slowly
varying variance, then the kurtosis computed inside a window of length Delta approaches 3 as
Delta shrinks below the time scale on which the variance moves. The crossing
kappa(Delta) = 3 defines T.

Implementation detail that matters for reading Fig. 3: kappa(Delta) is the **mean over
windows of the kurtosis computed inside each window**, with the window mean removed — not the
kurtosis of the window means. The curve is evaluated on 50 logarithmically spaced window
lengths from 0.25 to 500 ms, requiring at least 8 windows per length.

T is a *monotone proxy*, not a physical time scale. On synthetic superstatistics of known
input scale T0 (`simulations/s4`) the recovered T compresses by factors of 1.8 to 24, growing
with T0. And on Gaussian noise the criterion returns a finite crossing in about half of
realizations by chance (`simulations/s3`, 31 of 60 finite), so a finite T is not by itself
evidence of anything; the paired contrast is.

### 6.5 The residual index on each record's own clock (`analysis/03`)

If the elevation of q is produced by a fluctuating variance, then standardizing the signal
locally — subtracting the mean and dividing by the standard deviation inside short
non-overlapping windows — should remove it. The subtlety is the window length. A window fixed
at 50 ms resolves the fluctuation well in a record with a short T and badly in one with a long
T, so the residual index at a fixed window is largely a readout of T itself (Spearman
rho = -0.898, Fig. 5(b)). The remedy is to let each record set its own clock: Delta = k·T with
the same multiplier k for every record. The paper reports k = 2, 5, 12.4 and 25; k = 12.4
corresponds to a median window of 50.0 ms.

| window | median inside | median outside | contrast | p | d_z |
|---|---|---|---|---|---|
| fixed 50 ms | 1.208 | 1.090 | +0.122 | 3.6e-08 | 1.24 |
| k = 2 | 1.105 | 1.042 | +0.066 | 1.5e-07 | 1.15 |
| k = 5 | 1.162 | 1.075 | +0.093 | 4.7e-08 | 1.20 |
| k = 12.4 | 1.195 | 1.098 | +0.101 | 4.1e-08 | 1.21 |
| k = 25 | 1.220 | 1.122 | +0.110 | 5.3e-08 | 1.20 |

The contrast is present at every multiplier. It is largest at the fixed 50 ms window precisely
because that window lets the T difference leak into the index, which is why the paper leads
with the own-clock values instead.

### 6.6 Surrogates: two kinds, and why both are needed (`analysis/16`, `analysis/17`)

A **time-shuffled** surrogate destroys the temporal order and keeps the marginal exactly. It
is the right control for "is there any temporal structure at all", and it is what shows that
the collapse of q under local standardization is physics rather than an artifact of the
procedure: the surrogate's q does not collapse.

It is, however, a *weak* control for the anatomical contrast, because it destroys the linear
autocorrelation along with the volatility, so it cannot separate them. The decisive control is
the **phase-randomized** (Fourier) surrogate: randomize the phases, keep the modulus, and you
keep the power spectrum exactly while Gaussianizing the marginal and destroying any
amplitude-envelope structure. If the inside-outside difference in T or in the residual index
were a consequence of the two sides simply having different spectra, it would survive this
surrogate. It does not (Sec. 2).

Note that the IAAFT variant preserves the marginal and is therefore *not* the right surrogate
here; the one used is the plain phase randomization of Theiler et al.

### 6.7 Frequency decomposition (`analysis/11`)

Fourth-order Butterworth filters in a second-order-sections implementation (a
transfer-function implementation degenerates at the 1 Hz corner, which is 8e-5 of the Nyquist
frequency), bands 1-100, 100-500 and 500-2000 Hz, each band restandardized before fitting
because the grid estimator bins in absolute units. Each band is compared against a
phase-randomized surrogate of the same record, so that the reported excess is the
non-Gaussianity beyond what the power spectrum alone implies.

The same script measures the transfer function of local standardization directly, as the ratio
of power after to power before, which is how the 276 Hz half-power point of Fig. 6(b) is
obtained.

### 6.8 The q-triplet (`analysis/06`, `analysis/07`, `analysis/08`, `analysis/09`)

- **q_stat** is the index of the stationary amplitude distribution, i.e. the global q.
- **q_rel** is read from the relaxation of the autocorrelation of the 1 ms RMS envelope,
  fitted by a q-exponential. The fit is *not* protocol-invariant: the four protocols give
  medians 1.0, 0.855, 0.805 and 1.06, and the most common one is floor-censored in
  63 % of records. The paper quotes the least censored protocol and shows the full span.
- **q_sen** follows from the Lyra-Tsallis relation applied to the MFDFA singularity spectrum.
  In these records it is **not identifiable**: its median moves from 0.455 to 0.982 across
  sixteen defensible analysis choices, the negative moments do not scale (R² = 0.59 at
  s_min = 1 ms, rising to 0.93 at 20 ms), and f(alpha) exceeds 1 in 173 of 184 records.
  It is reported as a measurement, never as a test.

Two one-parameter relations are tested. The **mean relations** (Tsallis 2009, footnote p. 194)
require, with epsilon = 1 - q, that epsilon_stat be the arithmetic and epsilon_rel the
harmonic mean of the other two, which forces epsilon_sen to be their geometric mean; on the
nontrivial branch they reduce to q_rel = 4 q_stat - 3 and q_sen = 3 - 2 q_stat. The **duality
relations** come from composing the multiplicative duality q -> 1/q with the additive duality
q -> 2 - q into q -> 2 - 1/q, giving q_stat + 1/q_rel = 2 and q_rel + 1/q_sen = 2. Both are
satisfied by the solar-wind triplet (-1/2, 4, 7/4) of Tsallis, Gell-Mann & Sato; apart from
q = 1 they coincide only at q_stat = 7/4, so a triplet away from those two points tells them
apart. Here both fail, and in the same direction.

### 6.9 Multifractality (`analysis/07`)

Multifractal detrended fluctuation analysis with quadratic detrending, moments r in [-5, 5]
excluding 0, scales from s_min to N/10, segments taken from both ends of the series. The
width alpha_max - alpha_min is compared against the record's own surrogates and against
reference processes (`simulations/s5`): records 0.568, chi-squared superstatistics
0.468, phase-randomized surrogate 0.138, Student-t of the same q 0.097, time-shuffled
surrogate 0.046, Gaussian i.i.d. 0.039. Multifractality is the one claim of the paper
that no surrogate reproduces.

### 6.10 Per-record uncertainty (`analysis/15`)

Moving-block bootstrap (Künsch) with 500 ms blocks — 12 000 samples, which leaves six blocks
in the shortest record — and 1000 resamples per record, refitting q on each resample. This is
the only uncertainty statement the grid estimator admits, and it is large: see Sec. 2.

### 6.11 Choosing the marginal family (`analysis/14`)

Five families are fitted by maximum likelihood on the subsampled series and compared by AIC:
Gaussian, Student-t (equivalently the q-Gaussian), generalized Gaussian, symmetric
alpha-stable, and a two-component Gaussian scale mixture. The mixture wins in 161 of
184 records and the q-Gaussian in 15, but the margin over the q-Gaussian is
0.0065 AIC units per sample — negligible. A two-component scale mixture *is* a discrete
superstatistics, so this result is a second, independent statement that the marginal is
produced by a variance that takes more than one value.

### 6.12 Is q a useful descriptor? (`analysis/18`, `analysis/19`, `analysis/22`)

The head-to-head asks a different question from the paired test: how well does each
descriptor order the two sides *within* the same patient, with the direction fixed by
leave-one-patient-out. Area under the curve, bootstrap intervals:

| descriptor | AUC | 95 % interval |
|---|---|---|
| absolute power 500-3000 Hz | 0.928 | [0.878, 0.972] |
| background amplitude (from preprocessing metadata) | 0.911 | [0.861, 0.956] |
| reconstructed amplitude scale | 0.883 | [0.822, 0.939] |
| residual q (fixed 50 ms window) | 0.796 | [0.726, 0.864] |
| time scale T (Beck-Cohen-Swinney) | 0.766 | [0.685, 0.842] |
| relative power 500-3000 Hz | 0.690 | [0.609, 0.761] |
| absolute beta power | 0.628 | [0.539, 0.711] |
| q, whole record | 0.541 | [0.454, 0.625] |
| background / total SD (scale-free) | 0.473 | [0.386, 0.560] |

Read this honestly. Absolute spectral power leads once the lost channel scale is reconstructed
from the preprocessing metadata. Among quantities available in the deposit as standardized
traces, the residual index leads the relative spectral descriptors. The paper claims that q is
a *legitimate* descriptor — well defined, estimable, carrying information that the paired test
detects — and explicitly does not claim it is a *superior* one.

### 6.13 Confounders (`analysis/13`, `analysis/20`, `analysis/23`)

**Firing rate.** The STN fires faster than its surroundings, and firing rate is visible in
high-frequency power. `analysis/13` regresses the indices on relative 500-3000 Hz power at the
record level and repeats the paired contrast on the residuals; with a second band (3-8 kHz)
added the effect sizes shrink further but the contrasts survive. The adjustment is a control,
not a measurement of firing rate.

**Interpolated samples.** `analysis/23` assembles the artifact fraction next to the indices;
the dependence of the global index on it is strong (Sec. 2) and that of the residual index is
absent.

**Anatomical labels.** The file names carry a tag (`STN` or `Outra`) that disagrees with the
inside/outside class of the deposit in 18 records, 12 inside and 6 outside. The class of the
deposit is used throughout; removing those records strengthens both main contrasts.

---

## 7. Script reference

`raw` = reads the deposited traces. Each script carries a docstring naming its inputs, its
outputs and what it computes; the meaning of every column is documented there.

| script | raw | writes into `data/` | reads from `data/` |
|---|---|---|---|
| `analysis/01_bcs_timescale_per_record.py` | yes | bcs_T_por_registro_184.csv | - |
| `analysis/02_bcs_kappa_curves.py` | yes | bcs_kappa_curvas_184.csv | - |
| `analysis/03_residual_q_own_clock.py` | yes | qresid_janela_proporcional_T.csv | resultados_discriminadores_184.csv |
| `analysis/04_q_global_per_record.py` | yes | tripleto_qstat.csv | qresid_janela_proporcional_T.csv |
| `analysis/05_spectrum_per_record.py` | yes | espectro_184.csv | - |
| `analysis/06_qrel_envelope_protocols.py` | yes | tripleto_qrel.csv | - |
| `analysis/07_mfdfa_spectra.py` | yes | tripleto_mfdfa.csv | - |
| `analysis/08_mfdfa_smin20.py` | yes | tripleto_mfdfa_smin20.csv | - |
| `analysis/09_qsen_sensitivity.py` | yes | tripleto_qsen_sensibilidade.csv | - |
| `analysis/10_triplet_table_merge.py` | yes | tripleto_mfdfa_smin20.csv, tripleto_q_184.csv | tripleto_mfdfa.csv, tripleto_qrel.csv, tripleto_qstat.csv |
| `analysis/11_q_by_band.py` | yes | q_por_banda.csv, q_por_banda_resumo.csv | bcs_T_por_registro_184.csv |
| `analysis/11b_q_by_band_rerun.py` | yes | psd_freq.npy, psd_locz_todos.npy, psd_raw_todos.npy, q_por_banda.csv | q_por_banda.csv, qresid_janela_proporcional_T.csv |
| `analysis/11c_q_by_band_rerun_psd.py` | yes | psd_freq.npy, psd_locz_todos.npy, psd_raw_todos.npy, q_por_banda.csv | qresid_janela_proporcional_T.csv |
| `analysis/11d_q_by_band_rerun_psd_raw.py` | yes | psd_freq.npy, psd_locz_todos.npy, psd_raw_todos.npy, q_por_banda.csv | qresid_janela_proporcional_T.csv |
| `analysis/11e_q_by_band_rerun_psd_freq.py` | yes | psd_freq.npy | qresid_janela_proporcional_T.csv |
| `analysis/12_spectral_contrast_stn.py` | yes | espectro_184.csv, espectro_contraste_stn.csv | espectro_184.csv, tripleto_q_184.csv |
| `analysis/13_firing_rate_adjustment.py` | yes | espectro_ajuste_confundidor.csv | tripleto_q_184.csv |
| `analysis/14_marginal_model_selection.py` | yes | selecao_modelo_marginal.csv | stable_alphas.npy, stable_grid.npy, stable_pdfs.npy |
| `analysis/15_bootstrap_block_q_B1000.py` | yes | bootstrap_blocks_q_B1000.csv | - |
| `analysis/15b_bootstrap_block_q_B60.py` | yes | bootstrap_blocos_q.csv | tripleto_qstat.csv |
| `analysis/16_phase_surrogate_T_and_residual_q.py` | yes | m3_phase_surrogate_T_qres.csv | - |
| `analysis/17_phase_surrogate_residual_q_ml.py` | yes | residual_q_ft_control.csv | - |
| `analysis/18_headtohead_features.py` | yes | headtohead_features.csv | espectro_184.csv, qresid_janela_proporcional_T.csv, tripleto_q_184.csv |
| `analysis/19_headtohead_auc.py` | yes | headtohead_auc.csv, headtohead_features.csv | espectro_184.csv, qresid_janela_proporcional_T.csv |
| `analysis/20_preprocessing_metadata.py` | yes | metadados_preproc.csv | headtohead_features.csv |
| `analysis/21_absolute_scale_restored.py` | yes | escala_restaurada.csv | metadados_preproc.csv |
| `analysis/22_absolute_scale_ranking.py` | yes | ranking_escala_absoluta.csv | metadados_preproc.csv |
| `analysis/23_artifact_robustness_merge.py` | no | robustez_artefato.csv | bcs_T_por_registro_184.csv, metadados_preproc.csv, q_por_banda.csv, qresid_janela_proporcional_T.csv, tripleto_qstat.csv |
| `simulations/s1_null_ceiling_vs_N.py` | no | teto_nulo_q_vs_N.csv | teto_nulo_q_vs_N.csv |
| `simulations/s2_generative_process_battery.py` | no | bateria_processos.csv | - |
| `simulations/s3_bcs_gaussian_null.py` | no | bcs_nulo_gaussiano.csv | - |
| `simulations/s4_bcs_synthetic_calibration.py` | yes | bcs_calibracao_sintetica.csv | - |
| `simulations/s5_triplet_reference_processes.py` | yes | tripleto_bateria_nulos.csv, tripleto_bateria_resumo.csv | - |
| `simulations/s5b_triplet_reference_nulls.py` | no | tripleto_bateria_nulos.csv, tripleto_bateria_resumo.csv | - |

### Where the outputs appear in the paper

| paper item | built by |
|---|---|
| Fig. 1 size, clock and residual (synthetic illustration) | no input table -> `figures/fig1_concept.py` |
| Fig. 2 null ceiling and battery | `simulations/s1`, `simulations/s2` -> `figures/fig2_null_ceiling_battery.py` |
| Fig. 3 the clock: kappa curves and calibration of *T* | `analysis/01`, `analysis/02`, `simulations/s3`, `simulations/s4` -> `figures/fig3_clock.py` |
| Fig. 4 global *q*, *T* and residual *q* across the border | `analysis/23`, `analysis/01`, `analysis/03` -> `figures/fig4_dissociation.py` |
| Fig. 5 residual *q* on its own clock | `analysis/03`, `analysis/01`, `analysis/23` -> `figures/fig5_residual_own_clock.py` |
| Fig. 6 *q* by frequency band | `analysis/11` and its re-runs -> `figures/fig6_q_across_bands.py` |
| Fig. 7 *q*-triplet and multifractality | `analysis/06`, `analysis/07`, `analysis/10`, `simulations/s5` -> `figures/fig7_triplet_multifractal.py` |
| Fig. S1 marginal model selection | `analysis/14` -> `figures/figS1_model_selection.py` |
| Fig. S2 bootstrap and artifact sensitivity | `analysis/15`, `analysis/23` -> `figures/figS2_estimator_robustness.py` |
| Fig. S3 *q*-beta and median spectrum | `analysis/23`, `analysis/11c` -> `figures/figS3_qbeta_and_spectrum.py` |
| Fig. S4 head-to-head comparison | `analysis/18`, `analysis/19`, `analysis/22` -> `figures/figS4_head_to_head.py` |
| Table S1 controls | `analysis/16`, `analysis/17`, `analysis/13`, `analysis/01` |
| Null ceiling (Sec. 2.2 and supplementary note S1) | `simulations/s1` |
| Firing-rate adjustment | `analysis/13` |

---

## 8. Provenance of this code, and what was edited

These scripts were recovered from the execution record of the analysis sessions, where they
ran as notebook-style cells rather than as files. Five classes of edit were applied, and
nothing else:

1. **Paths.** Absolute paths into the session artifact store were replaced by
   `DATA / '<name>.csv'`, and the absolute path of the deposit by `RAW_ZIP`.
2. **Estimator import.** Some cells called `qstat_fit_q` from a helper module that was
   preloaded in the session kernel. Where the recovery produced a placeholder instead of the
   real function, the placeholder was deleted and replaced by `from qstat import ...`. The
   affected scripts are:
   - `analysis/15_bootstrap_block_q_B1000.py`
   - `analysis/12_spectral_contrast_stn.py`
   - `analysis/19_headtohead_auc.py`
   - `analysis/11e_q_by_band_rerun_psd_freq.py`
   - `analysis/11c_q_by_band_rerun_psd.py`
   - `analysis/11d_q_by_band_rerun_psd_raw.py`
   - `analysis/11b_q_by_band_rerun.py`
   - `analysis/03_residual_q_own_clock.py`
   - `analysis/06_qrel_envelope_protocols.py`
   - `analysis/04_q_global_per_record.py`
   This substitution was verified, not assumed: for five records spanning the range
   (`01/inside`, `13/outside`, `26/inside`, `38/outside`, `46/outside`) `qstat.py` reproduces
   *q*, beta and R² of `tripleto_qstat.csv` to all stored digits. Scripts that carried their
   own complete implementation were left untouched, so a few define the estimator locally.
3. **Headers.** A docstring naming inputs, outputs and purpose was added to each file.
4. **Figures.** The figure scripts correspond to the figures as submitted, which differ from
   the first drafts: claim-style panel titles were removed in favour of `(a)`, `(b)`, `(c)`,
   spelling was made consistent, the obsolete ceiling constant 2.534 was replaced by the
   simulated 2.558 and the asymptotic 2.09, Fig. 7(a) gained the duality prediction and lost
   the q_sen prediction, and Fig. S2(a) was redrawn from the 1000-resample bootstrap.
5. **Figure set of the present version.** The article has seven main figures and four
   supplementary ones. `fig1_concept.py` is new and uses synthetic data only (fixed seed): two
   signals built from the same samples in a different order, which share their marginal and
   differ in their clock. `fig4_dissociation.py` is new and draws, from the shipped tables,
   the paired per-patient values of the global index, of *T* and of the residual index.
   `fig3_clock.py` keeps the kappa curves and the calibration panel of the former
   `fig2_bcs_timescale.py`, whose paired panel moved to `fig4_dissociation.py`. The scripts
   for the null ceiling, the residual index, the frequency bands and the triplet were
   renumbered to `fig2_`, `fig5_`, `fig6_` and `fig7_`, with no change other than the figure
   number and the name of the output file.

**Verification.** The nine figure scripts of the first version of this repository were run
against the shipped tables on 29 October 2026, and all nine completed. Each output was then
compared with the corresponding submitted figure in two ways. First, every one of the nine was inspected
visually side by side with the submitted version: panels, series, annotations and printed
values agree. Second, because the two renderings do not share a canvas size, they were
resampled onto a common grid and differenced in greyscale; the mean absolute difference lies
between 0.010 and 0.043 on a 0-1 scale, between 1.3 % and 6.2 % of pixels differ by more than
0.25, and the inked fraction of the two images agrees to within 0.4 percentage points. That
residual is what a sub-pixel shift produces, not a change of content.

The outputs are **not** byte-identical to the submitted PNGs, and no pixel-exact equality is
claimed. The delivered figures were produced with the styling helper of the lab's figure
toolchain, which is not redistributed here; `figures/_common.py` reimplements the same
mechanics (size ladder, open frame, 300 dpi), and the small typographic differences move the
tight bounding box by at most 18 px out of about 3800. Anyone needing pixel-exact
reproduction should re-render with the original styling helper.

For the present version all eleven scripts were run again on 12 November 2026, and all eleven
complete. The outputs of `fig1_concept.py`, `fig3_clock.py` and `fig4_dissociation.py` are the
files used in the article, pixel for pixel. The other eight scripts are those verified above,
apart from the renumbering, and their outputs differ from the figures of the article by the
same sub-pixel residual (mean absolute difference between 0.011 and 0.044).

**Re-runs.** Analysis steps 11, 11b, 11c, 11d, 11e are successive runs of the same band
decomposition, and 15/15b are the 1000- and 60-resample bootstraps. They are all kept because
each produced one of the tables that the paper or an earlier figure drew on; the paper reports
step 11 and step 15. Likewise `s5`/`s5b` are the summary and per-realization forms of the same
battery.

**Not recovered.** `data/resultados_discriminadores_184.csv` comes from an earlier exploratory
round whose cell is not part of this set; it is shipped because step 03 reads it. The
alpha-stable lookup arrays (`stable_grid.npy`, `stable_pdfs.npy`, `stable_alphas.npy`) used by
step 14 are shipped for the same reason.

## 9. Caveats that travel with the numbers

- **The published per-record *q* of the companion manuscript cannot be reproduced** by this
  code. The best reimplementation correlates at r = 0.71 per segment (median 1.42–1.50 against
  the published 1.578) across 48 binning configurations. The fitting script of that work is
  absent from its repository. Absolute values here are therefore not comparable with it.
- **Reported medians are floors, not point estimates**, because the estimator is biased low.
- **Firing rate is a confounder** for the inside/outside contrasts. Step 13 adjusts for
  relative high-frequency power and the contrasts weaken but survive; the adjustment is not a
  measurement of firing rate.
- **File-name tags disagree with the class in 18 records** (12 "inside" named `Outra`, 6
  "outside" named `STN`). The class of the deposit is used throughout; removing those records
  strengthens both main contrasts.
- ***T* is a monotone proxy**, not a physical time scale: on synthetic superstatistics of known
  input scale it compresses by factors of 1.8 to 24, and on Gaussian noise the kappa = 3
  criterion returns a finite crossing in about half of realizations by chance (step s3).
- **beta is a nuisance parameter.** The q–beta relation is largely imposed by standardizing the
  traces; build no claim on beta.

## 10. Citation and contact

Please cite the paper above. Correspondence: Bruno Duarte Gomes, brunodgomes@ufpa.br.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE`.
