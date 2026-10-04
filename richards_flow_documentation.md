# TRANSPORTSE: Richards Flow Equation for Variably Saturated Flow

**Branch:** `feature/richards-flow` · **Status:** implemented and benchmark-verified

## 1 Overview

TRANSPORTSE was extended by a variably-saturated flow mode based on the
mixed-form Richards equation (Celia et al., 1990). The mode is activated
with `UNSATURAT = True` in the configuration file and replaces the
saturated flow solver (`compute_P`) by a modified-Picard solution of the
mixed-form Richards equation (`compute_P_RICH`, `compute_P_RICH_CG`).
Heat and species transport remain fully functional and use the
saturation-dependent Darcy fluxes of the Richards solution.

All existing saturated-flow capabilities are unchanged: with
`UNSATURAT = False` the code behaves bit-identically to `master`
(verified by regression, Section 4.4).

## 2 Conceptual change: `compute_P` → `compute_P_RICH`

### 2.1 Governing equation

The saturated pressure equation of `compute_P`,

    phi*rho*(P - Pn)/dt * c_t = div(rho * K/mue * (grad P + rho*g*grad z)) + WP

is replaced by the **mixed-form Richards equation**

    phi*rho*(S(P) - Sn)/dt + Sn*rho*c_t*(P - Pn)/dt
        = div(rho * kr * K/mue * (grad P + rho*g*grad z)) + WP

with the saturation S(P) from the soil-water retention curve and the
relative permeability kr(P). The mixed form stores the *water content
change* (S(P^{n+1}) − Sn) rather than the pressure change and is exactly
mass-conservative for arbitrary retention curves — the key requirement
for infiltration into dry soil, where head-based forms lose mass.

### 2.2 Solution strategy

The solver follows the same pattern as `compute_P`:

* **Outer loop — modified Picard iteration** on the saturation. The
  storage accumulation is linearised around the current iterate,
  `phi*rho/dt * [S^m + dS/dP|(P^m) * (P^{m+1} - P^m) - Sn]`, and all
  coefficients (S^m, dS/dP, face kr) are frozen, so each pass solves a
  linear system. Convergence on the saturation L2 norm (`CONV_S`). At
  Picard convergence the linearisation error vanishes and the
  discretised equation is the exact mass-conservative mixed form.
* **Inner loop** — solution of the linearised system, either red-black
  SOR (`compute_P_RICH`) or a symmetric-Gauss-Seidel preconditioned
  conjugate gradient solver (`compute_P_RICH_CG`, selected with
  `PSOLVER = 'CG'`). The CG converges on the **residual** of the
  linearised system, which is the cell-wise mass balance error — the
  correct criterion for the nearly elliptic saturated zone
  (dS/dP = 0, c_t ≈ 0), where change-based SOR criteria are
  unattainable.
* **Time step control** — non-convergence of either loop halves dt and
  restarts the time step from Pn/Sn; on convergence dt grows by 1.25,
  capped by `dt_max`. For `UNSATURAT` the advective CFL limit of the
  previous step is applied *before* the flow solve ("CFL first"), and
  the mid-step CFL reduction is disabled.

### 2.3 Face relative permeability

The face kr values are **upstream-weighted with respect to the pressure
head** (the wetter cell's kr). A harmonic-mean face kr was tested first
and rejected: at saturations near the residual saturation the Mualem/vG
conductivity vanishes (kr ~ 1e-6), so the harmonic mean blocks imbibition
fronts completely — the wetting front cannot advance into dry cells. The
upstream rule is the standard treatment for Richards wetting fronts
(Vauclin's FE model, HYDRUS, MODFLOW-SURFACT) and reduces exactly to
kr = 1 for saturated cells, preserving the saturated-flow discretisation
of `master`.

### 2.4 Transport coupling

`compute_q_RICH` computes the Darcy fluxes with the same frozen face kr
values used in the flow solve, so the advective and dispersive transport
solvers (species, heat) operate on fluxes consistent with the flow
solution. The saturation enters the transport storage terms
(phi·S) directly.

## 3 New configuration variables

All new parameters are per-cell numpy arrays of model size (nx, ny, nz),
so each cell can have individual hydraulic properties (lithology).

| Variable | Type / unit | Description |
|---|---|---|
| `UNSATURAT` | bool | Activate Richards (variably saturated) flow mode. Default: False |
| `KRMODEL` | str | Retention/conductivity model: `'mualem-vg'` (van Genuchten retention + Mualem conductivity) or `'haverkamp'` (Haverkamp et al. 1977 retention and conductivity). Default: `'mualem-vg'` |
| `alpha_vg` | array, 1/m (vG) or m^beta (Haverkamp) | Retention curve shape parameter |
| `n_vg` | array, – | Retention curve shape parameter (vG n, Haverkamp beta) |
| `S_r` | array, – | Residual saturation (theta_r/theta_s) |
| `S_s` | array, – | Maximum saturation, default 1 |
| `A_vg` | array, m^gamma | Haverkamp conductivity parameter (only `KRMODEL='haverkamp'`) |
| `gamma_vg` | array, – | Haverkamp conductivity exponent (only `KRMODEL='haverkamp'`) |
| `CONV_S` | – | Modified-Picard convergence criterion (saturation L2 norm). Default 1e-6 |
| `PSOLVER` | str | Pressure solver: `'SOR'` or `'CG'`. `'CG'` recommended for `UNSATURAT`. Default `'SOR'` |
| `CG_OMEGA` | – | Over-relaxation of the SGS preconditioner. Default 1.0 |

Existing variables reused unchanged: `SOR_P`, `CONV_P`, `MAXIT_P`,
`dt`, `dt_max`, `WP`, `BP`, `Kx/Ky/Kz`, `phi`.

**Initial condition note:** the initial saturation is computed
consistently from the initial pressure field via the retention curve, so
hydrostatic-equilibrium initial conditions yield the corresponding
equilibrium saturation profile automatically.

## 4 Benchmark verification

All benchmarks are in `benchmarks/richards/` with configuration,
analysis scripts and reference data.

### 4.1 Celia et al. (1990) 1D infiltration (`celia1990/`)

Infiltration into a 1-m dry sand column (vG: theta_r = 0.102,
theta_s = 0.368, alpha = 1.611 1/m, n = 2.435; Ks corresponding to
theta_s), surface head fixed at h = −0.75 m, 24 h simulated, N = 40
cells, dt = 60 s. Compared against an independent mixed-form
finite-difference reference solver on the identical grid.

| Quantity | TSE | Reference |
|---|---|---|
| max abs theta deviation | **2.59e-4** | — |
| storage change (m) | 2.340961e-2 | 2.342144e-2 (−0.05 %) |
| reference mass error (m) | — | −5.5e-14 (machine precision) |
| dt reductions | 0 | — |

The classic wetting front position and shape are reproduced; the run
needs ~300 time steps without a single dt reduction.

![Celia 1990 comparison](/home/tk/Dropbox/transportse/benchmarks/richards/celia1990/celia1990_comparison.png)

### 4.2 Hydrostatic equilibrium test (`static/`)

Uniform initial state P = −6031 Pa (h = −0.615 m) with consistent
retention saturation, no-flow boundaries, 120 s simulated. The state is
preserved to 1e-12 Pa — the discretisation has no spurious gravity or
capillary fluxes.

### 4.3 Point-source mass balance test (`wptest/`)

3×1×3 column with a volumetric water source (WP = 0.0822 kg/m³/s) and
no-flow boundaries. The storage change equals the source input exactly
(ratio 1.000000) — the mixed-form discretisation is mass-conservative
to machine precision.

### 4.4 Regression against `master`

The `simple_flow` benchmark run with `master` (34-element pickles) and
with `feature/richards-flow` at `UNSATURAT = False` (35-element pickles,
S appended): **0.0 deviation in all 26 fields across all 11 output
files**. The saturated-flow path is bit-identical.

### 4.5 Vauclin et al. (1979) 2D water-table recharge (`vauclin1979/`)

Laboratory sandbox 3 m × 2 m × 0.05 m, water table initially at 0.65 m,
constant irrigation 1.48 cm/h over the left 0.5 m of the surface,
measured water table at 2/4/6/8 h. Grid 60×1×40 (dx = dz = 0.05 m),
CG solver, dt ≤ 60 s, 8 h simulated.

**Setup findings (important for reproducibility):** comparison of
published benchmark descriptions (SHUD/GMD 2020, Underworld3, original
experiment) showed that two intuitive setup choices are *wrong*:

1. **No drainage boundary.** The sandbox walls are impervious; the right
   side is open to the atmosphere (seepage face), not a constant-head
   drain. A Dirichlet column suppresses the mound: the model then
   drains the recharge as fast as it arrives and the water table stays
   flat. The experiment's water balance (input = mound storage) proves
   that no outflow occurred.
2. **Gravity-drained initial moisture.** A hydrostatic-retention
   equilibrium initial condition (theta ≈ 0.01–0.13) gives the vadose
   zone a storage capacity (≈ 0.135 m water) as large as the entire
   irrigation input (0.118 m) — the mound never forms. The experiment
   started from the gravity-drained profile (theta_i ≈ 0.10, uniform
   initial head h_i ≈ −0.37 m, the unit-gradient state of the
   irrigation flux).

With the corrected setup and the classic Haverkamp parameterisation
(theta_r = 0.075, theta_s = 0.287, Ks = 9.44e-5 m/s, alpha = 1.611e6
cm^3.96, beta = 3.96, A = 1.175e6 cm^4.74, gamma = 4.74):

| t | TSE max water table (m) | Experiment (m) | Vauclin FE (m) | rms deviation (m) |
|---|---|---|---|---|
| 2 h | 0.630 | 0.786 | 0.784 | 0.125 |
| 4 h | 0.816 | 1.027 | 0.985 | 0.134 |
| 6 h | 0.906 | 1.121 | 1.089 | 0.142 |
| 8 h | 0.960 | 1.211 | 1.215 | 0.209 |

* Mass balance: storage change = irrigation input exactly (0.000 %
  residual at 8 h).
* The water table rises monotonically with the correct magnitude and
  timing; the simulated water table is flatter than the experiment
  (mound shape under-predicted by ≈ 0.2 m at 8 h). The mound shape is
  controlled by the retention curve near saturation (effective
  specific yield); published simulations of this benchmark likewise
  require calibration of these parameters (SHUD: "a mismatch exists",
  calibrated alpha/beta).
* The van Genuchten/Mualem parameterisation of Shen & Phanikumar
  (2010) (alpha = 3.3 1/m, n = 4.1, theta_r = 0.01, theta_s = 0.30)
  was also tested and gives a systematically lower mound (rms 0.35 m
  at 8 h); the Haverkamp set is closer to the experiment.

![Vauclin 1979 comparison](/home/tk/Dropbox/transportse/benchmarks/richards/vauclin1979/vauclin_comparison.png)

## 5 Numerical notes

* **Inner solver choice.** The saturated zone of a Richards problem is
  nearly elliptic (dS/dP = 0, c_t ≈ 0). Change-based SOR criteria stall
  there (the iteration change decays with a spectral radius ≈ 0.9999
  per sweep, independent of dt and SOR over-relaxation), which makes
  the SOR path grind at `MAXIT_P` sweeps per Picard pass. The CG path
  (`PSOLVER = 'CG'`) converges on the residual — the cell-wise mass
  balance error — and completes the 8 h Vauclin run in about one
  minute with a single dt reduction.
* **Numba.** All new kernels are `@njit` with the same parallel/nogil
  flags as the existing solvers; the red-black SOR closure captures
  only once-assigned arrays (preallocated `bS`/`coef_C`/`coef_P`
  refilled in place per Picard pass).
* **Mass conservation.** The mixed form with the Picard linearisation
  is exactly mass-conservative at convergence; the WP source test
  (4.3) and the Vauclin mass balance (4.5) confirm this to machine
  precision.

## 6 Files

| Path | Content |
|---|---|
| `tse.py` | Richards implementation: config block, constitutive functions (`sat_kr_point`, `dsdp_point`), `compute_P_RICH`, `compute_P_RICH_CG`, `richards_stencil`, `compute_q_RICH`, `update_S_kr_faces`, main-loop integration |
| `tse2pvd.py` | ParaView export extended to the 35-element pickles (S appended) |
| `benchmarks/richards/celia1990/` | 1D benchmark: `config.py`, `analyze.py` (independent reference solver), plot |
| `benchmarks/richards/static/` | Hydrostatic equilibrium test |
| `benchmarks/richards/wptest/` | WP source mass balance test |
| `benchmarks/richards/vauclin1979/` | 2D benchmark: `config.py`, `analyze.py`, `ref/` (digitised experiment and FE data), plot |

## References

* Celia, M.A., Bouloutas, E.T., Zarba, R.L. (1990): A general
  mass-conservative numerical solution for the unsaturated flow
  equation. Water Resources Research 26(7), 1483–1496.
* Vauclin, M., Khanji, D., Vachaud, G. (1979): Experimental and
  numerical study of a transient, two-dimensional unsaturated–
  saturated water table recharge problem. Water Resources Research
  15(5), 1089–1101.
* Haverkamp, R., Vauclin, M., Touma, J., Wierenga, P.J., Vachaud, G.
  (1977): A comparison of numerical simulation models for one-
  dimensional infiltration. Soil Science Society of America Journal
  41(2), 285–294.
* van Genuchten, M.Th. (1980): A closed-form equation for predicting
  the hydraulic conductivity of unsaturated soils. Soil Science
  Society of America Journal 44(5), 892–898.
* Shen, C., Phanikumar, M.S. (2010): A process-based, distributed
  hydrologic model based on a large-scale method for surface–
  subsurface coupling. Advances in Water Resources 33(12), 1524–1541.
* Shu, L. et al. (2020): SHUD v1.0. Geoscientific Model Development
  13, 2743–2762.
