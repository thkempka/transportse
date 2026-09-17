```
    ___________  ___   _   _  ___________ ___________ _____ _____ _____
   |_   _| ___ \/ _ \ | \ | |/  ___| ___ \  _  | ___ \_   _/  ___|  ___|
     | | | |_/ / /_\ \|  \| |\ `--.| |_/ / | | | |_/ / | | \ `--.| |__
     | | |    /|  _  || . ` | `--. \  __/| | | |    /  | |  `--. \  __|
     | | | |\ \| | | || |\  |/\__/ / |   \ \_/ / |\ \  | | /\__/ / |___
     \_/ \_| \_\_| |_|\_| \_/\____/\_|    \___/\_| \_| \_/ \____/\____/
+---| TRANSPORT Simulation Environment---------------------------------------+
```

Python program to calculate density-driven fluid flow in porous media, and
advective, diffusive and dispersive transport of chemical species as well as
conductive and convective transport of heat.

## Features

**Fluid flow**
- Finite-difference discretisation of density-driven (variable-density)
  Darcy flow in 3-D on structured grids with variable node spacing
  (nx, ny, nz)
- Pressure solver: red-black SOR (default) or preconditioned Conjugate
  Gradient (`PSOLVER = "CG"` in the config file), both based on a
  precomputed 7-point stencil
- Gravity effects and sloped models (model slope angle beta)
- Red-black successive over-relaxation (SOR) pressure solver
- Fluid storage (compressibility of fluid and rock) and source/sink terms
- Sequential Iterative Approach (SIA): outer Picard iteration on the
  fluid density for coupled flow–transport solution, relaxing the CFL
  time-step constraint
- Automatic time-step control: convergence-based, CFL and Neumann criteria

**Species transport**
- Advective (upwind), diffusive and hydrodynamic-dispersive transport of
  multiple chemical species (longitudinal and transversal dispersivity)
- Retardation and lithology-dependent properties
- Implicit/explicit temporal weighting (OMEGA, 0 <= OMEGA <= 1)
- Immobile-fluid storage correction for transient flow (IMBLFLO)
- Smolarkiewicz correction of numerical dispersion
- Chemical reactions via PHREEQC (phreeqpy) or user-defined surrogate
  models, executed in parallel (joblib)

**Heat transport**
- Conductive and convective heat transport with rock and fluid heat storage
- Temperature-dependent thermal conductivity and heat capacity
- Two-phase (non-local thermal equilibrium, NLTE) option
- Hydrodynamic heat dispersion

**Equation of state**
- User-definable fluid and rock properties (density, viscosity,
  compressibility, heat capacity, thermal conductivity, diffusion
  coefficients) as functions of temperature, concentration and pressure

**General**
- Dynamic boundary conditions (time-dependent Dirichlet/Neumann)
- Mass balance tracking (per species, water volume and heat)
- Diagnostics: Peclet numbers, L2 convergence history, flow balance error
- Parallelised with Numba (selectable threading layer and thread count)
- Output: ParaView VTS/PVD files (tse2pvd.py), mass balance CSV,
  pickle snapshots for restart and post-processing

## Installation

Requires [uv](https://docs.astral.sh/uv/); creates a local `.venv`:

    uv sync

## Usage

    uv run python tse.py <config file>
    uv run python tse.py config.py 2>&1 | tee config.log
    uv run python tse2pvd.py [output file prefix]

- `tse.py`: simulation code (requires a custom config file, see the
  examples in `course/` and `benchmarks/`)
- `tse2pvd.py`: converts simulation output to ParaView format (vts/pvd)
  and writes massbal.csv

## License and citation

GPL v3 — see COPYING and LICENSE.

Kempka, T. (2020): Verification of a Python-based TRANSPORT Simulation
Environment for density-driven fluid flow and coupled transport of heat and
chemical species. Advances in Geosciences, 54, 67-77.
https://doi.org/10.5194/adgeo-54-67-2020
