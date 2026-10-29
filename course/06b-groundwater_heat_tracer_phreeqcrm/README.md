# Example 6b — same model as example 6, chemistry via PhreeqcRM

Same physical model as `06-groundwater_heat_tracer_phreeqc` (51 × 51 × 1
geothermal reservoir, 4 days, 50 m³/h injection of 20 °C water into a 100 °C
reservoir saturated with respect to anhydrite, tracer + Ca transport), but the
reaction step is done by the **PhreeqcRM** reaction module (USGS) instead of
one `IPhreeqc` instance called per cell via phreeqpy.

| | example 6 | example 6b |
|---|---|---|
| reaction engine | phreeqpy (`IPhreeqc`) | PhreeqcRM |
| call pattern | one `run_string()` per cell and time step (~2.2 M calls) | one `RunCells()` per time step (852 calls) |
| parallelisation | joblib workers (`parallel_chemistry`) | OpenMP inside the module (`parallel_chemistry` threads) |
| pH | output (`C[3]`, written back from the selected output) | output (`C[4]`) |

## Requirements and run

```bash
uv sync                             # phreeqcrm is a project dependency
../../.venv/bin/python ../../tse.py config.py
```

`tse.py` calls `config.chemical_reactions_all(C, P, T, rho_f, phi, Kx, Ky, Kz,
dt_chem)` once per time step if the config defines it; otherwise the per-cell
`chemical_reactions()` path (examples 1–10) is used unchanged.

## Species layout

```
C[0] tracer       transported
C[1] Ca           transported        -> module component 'Ca'
C[2] S(6)         transported        -> module component 'S'
C[3] anhydrite    output             <- selected output 'Anhydrite'
C[4] pH           output             <- selected output 'pH'
```

S(6) is a real transported species here (SPTRANS = [0, 1, 2]), because PhreeqcRM
needs every component in its transport array. Example 06 needs one species less:
it writes `S str(C[1])` into the PHREEQC input, i.e. ties sulphur to Ca inside
the solution definition. Both are equivalent, since anhydrite dissolution and
precipitation move Ca and S(6) in a 1:1 ratio - the per-cell difference between
C[1] and C[2] stays at ~1e-11. The well element is the injectate in both cases
(Ca = S(6) = 0, no anhydrite, pH 7), held constant as a Dirichlet cell.

## Two things the coupling needs

1. **Units.** PhreeqcRM derives its own water mass per cell from porosity,
   saturation and representative volume, and its own density from the
   database. Both enter the conversion between TSE concentrations and the
   module's transport array, so the factor `K_C` and the cell volume
   `V_CELL_MOD` are *measured* once at config import instead of assumed, and
   the initial mineral amount is scaled accordingly. Note that
   `SetUnitsPPassemblage(1)` would rescale the initial mineral by the water
   content of a cell — it is left at the module default (moles per cell).
2. **Redox.** The module transports the complete master set (H, O, Charge, Ca,
   Cl, Na, S), TSE only the solutes in `SPTRANS`. Sulfur is therefore
   transported as S(6) and the module's total oxygen is kept consistent with it
   (`O = O_base + 4 S(6)`, with `O_base` re-derived from the module state after
   every `RunCells()`). Without this, PHREEQC re-speciates part of the sulfur
   as sulfide, the redox state drifts and pH goes basic (~10 instead of ~6.5).

## Transport options

`C_IMBLFLO = T_IMBLFLO = True`: the flow balance error `e` of the pressure
solution is added as a source term to the species and heat transport equations
(`C/ρ_f · e` and `cp_f · T · e`). In this model the effect is below the fourth
significant digit — `e` is −1.2e-13 … 4.0e-3 kg m⁻³ s⁻¹ (mean 1.8e-4), so the
correction is negligible against the transport terms; the terms only matter for
a poorly converged flow field. Example 6 keeps both flags off, which is why the
comparison below is quoted to four digits rather than bit-exactly.

## Validation against example 6 (last time step, t = 4 d)

| | example 6 | example 6b | max abs diff | mean diff |
|---|---|---|---|---|
| Ca (mol kgw⁻¹) | 0.000 – 0.0784 | 0.000 – 0.0815 | 2.7e-2 | 2.7e-5 |
| anhydrite (mol m⁻³) | 0 – 176.7 | 0 – 178.7 | 12.2 | 0.02 |
| pH (–) | 6.13 – 7.01 | 6.13 – 7.04 | 0.059 | −6.5e-4 |
| temperature | — | — | 0.00 K | — |

pH agrees to 0.07, Ca and anhydrite to 2.7e-2 mol kgw⁻¹ and 12.4 mol m⁻³. The
remaining differences sit on the steep flank of the Ca front a few metres
around the well (224 of 2601 cells differ by more than 1e-3 mol kgw⁻¹) and stem
from the `K_C` conversion residual plus the Dirichlet cells, where example 6
skips chemistry and 6b does not. `compare_06_vs_06b.py` in `course/` regenerates
the difference plot (`06b-groundwater_heat_tracer_phreeqcrm/compare_06_vs_06b.png`).

### The pH artefact in the leached core

The pH field of 06b is not as smooth as the one of example 06. Two effects were
separated by measurement:

**1. The 0.07 spikes (fixed).** An earlier version showed two narrow spikes at
7.07 instead of ~7.00 in the cells where the anhydrite had been dissolved
completely (`anhydrite = 0`, `Ca ≈ 0.0033 mol kgw⁻¹`). A dump of the module's
speciation there showed only 44 % of the total sulfur as `SO4²⁻`, the rest as
`H2S(aq)`/`HS⁻`, leaving 0.0036 eq kgw⁻¹ of cation excess. Cause: with the
oxygen total at `O_base + 4 S(6)` the module's dissolved O(0) sits exactly on
the redox boundary, where part of the sulfate is reduced. Fix: the pore water of
this model is oxic, so the redox state is defined explicitly by keeping
`O2_MIN = 1e-4 mol L⁻¹` (3.2 mg L⁻¹) of dissolved oxygen in the oxygen total.
No sulfide forms any more and the spikes are gone (max pH 7.038, no cell above
7.05).

**2. The remaining ±0.02 wiggle (not redox).** Cell-to-cell pH still varies by
about ±0.02 in the leached core (6.968 … 7.023 along the injection row, where
example 06 gives a smooth 7.006). This is *not* sulfide: it was tested with a
copy of `phreeqc.dat` from which the S(-2) master species and all 25 solution
species and phases that need it were removed (so sulfur can only be S(6)). That
database loads and runs, and the wiggle is **unchanged** (6.968 … 7.023, same
amplitude, same cells). Other attempts:

| attempt | result |
|---|---|
| set the `Charge` component to 0 each step | worse: pH 7.13 – 7.36, 60 cells > 7.05 |
| pin the `H` component to its initial value | worse: pH 7.13, Ca max +5 % |
| `O2_MIN = 1e-3` / `1e-2` | wiggle 7.029 / 7.013, but at 1e-2 Ca max drops 10 % |

The transported fields themselves are not the cause either: feeding 06b's exact
per-cell `Ca`, `S(6)`, `T` and anhydrite values from the last time step to a
*fresh* IPhreeqc instance (example 06's input template) gives a smooth pH of
7.0062 – 7.0069 across the leached core, while the module reports
6.968 – 7.023 for the same cells. Two independently transported species
agreeing to ~1e-11 could not move pH by 0.02 (that needs ~3e-8 eq kgw⁻¹), which
confirms the measurement.

So the residual comes from the module's per-cell charge bookkeeping, which is
not reachable through the public API. It is cosmetic: 0.02 pH units is 5 % of
[H⁺] in water that carries 4 % of the plume's Ca, and Ca and anhydrite are
identical to the run without any pin (0.0815 / 178.7).

## Performance (measured on an i7-1260P, both runs back to back, same load)

| | example 6 | example 6b |
|---|---|---|
| wall time | 288.8 s | 136 – 145 s (**≈2.0× faster**) |
| per time step | 0.339 s | 0.164 – 0.170 s |
| pressure / species / heat iterations | 13 939 / 5 465 / 1 675 | 6 373 / 11 633 / 2 302 |

Run-to-run spread on a busy laptop is about ±15 %; the ratio is stable across
repeated measurements.

Chemistry thread count no longer matters after the switch:

| `parallel_chemistry` | wall time |
|---|---|
| 2 | 138.6 s |
| 4 | 144.9 s |
| 8 | 141.8 s |

All three runs produce identical iteration counts and fields — the chemistry is
now only ~10–20 ms per time step, and the remaining ~0.14 s is the
single-threaded transport/flow solver (`parallel_processes`), which is the next
lever for speedup.
