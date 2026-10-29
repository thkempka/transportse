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
| Ca (mol kgw⁻¹) | 0.000 – 0.0784 | 0.000 – 0.0816 | 2.7e-2 | 6.2e-5 |
| anhydrite (mol m⁻³) | 0 – 176.7 | 0 – 178.7 | 12.4 | 0.02 |
| pH (–) | 6.13 – 7.01 | 6.13 – 7.07 | 0.069 | −1.4e-3 |
| temperature | — | — | 0.00 K | — |

pH agrees to 0.07, Ca and anhydrite to 2.7e-2 mol kgw⁻¹ and 12.4 mol m⁻³. The
remaining differences sit on the steep flank of the Ca front a few metres
around the well (224 of 2601 cells differ by more than 1e-3 mol kgw⁻¹) and stem
from the `K_C` conversion residual plus the Dirichlet cells, where example 6
skips chemistry and 6b does not. `compare_06_vs_06b.py` in `course/` regenerates
the difference plot (`06b-groundwater_heat_tracer_phreeqcrm/compare_06_vs_06b.png`).

### The two pH spikes

The pH profile of 06b has two narrow spikes (7.07 instead of ~7.00) at the
flanks of the leached core, where the anhydrite has been completely dissolved:
all 12 cells above pH 7.05 have `anhydrite = 0` and `Ca ≈ 0.0034–0.0053
mol kgw⁻¹`, i.e. almost pure water.

Feeding undersaturated water into a cell with finite mineral shows pH creeping
up as the phase is consumed (7.011 → 7.022 → 7.026 at anhydrite 0.139 → 0.092 →
0.018 mol), so the spike is the **exhaustion endpoint of the phase**: the module
keeps the phase in the assemblage at exactly 0 moles while the phase boundary
(SI = 0) is still enforced, and in the resulting nearly pure water a residual of
~1e-5 in the strong-ion balance moves pH by 0.07 units.

The same solution state fed to a *fresh* PHREEQC instance (`Ca = S(6) = 0.0034
mol kgw⁻¹`, T = 20 °C, no phase) gives pH 7.0067, i.e. the artefact is not a
property of the composition but of the exhausted phase in the module's per-cell
state. Example 06 cannot show it: it runs all 2601 cells through **one** shared
IPhreeqc instance, so the complete solution (pH/charge, Na, Cl, water) has to be
re-specified for every cell - there is no per-cell state to carry over. A test
that does carry the state over in a single instance gives pH exactly 7.0000
throughout, i.e. a different behaviour again, not the spikes.

## Performance (measured on an i7-1260P, both runs back to back, same load)

| | example 6 | example 6b |
|---|---|---|
| wall time | 288.8 s | 139.9 – 144.9 s (**≈2.0× faster**) |
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
