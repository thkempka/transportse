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
| pH | input only (`C[3]`), stays constant at 7 | output (`C[4]`), read from the selected output |

## Requirements and run

```bash
uv pip install phreeqcrm          # not a dependency of the project
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

## Validation against example 6 (last time step, t = 4 d)

| | example 6 | example 6b | max abs diff | mean diff |
|---|---|---|---|---|
| Ca (mol kgw⁻¹) | 0.000 – 0.0784 | 0.000 – 0.0816 | 2.7e-2 | 6.2e-5 |
| anhydrite (mol m⁻³) | 0 – 176.7 | 0 – 178.7 | 12.4 | 0.02 |
| temperature | — | — | 0.00 K | — |

The remaining differences sit on the steep flank of the Ca front a few metres
around the well (224 of 2601 cells differ by more than 1e-3 mol kgw⁻¹) and stem
from the `K_C` conversion residual plus the Dirichlet cells, where example 6
skips chemistry and 6b does not. `compare_06_vs_06b.py` in `course/` regenerates
the difference plot (`06b-groundwater_heat_tracer_phreeqcrm/compare_06_vs_06b.png`).

## Performance (measured on an i7-1260P, both runs back to back, same load)

| | example 6 | example 6b |
|---|---|---|
| wall time | 288.8 s | 144.9 s (**2.0× faster**) |
| per time step | 0.339 s | 0.170 s |
| pressure / species / heat iterations | 13 939 / 5 465 / 1 675 | 6 373 / 11 633 / 2 302 |

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
