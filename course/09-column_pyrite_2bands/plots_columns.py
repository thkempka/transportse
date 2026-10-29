'''
    plots_columns.py - plotting workflow for the 1D column scenario 2
    (config.py), non-interactive reproduction of
    Plots_columns.ipynb.

    pipeline
        uv run python -u ../../tse.py    config.py
        uv run python    ../07-column_conservative/tse2csv.py            # *.p -> simulation_output.csv
        uv run python    plots_columns.py         # -> comb.csv, figs/

    figures (all written to figs/, nothing else in the folder is overwritten)
        1 profiles_vs_battistel_pub  notebook cell 2 (column1_comparison.pdf)
        2 profiles_vs_battistel      notebook cell 1 (marker plot)
        3 runs_old_vs_new            overlay of --sim and --compare (optional)
        + comb.csv                   simulation + Battistel profiles (cell 1/2)

    options
        --sim CSV       simulation table to plot (default simulation_output.csv)
        --suffix STR    appended to every output name, so a second run (e.g. a second run) creates its own files:
                            --sim simulation_output_2.csv --suffix _2
        --compare CSV   additionally draw this simulation together with --sim
                        and Battistel -> figs/runs_old_vs_new<suffix>.*

    notes
      * Column 2 carries TWO pyrite inclusions; the shaded bands are taken
        from the lithology column L == 2 of the CSV (3.55-5.65 cm and
        7.55-9.65 cm) instead of the hard coded axvspan(3.45,5.45) /
        axvspan(7.45,9.45) of the notebook.
      * notebook cell 0 (integrated_data.csv) additionally merges the lab
        breakthrough file Battistel_1D_Ox_raw.csv, which is not part of this
        scenario folder -> that step is skipped when the file is missing.
      * unlike scenario 1 there is no Battistel Fe/S reference file here.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(CWD)
FIGS = 'figs'
os.makedirs(FIGS, exist_ok=True)

SIM_CSV = 'simulation_output.csv'
OBS_CSV = 'Battistel_1D_Ox_raw.csv'      # lab O2 data (not shipped here)
BAT_PROFILES = 'Bat_col2.csv'            # Battistel et al. O2 profiles, column 2
Z_OBS = [7.5, 12.5]                      # sensor positions used in cell 0 [cm]


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, '%s.%s' % (name, ext)),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  figs/%s.{pdf,png}' % name)


def near(values, targets, atol=1e-9):
    '''boolean mask: values within atol of any of the targets'''
    return np.abs(values[:, None] - np.asarray(targets)[None, :]).min(axis=1) <= atol


def inclusion_bands(sim):
    '''contiguous Z ranges [cm] of the pyrite inclusion (L == 2)'''
    z = np.sort(sim.loc[sim['L'] == 2, 'Z'].unique()) * 100
    if not len(z):
        return []
    dz = np.median(np.diff(np.sort(sim['Z'].unique()))) * 100
    bands, start, prev = [], z[0], z[0]
    for v in z[1:]:
        if v - prev > 1.5 * dz:
            bands.append((start, prev))
            start = v
        prev = v
    bands.append((start, prev))
    return bands


def integrated_data(sim, int_csv):
    '''notebook cell 0: O2 time series at both sensors + lab data'''
    if not os.path.exists(OBS_CSV):
        print('\n[1] %s SKIPPED - %s not found '
              '(lab data of another scenario, see notebook cell 0)'
              % (int_csv, OBS_CSV))
        return
    mask = np.zeros(len(sim), dtype=bool)
    for z in Z_OBS:
        mask |= np.isclose(sim['Z'], z / 100, atol=1e-3)
    sel = sim.loc[mask].copy()
    sel['Z'] = sel['Z'].round(3) * 100
    sel['Origin'] = 'simulation'
    sim_series = sel.rename(columns={'O(0) - normalised': 'O2'})[
        ['Simulation Time (hours)', 'Z', 'O2', 'Origin']]

    lab = pd.read_csv(OBS_CSV)
    lab['O2'] = pd.to_numeric(lab['O2'], errors='coerce')
    lab = lab.dropna(subset=['O2']).rename(
        columns={'Time': 'Simulation Time (hours)'})
    out = pd.concat([sim_series,
                     lab[['Simulation Time (hours)', 'Z', 'O2', 'Origin']]],
                    ignore_index=True)
    out = out.sort_values(['Simulation Time (hours)', 'Z'], ignore_index=True)
    out.to_csv(int_csv, index=False)
    print('\n[1] %s written (%d simulation rows, %d lab rows)'
          % (int_csv, len(sim_series), len(lab)))


def hour_slice(sim, hours):
    '''simulation profiles at the requested hours, Z in cm, as in the notebook.
    Keeps exactly one output per hour - the time step closest to it (PLOTTIME of
    this scenario is 0.303 h apart, so a plain round() filter would stack three
    time steps per hour; notebook cells 1/2 do the same).'''
    sim_h = sim.copy()
    sim_h['Z'] = sim_h['Z'] * 100
    sim_h['Nearest Hour'] = sim_h['Simulation Time (hours)'].round()
    sim_h['dt_off'] = (sim_h['Simulation Time (hours)']
                       - sim_h['Nearest Hour']).abs()
    times = sim_h[['Nearest Hour', 'Simulation Time (hours)',
                   'dt_off']].drop_duplicates('Simulation Time (hours)')
    times = times[times['Nearest Hour'].isin(hours)]
    chosen = times.loc[times.groupby('Nearest Hour')['dt_off'].idxmin(),
                       'Simulation Time (hours)'].to_numpy()
    sel = sim_h[near(sim_h['Simulation Time (hours)'].to_numpy(),
                     chosen)].copy()
    sel = sel[sel['Z'] <= 12.5]                  # as in the notebook
    sel['Origin'] = 'My_sim'
    return sel


def compare_runs(runs, hours, name='runs_old_vs_new'):
    '''overlay several simulations (label, frame) and Battistel - column 2 has
    TWO pyrite inclusions, so both bands are shaded'''
    bat = pd.read_csv(BAT_PROFILES)
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette('colorblind')
    colors = ['#0077BB', '#EE7733', '#332288', '#88CCEE', '#117733']
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, time in enumerate(sorted(hours)):
        for j, (label, sim) in enumerate(runs):
            sel = hour_slice(sim, [time])
            ax.plot(sel['Z'], sel['O(0) - normalised'],
                    label='%s: t = %d h' % (label, time), linestyle='-',
                    color=colors[i % len(colors)],
                    linewidth=3.0 - j, alpha=0.85)
        sub = bat[np.isclose(bat['Simulation Time (hours)'].round(), time)]
        ax.plot(sub['Z'], sub['O(0) - normalised'],
                label='Battistel et al. 2018: t = %d h' % time,
                linestyle=':', color='0.35', linewidth=3.0, alpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.7, which='both')
    ax.set_axisbelow(True)
    ax.set_xlabel('Z [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('C$_{norm}$ [-]', fontsize=12, fontweight='bold')
    for k, (z0, z1) in enumerate(inclusion_bands(runs[0][1])):
        ax.axvspan(z0, z1, color='gray', alpha=0.2,
                   label='Pyrite inclusion' if k == 1 else '')
    ax.set_xlim(2, 13)
    leg = ax.legend(loc='center left', bbox_to_anchor=(0.60, 0.875),
                    frameon=True, fancybox=True, shadow=True, fontsize=10)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    save(fig, name)


def profiles(sim, hours, comb_csv, suffix):
    '''notebook cells 1 and 2: selected O2 profiles vs Battistel et al.'''
    sel = hour_slice(sim, hours)

    bat = pd.read_csv(BAT_PROFILES)
    for col in set(sel.columns) - set(bat.columns):
        bat[col] = np.nan
    comb = pd.concat([sel, bat], ignore_index=True)
    comb.to_csv(comb_csv, index=False)
    n = {int(h): int((sel['Simulation Time (hours)'].round() == h).sum())
         for h in hours}
    print('\n[2] profiles at t = %s h vs Battistel -> %s '
          '(%s simulation points, %d Battistel points)'
          % (hours, comb_csv, n, len(bat)))
    bands = inclusion_bands(sim)
    print('    pyrite inclusions (L == 2): %s'
          % ', '.join('%.2f-%.2f cm' % b for b in bands))

    # -- 2a: publication style (cell 2) -----------------------------------
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette('colorblind')
    colors = ['#0077BB', '#EE7733', '#332288', '#88CCEE', '#117733']
    styles = {'My_sim': '-', 'Bat_sim': ':'}     # as in the notebook
    labels = {'My_sim': 'This study, simulation:',
              'Bat_sim': 'Battistel et al. 2018, simulation:'}
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, time in enumerate(sorted(hours)):
        for origin in sorted(comb['Origin'].unique()):
            sub = comb[(comb['Simulation Time (hours)'].round() == time) &
                       (comb['Origin'] == origin)]
            if len(sub) > 1:
                ax.plot(sub['Z'], sub['O(0) - normalised'],
                        label='%s t = %d h' % (labels[origin], time),
                        linestyle=styles[origin],
                        color=colors[i % len(colors)],
                        linewidth=2.5, alpha=0.8)
    ax.grid(True, linestyle=':', alpha=0.7, which='both')
    ax.set_axisbelow(True)
    ax.set_xlabel('Z [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('C$_{norm}$ [-]', fontsize=12, fontweight='bold')
    for k, (z0, z1) in enumerate(bands):
        ax.axvspan(z0, z1, color='gray', alpha=0.2,
                   label='Pyrite inclusion' if k == 1 else '')
    ax.set_xlim(2, 13)
    leg = ax.legend(loc='center left', bbox_to_anchor=(0.675, 0.875),
                    frameon=True, fancybox=True, shadow=True, fontsize=10)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    leg.get_title().set_fontsize(10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    plt.tight_layout()
    save(fig, 'profiles_vs_battistel_pub' + suffix)

    # -- 2b: simple marker style (cell 1) ---------------------------------
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, time in enumerate(sorted(hours)):
        for j, origin in enumerate(sorted(comb['Origin'].unique())):
            sub = comb[(comb['Simulation Time (hours)'].round() == time) &
                       (comb['Origin'] == origin)]
            ax.plot(sub['Z'], sub['O(0) - normalised'],
                    label=f't = {time} h, {origin}',
                    marker=['o', 'x'][i % 2], linestyle='',
                    color=['blue', 'red'][j % 2])
    for z0, z1 in bands:
        ax.axvspan(z0, z1, color='gray', alpha=0.3)
    ax.set_xlabel('Z (Distance)')
    ax.set_ylabel('C_norm [-]')
    ax.set_title('Column 2')
    ax.legend(title='Time and Origin', loc='best', fontsize='small')
    ax.grid(True)
    ax.set_xlim(2, 13)
    save(fig, 'profiles_vs_battistel' + suffix)


def main(sim_csv, hours, suffix, compare_csv):
    comb_csv = 'comb%s.csv' % suffix
    int_csv = 'integrated_data%s.csv' % suffix
    if not os.path.exists(sim_csv):
        sys.exit('"%s" not found - run "uv run python ../07-column_conservative/tse2csv.py" first'
                 % sim_csv)
    sim = pd.read_csv(sim_csv)
    sim['Simulation Time (hours)'] = sim['Simulation Time'] / 3600
    print('simulation: %s\n            %d rows, %d times, %.1f h, %d nodes'
          % (sim_csv, len(sim), sim['Simulation Time'].nunique(),
             sim['Simulation Time'].max() / 3600, sim['Z'].nunique()))
    integrated_data(sim, int_csv)
    profiles(sim, hours, comb_csv, suffix)

    # --------------------------------------------------------------
    # 3) optional overlay of a second run (--compare) - not from the notebook
    # --------------------------------------------------------------
    if compare_csv:
        other = pd.read_csv(compare_csv)
        other['Simulation Time (hours)'] = other['Simulation Time'] / 3600
        stem = lambda f: os.path.splitext(os.path.basename(f))[0]
        pretty = {'simulation_output_fixed': 'this study, corrected config',
                  'simulation_output': 'this study, original config'}
        label = lambda f: pretty.get(stem(f), stem(f))
        print('\n[3] overlay of %s (thick) and %s (thin) + Battistel'
              % (sim_csv, compare_csv))
        compare_runs([(label(sim_csv), sim), (label(compare_csv), other)],
                     hours, 'runs_old_vs_new' + suffix)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hours', type=int, nargs='+', default=[4, 24],
                    help='profile times compared with Battistel et al. '
                         '(default: 4 24, as in the notebook)')
    ap.add_argument('--sim', default=SIM_CSV,
                    help='simulation table to plot (default %s)' % SIM_CSV)
    ap.add_argument('--suffix', default='',
                    help='appended to every output name (e.g. "_fixed")')
    ap.add_argument('--compare', default=None, metavar='CSV',
                    help='second simulation table for an overlay figure')
    a = ap.parse_args()
    main(a.sim, a.hours, a.suffix, a.compare)
