'''
    plots_columns.py - plotting workflow for the 1D column scenario 1
    (config.py), non-interactive reproduction of
    Plots_columns.ipynb and Plot_Fe_S.ipynb.

    pipeline
        uv run python -u ../../tse.py    config.py
        uv run python    ../../backup/tse2csv.py            # *.p -> simulation_output.csv
        uv run python    plots_columns.py         # -> integrated_data.csv, comb.csv, figs/

    figures (all written to figs/, nothing in the folder is overwritten)
        1 profiles_vs_battistel_pub  notebook cell 3 (Column1_sim_257_z003.pdf)
        2 profiles_vs_battistel      notebook cell 4 (Column1_sim_257_z003.png)
        3 fe_s                       Plot_Fe_S.ipynb cell 0
        4 fe_s_refined               Plot_Fe_S.ipynb cell 1

    options
        --sim CSV       simulation table to plot (default simulation_output.csv)
        --suffix STR    appended to every output name, so a second run (e.g. a second run) creates its own files:
                            --sim simulation_output_2.csv --suffix _2
                        -> comb_fixed.csv, figs/profiles_vs_battistel_fixed.png ...
        --compare CSV   additionally draw this simulation together with --sim
                        and Battistel -> figs/runs_old_vs_new<suffix>.*

    example (plot two runs next to each other)
        uv run python plots_columns.py                       # original names
        uv run python plots_columns.py --sim simulation_output_2.csv \
                       --suffix _2 --compare simulation_output.csv
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
OBS_CSV = 'Battistel_1D_Ox_raw.csv'      # lab O2 data
BAT_PROFILES = 'Bat_fig5.csv'            # Battistel et al. O2 profiles
BAT_FE_S = 'Batt_a.csv'                  # Battistel et al. Fe/S breakthrough
Z_OBS = [7.5, 12.5]                      # sensor positions [cm]
# backup of the previous integrated_data.csv, used only to keep the literature
# reference series ("lab", "sim_no_part", ...)
REF_CSV = next((f for f in ('out_2025/integrated_data.csv',
                            'integrated_data.csv') if os.path.exists(f)), None)

# pyrite inclusion (L == 2) is shaded using the lithology column of the CSV
FE_S_Z = 12.5                            # sampling depth for the Fe/S plot [cm]


def save(fig, name, note=''):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, '%s.%s' % (name, ext)),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  figs/%s.{pdf,png} %s' % (name, note))


def hour_slice(sim, hours):
    '''simulation profiles at the requested hours, Z in cm, as in the notebook'''
    sim_h = sim.copy()
    sim_h['Z'] = sim_h['Z'] * 100
    # keep exactly one output per requested hour - the time step closest to it
    sim_h['Nearest Hour'] = sim_h['Simulation Time (hours)'].round()
    sim_h['dt_off'] = (sim_h['Simulation Time (hours)']
                       - sim_h['Nearest Hour']).abs()
    times = sim_h[['Nearest Hour', 'Simulation Time (hours)',
                   'dt_off']].drop_duplicates('Simulation Time (hours)')
    times = times[times['Nearest Hour'].isin(hours)]
    chosen = times.loc[times.groupby('Nearest Hour')['dt_off'].idxmin(),
                       'Simulation Time (hours)'].to_numpy()
    keep = np.abs(sim_h['Simulation Time (hours)'].to_numpy()[:, None]
                  - chosen[None, :]).min(axis=1) <= 1e-9
    sel = sim_h[keep].copy()
    sel = sel[sel['Z'] <= 12.5]
    sel['Origin'] = 'My_sim'
    return sel


def compare_runs(runs, hours, name='runs_old_vs_new'):
    '''overlay several simulations (label, frame) and Battistel on one figure'''
    bat = pd.read_csv(BAT_PROFILES)
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette('colorblind')
    colors = ['#0077BB', '#EE7733', '#332288', '#88CCEE', '#117733']
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, time in enumerate(sorted(hours)):
        for j, (label, sim) in enumerate(runs):
            sel = hour_slice(sim, [time])
            ax.plot(sel['Z'], sel['O(0) - normalised'], label='%s: t = %d h'
                    % (label, time), linestyle='-',
                    color=colors[i % len(colors)],
                    linewidth=3.0 - j, alpha=0.85)
        sub = bat[np.isclose(bat['Simulation Time (hours)'].round(), time)]
        ax.plot(sub['Z'], sub['O(0) - normalised'],
                label='Battistel et al. 2018: t = %d h' % time,
                linestyle=':', color='0.35', linewidth=3.0, alpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.set_axisbelow(True)
    ax.set_xlabel('Z [cm]', fontsize=20, fontweight='bold')
    ax.set_ylabel('c(O$_2$)$_{norm}$ [-]', fontsize=20, fontweight='bold')
    inc = runs[0][1]
    inc = inc[inc['L'] == 2]['Z'] * 100
    if len(inc):
        ax.axvspan(inc.min(), inc.max(), color='gray', alpha=0.2,
                   label='Pyrite inclusion')
    ax.set_xlim(2, 13)
    leg = ax.legend(loc='center left', bbox_to_anchor=(0.45, 0.85),
                    frameon=True, fancybox=True, shadow=True, fontsize=14)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    save(fig, name, '(overlay of the runs passed with '
         '--sim/--compare)')


def main(sim_csv, hours, suffix, compare_csv):
    comb_csv = 'comb%s.csv' % suffix
    int_csv = 'integrated_data%s.csv' % suffix
    if not os.path.exists(sim_csv):
        sys.exit('"%s" not found - run "uv run python ../../backup/tse2csv.py" first'
                 % sim_csv)
    sim = pd.read_csv(sim_csv)
    sim['Simulation Time (hours)'] = sim['Simulation Time'] / 3600
    print('simulation: %s\n            %d rows, %d times, %.1f h, %d nodes'
          % (sim_csv, len(sim), sim['Simulation Time'].nunique(),
             sim['Simulation Time'].max() / 3600, sim['Z'].nunique()))

    # =====================================================================
    # 1) integrated_data.csv - O2 at the sensor depths + lab data (cell 0)
    # =====================================================================
    mask = np.zeros(len(sim), dtype=bool)
    for z in Z_OBS:
        mask |= np.isclose(sim['Z'], z / 100, atol=1e-3)
    sel = sim.loc[mask].copy()
    sel['Z'] = (sel['Z'].round(3) * 100)
    sel['Origin'] = 'simulation'
    sim_series = sel.rename(columns={'O(0) - normalised': 'O2'})[
        ['Simulation Time (hours)', 'Z', 'O2', 'Origin']]

    lab = pd.read_csv(OBS_CSV)
    lab['O2'] = pd.to_numeric(lab['O2'], errors='coerce')
    lab = lab.dropna(subset=['O2']).rename(
        columns={'Time': 'Simulation Time (hours)'})

    ref = pd.DataFrame()
    if REF_CSV:
        ref = pd.read_csv(REF_CSV)
        ref = ref.loc[ref['Origin'] != 'simulation',
                      ['Simulation Time (hours)', 'Z', 'O2', 'Origin']]
        print('reference series kept from %s: %s'
              % (REF_CSV, ref['Origin'].value_counts().to_dict()))

    combined = pd.concat([sim_series,
                          lab[['Simulation Time (hours)', 'Z', 'O2', 'Origin']],
                          ref], ignore_index=True)
    combined = combined.sort_values(['Simulation Time (hours)', 'Z'],
                                   ignore_index=True)
    combined.to_csv(int_csv, index=False)
    print('%s written (%d simulation rows)' % (int_csv, len(sim_series)))

    # =====================================================================
    # 2) selected profiles vs Battistel - notebook cells 3 and 4
    # =====================================================================
    sel = hour_slice(sim, hours)

    bat = pd.read_csv(BAT_PROFILES)
    for col in set(sel.columns) - set(bat.columns):
        bat[col] = np.nan
    comb = pd.concat([sel, bat], ignore_index=True)
    comb.to_csv(comb_csv, index=False)
    n_sim = {int(h): int((sel['Simulation Time (hours)'].round() == h).sum())
             for h in hours}
    print('\n[2a/2b] profiles at t = %s h vs Battistel (cells 3, 4) -> '
          '%s (%s sim points)' % (hours, comb_csv, n_sim))

    # -- 2a: publication style (cell 3) -----------------------------------
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette('colorblind')
    colors = ['#0077BB', '#EE7733', '#332288', '#88CCEE', '#117733']
    styles = {'My_sim': '-', 'Bat_sim': '--'}
    labels = {'My_sim': 'This study, simulation:',
              'Bat_sim': 'Battistel et al. 2018, simulation:'}
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, time in enumerate(hours):
        for origin in sorted(comb['Origin'].unique()):
            sub = comb[(comb['Simulation Time (hours)'].round() == time) &
                       (comb['Origin'] == origin)]
            if len(sub) > 1:
                ax.plot(sub['Z'], sub['O(0) - normalised'],
                        label='%s t = %d h' % (labels[origin], time),
                        linestyle=styles[origin],
                        color=colors[i % len(colors)],
                        linewidth=3.5, alpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.set_axisbelow(True)
    ax.set_xlabel('Z [cm]', fontsize=20, fontweight='bold')
    ax.set_ylabel('c(O$_2$)$_{norm}$ [-]', fontsize=20, fontweight='bold')
    inc = sim[(sim['L'] == 2)]['Z'] * 100
    if len(inc):
        ax.axvspan(inc.min(), inc.max(), color='gray', alpha=0.2,
                   label='Pyrite inclusion')
    ax.set_xlim(2, 13)
    leg = ax.legend(loc='center left', bbox_to_anchor=(0.47, 0.85),
                    frameon=True, fancybox=True, shadow=True, fontsize=16)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    save(fig, 'profiles_vs_battistel_pub' + suffix)

    # -- 2b: simple marker style (cell 4) ---------------------------------
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, time in enumerate(hours):
        for j, origin in enumerate(sorted(comb['Origin'].unique())):
            sub = comb[(comb['Simulation Time (hours)'].round() == time) &
                       (comb['Origin'] == origin)]
            ax.plot(sub['Z'], sub['O(0) - normalised'],
                    label=f't = {time} h, {origin}',
                    marker=['o', 'x'][i % 2], linestyle='',
                    color=['blue', 'red'][j % 2])
    inc = sim[(sim['L'] == 2)]['Z'] * 100
    if len(inc):
        ax.axvspan(inc.min(), inc.max(), color='gray', alpha=0.3)
    ax.set_xlabel('Z (Distance)')
    ax.set_ylabel('C_norm [-]')
    ax.set_title('Column 1')
    ax.legend(title='Time and Origin', loc='best', fontsize='small')
    ax.grid(True)
    ax.set_xlim(2, 13)
    save(fig, 'profiles_vs_battistel' + suffix)

    # =====================================================================
    # 3) Fe and S at the outlet - Plot_Fe_S.ipynb cells 0 and 1
    # =====================================================================
    fe_s = sim[np.isclose(sim['Z'] * 100, FE_S_Z, atol=1e-6)].copy()
    batt = pd.read_csv(BAT_FE_S)
    print('[3] Fe/S at Z = %.1f cm vs Battistel (%d sim samples, '
          '%d Battistel points)' % (FE_S_Z, len(fe_s), len(batt)))

    for name, refined in (('fe_s', False), ('fe_s_refined', True)):
        if not refined:
            plt.style.use('default')
            colors = {'Fe': 'green', 'S': 'blue'}
        else:
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette('colorblind')
            colors = {'Fe': '#0077BB', 'S': '#EE7733'}
        fig, ax = plt.subplots(figsize=(12 if refined else 10,
                                        8 if refined else 6),
                               dpi=300 if refined else 100)
        for el in ('Fe', 'S'):
            col = '%s (mol/kgw)' % el
            if refined:
                ax.plot(fe_s['Simulation Time (hours)'], fe_s[col],
                        color=colors[el], linewidth=3.5, alpha=0.9,
                        label='%s (This study, simulation)' % el)
            else:
                ax.scatter(fe_s['Simulation Time (hours)'], fe_s[col],
                           color=colors[el], alpha=0.7,
                           label='%s (My_sim)' % el)
        for el in batt['Element'].unique():
            sub = batt[batt['Element'] == el]
            if refined:
                ax.plot(sub['Hour'], sub['Concentration'], color=colors[el],
                        linestyle='--', linewidth=3.5, alpha=0.9,
                        label='%s (Battistel et al. 2018, simulation)' % el)
            else:
                ax.plot(sub['Hour'], sub['Concentration'], color=colors[el],
                        linestyle='-', label='%s (Batt_sim)' % el)
        if refined:
            ax.set_xlabel('Time [h]', fontsize=20, fontweight='bold')
            ax.set_ylabel('Concentration [mol/L]', fontsize=20,
                          fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=18)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            leg = ax.legend(bbox_to_anchor=(0.98, 0.02), loc='lower right',
                            fontsize=16, frameon=True, fancybox=True,
                            shadow=True)
            leg.get_frame().set_alpha(0.9)
        else:
            ax.set_title('Concentration vs Time')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Concentration')
            ax.legend()
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        save(fig, name + suffix)

    # =====================================================================
    # 4) optional overlay of a second run (--compare) - not from the notebook
    # =====================================================================
    if compare_csv:
        other = pd.read_csv(compare_csv)
        other['Simulation Time (hours)'] = other['Simulation Time'] / 3600
        stem = lambda f: os.path.splitext(os.path.basename(f))[0]
        pretty = {'simulation_output_fixed': 'this study, corrected config',
                  'simulation_output': 'this study, original config'}
        label = lambda f: pretty.get(stem(f), stem(f))
        print('\n[4] overlay of %s (thick) and %s (thin) + Battistel'
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
