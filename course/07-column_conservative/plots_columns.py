'''
    plots_columns.py - reproduction of Plots_columns.ipynb for
    config.py (1D column, Battistel et al. 2019).

    Run after the simulation + tse2csv.py:

        uv run python -u ../../tse.py config.py
        uv run python ../../backup/tse2csv.py
        uv run python plots_columns.py

    Outputs
      integrated_data.csv          simulation + lab + Battistel reference series
      figs/o2_breakthrough.png/.pdf        notebook cell 1 style
      figs/o2_breakthrough_pub.pdf/.png    notebook cell 2 style (seaborn)
      figs/column_profiles.png/.pdf        notebook cell 5 style (C_norm vs Z)
'''

import os
import glob
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
LAB_CSV = 'Battistel_1D_Ox_raw.csv'
# previous integrated_data.csv (or a backup of it) - only used to carry over the
# literature reference series ("simulation_battistel", "sim_no_part")
REF_CSV = next((f for f in ('integrated_data.csv',
                            'out_buggy_lf0.6/integrated_data.csv',
                            '.ipynb_checkpoints/integrated_data-checkpoint.csv')
                if os.path.exists(f)), None)
Z_OBS = [7.5, 12.5]          # sensor depths of the laboratory column [cm]
T_MAX = 15.0                 # laboratory data available up to 15 h


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGS, name + '.' + ext),
                    bbox_inches='tight', dpi=200)
    print('  wrote figs/%s.{pdf,png}' % name)
    plt.close(fig)


# =========================================================================
# 1) build integrated_data.csv   (= notebook cell 0)
# =========================================================================
sim = pd.read_csv(SIM_CSV)
sim['Simulation Time (hours)'] = sim['Simulation Time'] / 3600

mask = np.zeros(len(sim), dtype=bool)
for z_obs in Z_OBS:
    mask |= np.isclose(sim['Z'], z_obs / 100, atol=1e-3)
sim = sim.loc[mask].copy()

sim['Z'] = (sim['Z'].round(3) * 100)
sim['Origin'] = 'simulation'
sim_result = sim.rename(columns={'O(0) - normalised': 'O2'})[
    ['Simulation Time (hours)', 'Z', 'O2', 'Origin']]

lab = pd.read_csv(LAB_CSV)
lab['O2'] = pd.to_numeric(lab['O2'], errors='coerce')
lab = lab.dropna(subset=['O2']).rename(columns={'Time': 'Simulation Time (hours)'})

# keep the literature simulation series that were added manually before
ref = pd.DataFrame()
if REF_CSV:
    ref = pd.read_csv(REF_CSV)
    ref = ref.loc[ref['Origin'] != 'simulation',
                  ['Simulation Time (hours)', 'Z', 'O2', 'Origin']]
    print('reference series taken from %s (%i rows, %s)'
          % (REF_CSV, len(ref), sorted(ref['Origin'].unique())))

combined = pd.concat([sim_result,
                      lab[['Simulation Time (hours)', 'Z', 'O2', 'Origin']],
                      ref], ignore_index=True)
combined = combined.sort_values(by=['Simulation Time (hours)', 'Z'],
                                ignore_index=True)
combined.to_csv('integrated_data.csv', index=False)
print('integrated data saved to integrated_data.csv '
      '(%i simulation rows)' % len(sim_result))

df = combined
sim_df = df[df.Origin == 'simulation']
lab_df = df[df.Origin == 'lab']
lab_df = lab_df[lab_df['Simulation Time (hours)'] <= T_MAX]
bat_df = df[df.Origin == 'simulation_battistel']

# =========================================================================
# 2) O2 breakthrough - simple version (notebook cell 1)
# =========================================================================
colors = ['red', 'black']
print('\nFig. 1 - O2 breakthrough (notebook style)')
fig, ax = plt.subplots(figsize=(10, 6))
for z_obs, col in zip(Z_OBS, colors):
    m = sim_df['Z'] == z_obs
    ax.plot(sim_df[m]['Simulation Time (hours)'], sim_df[m]['O2'], color=col,
            linestyle='-', linewidth=2, label=f'Simulation Z = {z_obs}')
    m = lab_df['Z'] == z_obs
    ax.scatter(lab_df[m]['Simulation Time (hours)'], lab_df[m]['O2'], color=col,
               marker='o', s=50, label=f'Laboratory Z = {z_obs}')
    if len(bat_df):
        m = bat_df['Z'] == z_obs
        ax.plot(bat_df[m]['Simulation Time (hours)'], bat_df[m]['O2'], color=col,
                linestyle=':', linewidth=2,
                label=f'No parti Battistel: Z = {z_obs}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('O2 (normalised) [-]')
ax.set_title('Oxygen Concentration vs Time for Different Z and Origin')
ax.legend()
ax.grid(True)
save(fig, 'o2_breakthrough')

# =========================================================================
# 3) O2 breakthrough - publication version (notebook cell 2)
# =========================================================================
plt.style.use('seaborn-v0_8-paper')
sns.set_palette('colorblind')
colors = ['#0077BB', '#EE7733']
markers = ['o', 's']
line_styles = ['-', '--']

print('Fig. 2 - O2 breakthrough (publication style)')
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
for idx, z_obs in enumerate(Z_OBS):
    m = sim_df['Z'] == z_obs
    ax.plot(sim_df[m]['Simulation Time (hours)'], sim_df[m]['O2'],
            color=colors[idx], linestyle=line_styles[0], linewidth=3.5,
            alpha=0.8, label=f'This study, simulation: Z = {z_obs} cm')
for idx, z_obs in enumerate(Z_OBS):
    m = lab_df['Z'] == z_obs
    ax.scatter(lab_df[m]['Simulation Time (hours)'], lab_df[m]['O2'],
               color=colors[idx], marker=markers[idx], s=100, alpha=0.8,
               edgecolor='black', linewidth=1,
               label=f'Battistel et al. 2018, experiment: Z = {z_obs} cm')
if len(bat_df):
    for idx, z_obs in enumerate(Z_OBS):
        m = bat_df['Z'] == z_obs
        ax.plot(bat_df[m]['Simulation Time (hours)'], bat_df[m]['O2'],
                color=colors[idx], linestyle=line_styles[1], linewidth=3.5,
                alpha=0.8,
                label=f'Battistel et al. 2018, simulation: Z = {z_obs} cm')
ax.grid(True, linestyle='--', alpha=0.7, which='both', linewidth=1.2)
ax.set_axisbelow(True)
ax.set_xlabel('Time [h]', fontsize=20, fontweight='bold')
ax.set_ylabel('c$_{norm}$ [-]', fontsize=20, fontweight='bold')
leg = ax.legend(bbox_to_anchor=(0.98, 0.02), loc='lower right',
                borderaxespad=0., frameon=True, fancybox=True, shadow=True,
                fontsize=14)
leg.get_frame().set_alpha(0.9)
save(fig, 'o2_breakthrough_pub')

# =========================================================================
# 4) normalised concentration profiles vs distance (notebook cell 5)
# =========================================================================
plt.style.use('default')
sim = pd.read_csv(SIM_CSV)
sim['Simulation Time (hours)'] = sim['Simulation Time'] / 3600

# one profile per whole simulated hour (nearest time step)
sim['Nearest Hour'] = sim['Simulation Time (hours)'].round()
sim['Difference'] = np.abs(sim['Simulation Time (hours)'] - sim['Nearest Hour'])
min_diff = sim.groupby('Nearest Hour')['Difference'].transform('min')
prof = sim[sim['Difference'] == min_diff].copy()
prof = prof[(prof['Z'] * 100 >= 3.5) & (prof['Z'] * 100 <= 12.5)]

markers = ['o', 's', '^', 'D', 'v', 'p', 'P', '*', 'h', '+', 'x']
times = sorted(prof['Simulation Time (hours)'].unique())

print('Fig. 3 - normalised profiles (notebook style, %d time slices)' % len(times))
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.viridis
for i, time in enumerate(times):
    t_df = prof[prof['Simulation Time (hours)'] == time]
    ax.plot(t_df['Z'] * 100, t_df['O(0) - normalised'],
            label=f't = {time:.0f} h', marker=markers[i % len(markers)],
            linestyle='', color=cmap(i / max(1, len(times) - 1)))

# pyrite inclusion (L == 2)
inc = prof[prof['L'] == 2]['Z'] * 100
if len(inc):
    ax.axvspan(inc.min(), inc.max(), color='gray', alpha=0.3,
               label='pyrite inclusion (L = 2)')

ax.set_xlabel('Z (Distance)')
ax.set_ylabel('C_norm [-]')
ax.set_title('Column 1')
ax.legend(title='Time', loc='best', fontsize='small', ncol=2)
ax.grid(True)
ax.set_xlim(2, 13)
save(fig, 'column_profiles')
