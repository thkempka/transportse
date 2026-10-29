'''
    Compare example 06 (phreeqpy) and 06b (PhreeqcRM): difference maps and
    profiles of Ca, anhydrite and pH for the last time step.

    Usage (from course/):  python compare_06_06b.py
    Requires the last snapshots of both runs (config-00010.p).
'''

import joblib
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = joblib.load('06-groundwater_heat_tracer_phreeqc/config-00010.p')      # phreeqpy
B = joblib.load('06b-groundwater_heat_tracer_phreeqcrm/config-00010.p')  # PhreeqcRM

# species layout differs: 06 = tracer, Ca, anhydrite, pH
#                          06b = tracer, Ca, S(6), anhydrite, pH
fields = [('Ca', A[12][:, :, 0, 1], B[12][:, :, 0, 1], 'mol kgw⁻¹', 'RdBu_r'),
          ('Anhydrite', A[12][:, :, 0, 2], B[12][:, :, 0, 3], 'mol m⁻³', 'PuOr_r'),
          ('pH', A[12][:, :, 0, 3], B[12][:, :, 0, 4], '-', 'BrBG')]
T_a, T_b = A[13][:, :, 0], B[13][:, :, 0]

nx, ny = T_a.shape
x = numpy.arange(nx) + 0.5
wy = ny // 2

fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.5))
fig.suptitle('Example 06 (phreeqpy) vs 06b (PhreeqcRM) — last time step, t = 4 d',
             fontsize=15, y=0.975)

for col, (name, a, b, unit, cmap) in enumerate(fields):
    d = b - a
    v = numpy.abs(d).max() or 1e-30
    im = ax[0, col].imshow(d.T, origin='lower', cmap=cmap, vmin=-v, vmax=v,
                           extent=(0, nx, 0, ny))
    ax[0, col].set_title('%s difference\nmax |diff| = %.3e %s'
                         % (name, numpy.abs(d).max(), unit), fontsize=11)
    ax[0, col].set_xlabel('x (m)')
    ax[0, col].set_ylabel('y (m)')
    fig.colorbar(im, ax=ax[0, col], shrink=0.85).set_label(
        '06b − 06 (%s)' % unit, fontsize=9)

    ax[1, col].plot(x, a[:, wy], 'k-', lw=2, label='06  phreeqpy')
    ax[1, col].plot(x, b[:, wy], 'r--', lw=2, label='06b PhreeqcRM')
    ax[1, col].axvline(x[25], color='0.6', lw=0.8, ls=':')
    ax[1, col].set_xlabel('x (m)')
    ax[1, col].set_ylabel('%s (%s)' % (name, unit))
    ax[1, col].set_title('%s along the injection row (y = %.0f m)'
                         % (name, wy + 0.5), fontsize=11)
    ax[1, col].legend(fontsize=9)
    ax[1, col].grid(alpha=0.3)

for a in ax.flat:
    a.tick_params(labelsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('06b-groundwater_heat_tracer_phreeqcrm/compare_06_vs_06b.png', dpi=130)

for name, a, b, unit, _ in fields:
    print('%-9s 06 [%9.5f, %9.5f]  06b [%9.5f, %9.5f]  max|diff| %.3e  mean %+.3e'
          % (name, a.min(), a.max(), b.min(), b.max(),
             numpy.abs(b - a).max(), (b - a).mean()))
print('T         max|diff| %.2f K' % numpy.abs(T_b - T_a).max())
