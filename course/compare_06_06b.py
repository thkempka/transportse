import joblib, numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = joblib.load('06-groundwater_heat_tracer_phreeqc/config-00010.p')      # phreeqpy
B = joblib.load('06b-groundwater_heat_tracer_phreeqcrm/config-00010.p')  # PhreeqcRM
Ca_a, An_a, T_a = A[12][:, :, 0, 1], A[12][:, :, 0, 2], A[13][:, :, 0]
Ca_b, An_b, T_b = B[12][:, :, 0, 1], B[12][:, :, 0, 3], B[13][:, :, 0]
nx, ny = Ca_a.shape
x = numpy.arange(nx) + 0.5
wy = ny // 2

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
fig.suptitle('Example 06 (phreeqpy) vs 06b (PhreeqcRM) — last time step, t = 4 d',
             fontsize=14, y=0.97)

def diffmap(a, b, title, unit, cmap, axi):
    d = b - a
    v = numpy.abs(d).max() or 1e-30
    im = axi.imshow(d.T, origin='lower', cmap=cmap, vmin=-v, vmax=v,
                    extent=(0, nx, 0, ny))
    axi.set_title('%s\nmax |diff| = %.3e %s' % (title, numpy.abs(d).max(), unit),
                  fontsize=11)
    axi.set_xlabel('x (m)'); axi.set_ylabel('y (m)')
    cb = fig.colorbar(im, ax=axi, shrink=0.85)
    cb.set_label('06b − 06 (%s)' % unit, fontsize=9)
    return d

dCa = diffmap(Ca_a, Ca_b, 'Ca difference', 'mol kgw⁻¹', 'RdBu_r', ax[0, 0])
dAn = diffmap(An_a, An_b, 'Anhydrite difference', 'mol m⁻³', 'PuOr_r', ax[0, 1])

ax[1, 0].plot(x, Ca_a[:, wy], 'k-', lw=2, label='06  phreeqpy')
ax[1, 0].plot(x, Ca_b[:, wy], 'r--', lw=2, label='06b PhreeqcRM')
ax[1, 0].axvline(x[25], color='0.6', lw=0.8, ls=':')
ax[1, 0].set_xlabel('x (m)'); ax[1, 0].set_ylabel('Ca (mol kgw⁻¹)')
ax[1, 0].set_title('Ca along the injection row (y = %.0f m)' % (wy + 0.5), fontsize=11)
ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(x, An_a[:, wy], 'k-', lw=2, label='06  phreeqpy')
ax[1, 1].plot(x, An_b[:, wy], 'r--', lw=2, label='06b PhreeqcRM')
ax[1, 1].axvline(x[25], color='0.6', lw=0.8, ls=':')
ax[1, 1].set_xlabel('x (m)'); ax[1, 1].set_ylabel('anhydrite (mol m⁻³)')
ax[1, 1].set_title('Anhydrite along the injection row (y = %.0f m)' % (wy + 0.5), fontsize=11)
ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=0.3)

for a in ax.flat:
    a.tick_params(labelsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('06b-groundwater_heat_tracer_phreeqcrm/compare_06_vs_06b.png', dpi=130)
print('Ca   : 06 [%.5f, %.5f]  06b [%.5f, %.5f]  max|diff| %.3e  mean %.3e'
      % (Ca_a.min(), Ca_a.max(), Ca_b.min(), Ca_b.max(),
         numpy.abs(dCa).max(), dCa.mean()))
print('anhy : 06 [%.2f, %.2f]  06b [%.2f, %.2f]  max|diff| %.3e  mean %.3e'
      % (An_a.min(), An_a.max(), An_b.min(), An_b.max(),
         numpy.abs(dAn).max(), dAn.mean()))
print('T    : max|diff| %.2f K' % numpy.abs(T_b - T_a).max())
