'''
    SOR/CG pressure solver equivalence test.

    Runs the same 3-D density-driven flow problem (slope, Dirichlet and
    Neumann boundaries, injection well) with the SOR solver on the
    committed reference implementation, the current SOR implementation
    and the current CG implementation, and compares the resulting
    pressure fields.

    Usage:
        uv run python tests/test_sor_cg_equivalence.py [nx ny nz]

    Exits with status 0 on success.
'''
import os
import shutil
import subprocess
import sys
import tempfile

import joblib
import numpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_TEMPLATE = '''
"""
    Auto-generated SOR/CG equivalence test configuration
"""
import numpy
from numba import njit

parallel_processes = 1
parallel_chemistry = 1
numba_fastmath = False
numba_tl = 'omp'
PSOLVER = "{psolver}"

lx, ly, lz = 101.0, 51.0, 20.0
nx, ny, nz = {nx}, {ny}, {nz}
dx, dy, dz = numpy.ones(nx), numpy.ones(ny), numpy.full(nz, 2.0)

g = 9.80665
beta = 10

t = 3000.0
dt = 10.0
dt_max = 1000.0

FLUIDFLOW = True
SPECTRANS = True
HEATTRANS = False
FLUIDPROP = True
CHEMREACT = False
DYNAMICBC = False
MASSBALAN = False
DTCHEM = 1
C_IMBLFLO = True
T_IMBLFLO = False
SIA = False
SIA_maxit = 5
SIA_conv = 1e-4
SMOLAR_C = False
SMOLAR_T = False

IONS, MINERALS = 1, 1
SPECIES = IONS + MINERALS
SPTRANS = numpy.array([0])

phi = numpy.full((nx, ny, nz), 0.2)
Kx = numpy.full((nx, ny, nz), 1.0e-13)
Ky = numpy.full((nx, ny, nz), 1.0e-13)
Kz = numpy.full((nx, ny, nz), 1.0e-13)
c_r = numpy.full((nx, ny, nz), 1.0e-10)
cp_r = numpy.full((nx, ny, nz), 1000.0)
l_r = numpy.full((nx, ny, nz), 2.5)
rho_r = numpy.full((nx, ny, nz), 2600.0)
lf_l = numpy.full((nx, ny, nz), 0.5)
lt_l = numpy.zeros((nx, ny, nz))
lf_t = numpy.full((nx, ny, nz), 0.5)
lt_t = numpy.zeros((nx, ny, nz))
L = numpy.zeros((nx, ny, nz))
R = numpy.ones((nx, ny, nz))
mass = numpy.zeros((3))

D_fx = numpy.full((nx, ny, nz, SPECIES), 1.0e-9)
D_fy = D_fx.copy()
D_fz = D_fx.copy()
c_f = numpy.full((nx, ny, nz), 4.5e-10)
cp_f = numpy.full((nx, ny, nz), 4180.0)
l_f = numpy.full((nx, ny, nz), 0.6)
rho_f = numpy.full((nx, ny, nz), 1000.0)
nue = numpy.full((nx, ny, nz), 1.0e-3)

P = numpy.full((nx, ny, nz), 1.0e6)
C = numpy.zeros((nx, ny, nz, SPECIES))
T = numpy.full((nx, ny, nz), 20.0)

BP = numpy.zeros((nx, ny, nz))
BC = numpy.zeros((nx, ny, nz))
BT = numpy.zeros((nx, ny, nz))
BP[0, :, :] = 1
BP[-1, :, :] = 1

WP = numpy.zeros((nx, ny, nz))
WT = numpy.zeros((nx, ny, nz))
WC = numpy.zeros((nx, ny, nz, SPECIES))
wx, wy = nx // 2, ny // 2
WP[wx, wy, :] = 50 / 3600 / dx[wx] / dy[wy] / dz[0] * rho_f[wx, wy, :]
BC[wx, wy, :] = 1          # fixed injection concentration at the well

C[:3, :, :, 0] = 1.0       # initial tracer band
C[wx, wy, :, 0] = 1.0      # injection concentration at the well

qx = numpy.zeros((nx, ny, nz))
qy = numpy.zeros((nx, ny, nz))
qz = numpy.zeros((nx, ny, nz))

PLOTSAVE = True
PLOTTIME = numpy.array([1000.0, 2000.0, 3000.0])
PRINTOUT = 1000

CONV_P = 1e-8
CONV_C = 1e-8
CONV_T = 1e-8
OMEGA_C = 1
OMEGA_T = 1
SOR_P = 1.9
SOR_C = 1.0
SOR_T = 1.0
MAXIT_P = 50000


@njit
def fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi,
                     Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r):
    rho_f = 1000.0 + 30.0 * C[0]
    nue = 1.0e-3
    return C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, \\
        Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r


@njit
def chemical_reactions(C, P, T, rho_f, phi, Kx, Ky, Kz, dt_chem, i, j, k):
    return C, P, T, rho_f, phi, Kx, Ky, Kz
'''


def run_case(name, psolver, nx, ny, nz, tse):
    """Run tse.py with the given solver, return the final pressure field."""
    rundir = os.path.join(tempfile.gettempdir(), 'tse_sor_cg_' + name)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    with open(os.path.join(rundir, 'config.py'), 'w') as f:
        f.write(CONFIG_TEMPLATE.format(psolver=psolver, nx=nx, ny=ny, nz=nz))
    subprocess.run(
        [sys.executable, '-u', tse, 'config.py'],
        cwd=rundir, check=True, timeout=3600,
        stdout=subprocess.DEVNULL)
    outfile = os.path.join(rundir, 'config-00002.p')
    data = joblib.load(outfile)
    shutil.rmtree(rundir, ignore_errors=True)
    return data[11], data[12]  # pressure and concentration fields


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    # reference implementation from the master branch
    reference = os.path.join(tempfile.gettempdir(), 'tse_master_ref.py')
    with open(reference, 'w') as f:
        f.write(subprocess.run(
            ['git', '-C', REPO, 'show', 'master:tse.py'],
            capture_output=True, text=True, check=True).stdout)

    print('running master SOR (reference) ...')
    P_ref, C_ref = run_case('ref', 'SOR', nx, ny, nz, reference)
    print('running current SOR ...')
    P_sor, C_sor = run_case('sor', 'SOR', nx, ny, nz,
                            os.path.join(REPO, 'tse.py'))
    print('running current CG ...')
    P_cg, C_cg = run_case('cg', 'CG', nx, ny, nz, os.path.join(REPO, 'tse.py'))

    scale = numpy.max(numpy.abs(P_ref))
    d_sor = numpy.max(numpy.abs(P_ref - P_sor)) / scale
    d_cg = numpy.max(numpy.abs(P_ref - P_cg)) / scale
    d_eq = numpy.max(numpy.abs(P_sor - P_cg)) / scale
    print('master SOR vs current SOR: max relative difference %.3e' % d_sor)
    print('master SOR vs current CG : max relative difference %.3e' % d_cg)
    print('current SOR vs current CG: max relative difference %.3e' % d_eq)

    # master deviates at the pressure-Dirichlet cells because it still
    # contains the uncorrected flow imbalance there (concentration
    # overshoot); the current implementation dissipates it, so master is
    # informational only. The strict checks are the SOR/CG equivalence
    # (identical linear system and trajectory) and the concentration
    # bound at the fixed-concentration source with C_IMBLFLO active
    tol = 1e-6
    assert d_eq < tol, 'SOR/CG equivalence: %.3e >= %.1e' % (d_eq, tol)

    # with the fixed-concentration source and C_IMBLFLO active, the
    # concentration must not exceed the injection concentration
    cmax = max(C_sor.max(), C_cg.max())
    print('max concentration: %.6f (injection: 1.0)' % cmax)
    assert cmax <= 1.0 + 1e-6, 'concentration overshoot: %.6f > 1.0' % cmax
    print('PASS')


if __name__ == '__main__':
    main()
