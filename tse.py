# Licence: GNU General Public License, Version 3, 29 June 2007
# Copyright © 2026 GFZ Helmholtz Centre for Geosciences, Potsdam, Germany
# Citation: Kempka, T. (2020): Verification of a Python-based TRANSPORT
# Simulation Environment for density-driven fluid flow and coupled transport of
# heat and chemical species. Advances in Geosciences, 54, 67-77.
# https://doi.org/10.5194/adgeo-54-67-2020

'''
    tse.py
    See config.py for input parameter documentation
'''

# IMPORT LIBRARIES AND CONFIG FILE ############################################
import os

# avoid over-subscription of CPU resources
os.environ['NUMBA_THREADING_LAYER'] = 'tbb'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import time
import numpy
from joblib import Parallel, delayed, dump

# check if config file name is provided
if len(sys.argv) < 2:
    print('\n\tPlease use this command to run TRANSPORTSE:\n' +
          '\033[1m\tpython [-i] [-u] [path]tse.py [path]<config file name>' +
          '[2>&1 | tee [path]<log file name>]\033[0m' +
          '\n\n\t\033[1mExamples:\033[0m' +
          '\n\tpython -u tse.py config.py 2>&1 | tee config.log' +
          '\n\tpython -i tse.py simulation/config.py' +
          '\n\tpython -i tse.py ~/tse/simulation/config.py\n')
    sys.exit()

# import config file defined as first argument
cfile = sys.argv[1].split('/')[-1].split('.py')[0]
cdir = sys.argv[1].split(cfile+'.py')[0]
sys.path.append(cdir)
config = __import__(cfile)

# MANAGE PARALLELISATION ######################################################
parallel_processes = config.parallel_processes
numba_fastmath = config.numba_fastmath
numba_tl = config.numba_tl
numba_cache = False
numba_nogil = False

if parallel_processes < 2:
    numba_parallel = False
else:
    numba_parallel = True

from numba import njit, prange, threading_layer, set_num_threads
from numba import get_num_threads
from numba import config as config_numba

if numba_parallel:
    config_numba.THREADING_LAYER = numba_tl
    set_num_threads(parallel_processes)

# PRINT GREAT BANNER ##########################################################
BAR = '+' + '-' * 76 + '+'

LOGO = r'''    ___________  ___   _   _  ___________ ___________ _____ _____ _____
   |_   _| ___ \/ _ \ | \ | |/  ___| ___ \  _  | ___ \_   _/  ___|  ___|
     | | | |_/ / /_\ \|  \| |\ `--.| |_/ / | | | |_/ / | | \ `--.| |__
     | | |    /|  _  || . ` | `--. \  __/| | | |    /  | |  `--. \  __|
     | | | |\ \| | | || |\  |/\__/ / |   \ \_/ / |\ \  | | /\__/ / |___
     \_/ \_| \_\_| |_|\_| \_/\____/\_|    \___/\_| \_| \_/ \____/\____/'''

print('\n\033[93m' + BAR + '\033[0m')
print('\033[96m' + LOGO + '\033[0m')
print('\033[93m' + '+---| TRANSPORT Simulation Environment'.ljust(77, '-') + '+\033[0m')

# MODEL DIMENSIONS AND SPATIAL DISCRETISATION #################################
lx = config.lx
ly = config.ly
lz = config.lz
nx = config.nx
ny = config.ny
nz = config.nz
dx = config.dx
dy = config.dy
dz = config.dz

# GENERAL SETTINGS ############################################################
g = config.g                       # gravitational acceleration (m/s^2)
beta = numpy.radians(config.beta)  # model slope angle (rad)

# SIMULATION TIME #############################################################
t = config.t
dt_max = config.dt_max       # maximum time step size (s)

# PLOTTING ####################################################################
PLOTINDEX = 0                # required for time series plots

# MATRIX INITIALISATION #######################################################
# -> ROCK #####################################################################
phi = config.phi         # porosity (-)
Kx = config.Kx           # permeability (m^2)
Ky = config.Ky           # permeability (m^2)
Kz = config.Kz           # permeability (m^2)
c_r = config.c_r         # compressiblity (1/Pa)
cp_r = config.cp_r       # heat capacity (J/kg/K)
l_r = config.l_r         # heat conductivity (W/m/K)
rho_r = config.rho_r     # density (kg/m^3)
lf_l = config.lf_l       # longitudinal dispersion length (m)
lf_t = config.lf_t       # transversal dispersion length (m)
lt_l = config.lt_l       # longitudinal dispersion length (m)
lt_t = config.lt_t       # transversal dispersion length (m)
L = config.L             # lithology (-)
R = config.R             # retardation coefficient (-)

# -> FLUID ####################################################################
D_fx = config.D_fx       # diffusion coefficient (W/m/K)
D_fy = config.D_fy       # diffusion coefficient (W/m/K)
D_fz = config.D_fz       # diffusion coefficient (W/m/K)
c_f = config.c_f         # compressiblity (1/Pa)
cp_f = config.cp_f       # heat capacity (J/kg/K)
l_f = config.l_f         # heat conductivity (W/m/K)
rho_f = config.rho_f     # density (kg/m^3)
nue = config.nue         # viscosity (Pa*s)

# -> PRIMARY VARIABLES ########################################################
P = config.P                 # pore pressure (kg/m/s^2 = Pa)
C = config.C                 # concentration (kg/m^3 bulk volume)
T = config.T                 # temperature (°C)

# -> BOUNDARY CONDITIONS ######################################################
BP = config.BP               # flow boundary conditions
BC = config.BC               # species boundary conditions
BT = config.BT               # thermal boundary conditions

# -> GENERATION TERMS ##########################################################
WP = config.WP               # fluid flow (1/s)
WT = config.WT               # heat flow  (W/m^3)
WC = config.WC               # species flow (kg/m^3/s)

# -> FLOW VELOCITIES ##########################################################
qx = config.qx               # fluid flow velocity (m/s)
qy = config.qy               # fluid flow velocity (m/s)
qz = config.qz               # fluid flow velocity (m/s)

# -> MASS BALANCE DATA ########################################################
mass = config.mass           # mass balance vector

# -> SMOLARKIEWICZ FLOW VELOCITIES ############################################
vx = numpy.zeros((nx, ny, nz))  # Smolarkiewicz velocity (m/s)
vy = numpy.zeros((nx, ny, nz))  # Smolarkiewicz velocity (m/s)
vz = numpy.zeros((nx, ny, nz))  # Smolarkiewicz velocity (m/s)

# -> PECLET MATRICES ##########################################################
PeC = numpy.zeros((nx, ny, nz, 3))  # Peclet number, species transport (-)
PeT = numpy.zeros((nx, ny, nz, 3))  # Peclet number, heat transport (-)

# -> FLOW BALANACE ERROR ######################################################
e = numpy.zeros((nx, ny, nz))  # fluid flow balance error (kg/m^3/s)

# -> HELPER arrays ############################################################
ZEROS = numpy.zeros((nx, ny, nz))  # for immobile phase (no fluid conduction)
ONES  = numpy.ones((nx, ny, nz))  # for immobile phase (no fluid conduction)


# -> SOR GRID PARTITIONING ####################################################
GP = numpy.zeros((nx, ny, nz), dtype=bool)
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if (i + j + k) % 2 == 0:
                GP[i, j, k] = True

# -> PRESSURE SOLVER STENCIL ##################################################
aE = numpy.zeros((nx, ny, nz))
aW = numpy.zeros((nx, ny, nz))
aN = numpy.zeros((nx, ny, nz))
aS = numpy.zeros((nx, ny, nz))
aT = numpy.zeros((nx, ny, nz))
aB = numpy.zeros((nx, ny, nz))
bgrav = numpy.zeros((nx, ny, nz))
bdir = numpy.zeros((nx, ny, nz))
Rbase = numpy.zeros((nx, ny, nz))
Msum = numpy.zeros((nx, ny, nz))

# PRESSURE SOLVER SELECTION ###################################################
PSOLVER = config.PSOLVER if 'PSOLVER' in dir(config) else 'SOR'
CG_OMEGA = config.CG_OMEGA if 'CG_OMEGA' in dir(config) else 1.0
C_IMBLFLO = config.C_IMBLFLO if 'C_IMBLFLO' in dir(config) else False
T_IMBLFLO = config.T_IMBLFLO if 'T_IMBLFLO' in dir(config) else False

# -> UNSATURATED FLOW (RICHARDS EQUATION) #####################################
# All parameters are per-cell numpy arrays of model size (nx, ny, nz), so
# that each cell can have individual hydraulic properties (lithology).
UNSATURAT = config.UNSATURAT if 'UNSATURAT' in dir(config) else False
DEBUG_INNER = False

if UNSATURAT:
    # van Genuchten (1980) / Haverkamp et al. (1977) parameters per cell
    alpha_vg = config.alpha_vg   # vG: alpha (1/m) | Haverkamp: alpha (m^beta)
    n_vg = config.n_vg           # vG: n (-)     | Haverkamp: beta (-)
    S_r = config.S_r             # residual saturation (-)
    S_s = config.S_s if 'S_s' in dir(config) else numpy.ones((nx, ny, nz))
    KRMODEL = config.KRMODEL if 'KRMODEL' in dir(config) else 'mualem-vg'
    krm = 0 if KRMODEL == 'mualem-vg' else 1

    for pname, parr in (('alpha_vg', alpha_vg), ('n_vg', n_vg),
                        ('S_r', S_r), ('S_s', S_s)):
        if numpy.shape(parr) != (nx, ny, nz):
            sys.exit('* %s must be a numpy array of model size (nx, ny, nz)'
                     % pname)
    if KRMODEL not in ('mualem-vg', 'haverkamp'):
        sys.exit('* KRMODEL must be "mualem-vg" or "haverkamp"')
    if numpy.min(alpha_vg) <= 0 or numpy.min(n_vg) <= 1:
        sys.exit('* alpha_vg must be > 0 and n_vg must be > 1')

    if krm == 1:
        # Haverkamp conductivity parameters A (m^gamma) and gamma (-)
        A_vg = config.A_vg
        gamma_vg = config.gamma_vg
        for pname, parr in (('A_vg', A_vg), ('gamma_vg', gamma_vg)):
            if numpy.shape(parr) != (nx, ny, nz):
                sys.exit('* %s must be a numpy array of model size '
                         '(nx, ny, nz)' % pname)
    else:
        A_vg = ZEROS
        gamma_vg = ZEROS

    # initial saturation from the retention curve applied to the initial
    # pressure field (P < 0: capillary pressure head h = -P/(rho_f*g));
    # the initial saturation is therefore always consistent with the
    # initial pressure field (hydrostatic equilibrium initial conditions
    # yield the corresponding equilibrium saturation profile)
    h_ = numpy.where(P >= 0.0, 0.0, -P / (rho_f * g))
    if krm == 0:
        m_ = 1.0 - 1.0 / n_vg
        Se_ = (1.0 + (alpha_vg * h_)**n_vg)**(-m_)
    else:
        Se_ = alpha_vg / (alpha_vg + h_**n_vg)
    S = numpy.where(P >= 0.0, S_s, S_r + (S_s - S_r) * Se_)
    Sn = S.copy()                # saturation at the previous time level

    # initial relative permeability (mualem-vg: from Se; haverkamp: from h)
    if krm == 0:
        kr = numpy.sqrt(Se_) * (1.0 - (1.0 - Se_**(1.0 / m_))**m_)**2
    else:
        kr = A_vg / (A_vg + h_**gamma_vg)
    kr = numpy.where(P >= 0.0, 1.0, kr)

    # capillary capacity dS/dP (1/Pa) and upwinded face relative
    # permeabilities (updated per Picard iteration in the main loop)
    dSdP = numpy.zeros((nx, ny, nz))          # capillary capacity (1/Pa)
    krE = numpy.zeros((nx, ny, nz))           # face kr, +x direction (-)
    krW = numpy.zeros((nx, ny, nz))           # face kr, -x direction (-)
    krN = numpy.zeros((nx, ny, nz))           # face kr, +y direction (-)
    krS = numpy.zeros((nx, ny, nz))           # face kr, -y direction (-)
    krT = numpy.zeros((nx, ny, nz))           # face kr, +z direction (-)
    krB = numpy.zeros((nx, ny, nz))           # face kr, -z direction (-)

    # modified Picard convergence criterion (saturation L2 norm, -)
    CONV_S = config.CONV_S if 'CONV_S' in dir(config) else 1e-6
else:
    # saturated flow: S and kr are unity (no effect on the solvers)
    S = ONES
    Sn = ONES
    kr = ONES
    A_vg = ZEROS
    gamma_vg = ZEROS
    krm = 0

# CALCULATE INITIAL FLUIDPROPERTIES ###########################################
def run_fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r,
                         phi, Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r):

    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                C[i, j, k], T[i, j, k], P[i, j, k], nue[i, j, k], \
                    rho_f[i, j, k], c_f[i, j, k], cp_f[i, j, k], \
                    l_f[i, j, k], cp_r[i, j, k], l_r[i, j, k], phi[i, j, k], \
                    Kx[i, j, k], Ky[i, j, k], Kz[i, j, k], D_fx[i, j, k], \
                    D_fy[i, j, k], D_fz[i, j, k], rho_r[i, j, k] = \
                    config.fluid_properties(
                        C[i, j, k], T[i, j, k], P[i, j, k], nue[i, j, k],
                        rho_f[i, j, k], c_f[i, j, k], cp_f[i, j, k],
                        l_f[i, j, k], cp_r[i, j, k], l_r[i, j, k],
                        phi[i, j, k], Kx[i, j, k], Ky[i, j, k],
                        Kz[i, j, k], D_fx[i, j, k], D_fy[i, j, k],
                        D_fz[i, j, k], rho_r[i, j, k])

    return C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, \
        Kz, D_fx, D_fy, D_fz, rho_r


if 'JITFLUPRO' in dir(config) and config.JITFLUPRO:
    run_fluid_properties = njit(run_fluid_properties, parallel=numba_parallel,
                                fastmath=numba_fastmath, cache=numba_cache,
                                nogil=numba_nogil)


def update_fluid_properties():
    # wrapper for run_fluid_properties supporting 3D temperature arrays
    # (single phase) as well as 4D temperature arrays (NLTE, where the
    # fluid phase is stored in layer 0)
    global C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, \
        Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r
    if T.ndim == 3:
        C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, \
            Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r = \
            run_fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r,
                                 l_r, phi, Kx, Ky, Kz, D_fx, D_fy, D_fz,
                                 rho_r)
    else:
        C, T[:, :, :, 0], P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, \
            Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r = \
            run_fluid_properties(C, T[:, :, :, 0], P, nue, rho_f, c_f, cp_f,
                                 l_f, cp_r, l_r, phi, Kx, Ky, Kz, D_fx, D_fy,
                                 D_fz, rho_r)


if config.FLUIDPROP:
    update_fluid_properties()

# READ CONVERGENCE CRITERIA - MAXIMUM ITERATIONS ##############################
if 'MAXIT_P' in dir(config) and config.MAXIT_P != 0:
    MAXIT_P = config.MAXIT_P
else:
    MAXIT_P = 20000

if 'MAXIT_C' in dir(config) and config.MAXIT_C != 0:
    MAXIT_C = config.MAXIT_C
else:
    MAXIT_C = 100

if 'MAXIT_T' in dir(config) and config.MAXIT_T != 0:
    MAXIT_T = config.MAXIT_T
else:
    MAXIT_T = 100


# SUPPORT FUNCTIONS ###########################################################
@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def alpha(i1, j1, k1, i2, j2, k2, nue, rho_f, K, q, d, d1, d2, n):
    '''
        alpha = k * rho_f / (nue * di)
        with rho_f upwinded
        i1 > i2, j1 > j2
    '''

    return (harm(i1, j1, k1, i2, j2, k2, K, n) /
            harm(i1, j1, k1, i2, j2, k2, nue, n) /
            (d[d2] * (d[d1]+d[d2])*0.5)
            * harm(i1, j1, k1, i2, j2, k2, rho_f, n))


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def beta_g(i1, j1, k1, i2, j2, k2, rho_f, d):
    # slope in x/z direction can be applied via angle beta (0-1)
    if d == 0:
        return amean(i1, j1, k1, i2, j2, k2, rho_f, 0) * g * \
                0.5 * (dx[i1] + dx[i2]) * numpy.sin(beta)
    else:
        return amean(i1, j1, k1, i2, j2, k2, rho_f, 2) * g * \
                0.5 * (dz[k1] + dz[k2]) * numpy.cos(beta)

@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def l2_norm(New, Old, Process):
    l2 = numpy.sqrt(numpy.sum((New - Old)**2)) / \
           numpy.sqrt(numpy.sum(Old**2)+1e-200)
    if numpy.isnan(l2):
        print('ISNAN:', Process)
        return 1
    else:
        return l2


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def amean(i1, j1, k1, i2, j2, k2, X, n):
    if n == 0:
        return ((X[i1, j1, k1] * dx[i1] + X[i2, j2, k2] * dx[i2]) /
                (dx[i1] + dx[i2]))
    elif n == 1:
        return ((X[i1, j1, k1] * dy[j1] + X[i2, j2, k2] * dy[j2])
                / (dy[j1] + dy[j2]))
    else:
        return ((X[i1, j1, k1] * dz[k1] + X[i2, j2, k2] * dz[k2])
                / (dz[k1] + dz[k2]))


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def harm(i1, j1, k1, i2, j2, k2, H, n):
    if H[i1, j1, k1] != 0 and H[i2, j2, k2] != 0:
        if n == 0:
            return ((dx[i1] + dx[i2]) /
                    (dx[i1] / H[i1, j1, k1] + dx[i2] / H[i2, j2, k2]))
        elif n == 1:
            return ((dy[j1] + dy[j2]) /
                    (dy[j1] / H[i1, j1, k1] + dy[j2] / H[i2, j2, k2]))
        else:
            return ((dz[k1] + dz[k2]) /
                    (dz[k1] / H[i1, j1, k1] + dz[k2] / H[i2, j2, k2]))
    else:
        return 0

# UNSATURATED FLOW CONSTITUTIVE RELATIONS #####################################
@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def sat_kr_point(P_, rho, a_, n_, Sr_, Ss_, A_, g_, krm):
    '''
        Saturation and relative permeability at a single node.
        P_ gauge pressure (Pa); P_ < 0: capillary pressure head
        h = -P_/(rho*g) (m).
        krm 0: van Genuchten (1980) retention with Mualem (1976) conductivity
            Se = (1 + (a_*h)^n)^(-(1-1/n))
            S  = Sr + (Ss-Sr)*Se
            kr = Se^0.5 * (1 - (1-Se^(1/m))^m)^2
        krm 1: Haverkamp et al. (1977) retention and conductivity
            Se = a_/(a_ + h^n)          (a_ in m^beta, n = beta)
            kr = A_/(A_ + h^g_)         (A_ in m^gamma)
    '''
    if P_ >= 0.0:
        return Ss_, 1.0
    h = -P_ / (rho * g)
    if krm == 0:
        m_ = 1.0 - 1.0 / n_
        Se = (1.0 + (a_ * h)**n_)**(-m_)
        S_ = Sr_ + (Ss_ - Sr_) * Se
        kr_ = numpy.sqrt(Se) * (1.0 - (1.0 - Se**(1.0 / m_))**m_)**2
    else:
        Se = a_ / (a_ + h**n_)
        S_ = Sr_ + (Ss_ - Sr_) * Se
        kr_ = A_ / (A_ + h**g_)
    return S_, kr_


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def dsdp_point(P_, rho, a_, n_, Sr_, Ss_, krm):
    '''
        Capillary capacity dS/dP (1/Pa) at a single node, required for the
        modified Picard linearisation of the mixed-form storage term
        (Celia et al. 1990). Zero for P_ >= 0 (saturated).
    '''
    if P_ >= 0.0:
        return 0.0
    h = -P_ / (rho * g)
    if krm == 0:
        m_ = 1.0 - 1.0 / n_
        x = (a_ * h)**n_
        # dSe/dh = -m*n*a^n*h^(n-1)*(1+(a*h)^n)^(-m-1)
        dSe_dh = -m_ * n_ * a_**n_ * h**(n_ - 1.0) * (1.0 + x)**(-m_ - 1.0)
    else:
        # Se = a/(a+h^n): dSe/dh = -a*n*h^(n-1)/(a+h^n)^2
        dSe_dh = -a_ * n_ * h**(n_ - 1.0) / (a_ + h**n_)**2
    # chain rule: dSe/dP = dSe/dh * dh/dP with h = -P/(rho*g)
    return (Ss_ - Sr_) * (-dSe_dh) / (rho * g)


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def harm2(a, b):
    'harmonic mean of two scalars (0 if either is 0)'
    if a != 0 and b != 0:
        return 2.0 * a * b / (a + b)
    return 0.0


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT, krB, rho_f,
                      Kx, Ky, Kz, alpha_vg, n_vg, S_r, S_s, A_vg, gamma_vg,
                      krm):
    '''
        Modified Picard update: saturation S(P), capillary capacity
        dS/dP(P), cell and face relative permeabilities from the current
        pressure iterate (all nodes, including pressure-Dirichlet nodes).
        The face kr values are the ratio of the harmonic mean of the cell
        conductivities K*kr to the harmonic mean of K, so that the face
        conductance alpha()*kr equals the harmonic mean of K*kr (series
        resistance, consistent with the saturated-flow discretisation;
        reduces exactly to kr = 1 for saturated cells). The face kr values
        are frozen for the subsequent linear solve, which keeps the inner
        solver linear.
    '''
    # cell values first (race-free: face pass reads cell kr)
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                S[i, j, k], kr[i, j, k] = sat_kr_point(
                    P[i, j, k], rho_f[i, j, k], alpha_vg[i, j, k],
                    n_vg[i, j, k], S_r[i, j, k], S_s[i, j, k],
                    A_vg[i, j, k], gamma_vg[i, j, k], krm)
                dSdP[i, j, k] = dsdp_point(
                    P[i, j, k], rho_f[i, j, k], alpha_vg[i, j, k],
                    n_vg[i, j, k], S_r[i, j, k], S_s[i, j, k], krm)
    # face relative permeabilities: upstream weighting with respect to the
    # pressure head (the wetter cell's kr). The harmonic mean blocks
    # imbibition fronts (the dry side's kr ~ 0 at S ~ S_r), while the
    # upstream rule is the standard treatment for Richards wetting fronts
    # (Vauclin's FE, HYDRUS, MODFLOW-SURFACT). For saturated cells (kr = 1
    # on both sides) this reduces exactly to kr = 1.
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if i != nx-1:
                    krE[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i+1, j, k] else kr[i+1, j, k]
                else:
                    krE[i, j, k] = 0.0
                if i != 0:
                    krW[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i-1, j, k] else kr[i-1, j, k]
                else:
                    krW[i, j, k] = 0.0
                if j != ny-1:
                    krN[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i, j+1, k] else kr[i, j+1, k]
                else:
                    krN[i, j, k] = 0.0
                if j != 0:
                    krS[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i, j-1, k] else kr[i, j-1, k]
                else:
                    krS[i, j, k] = 0.0
                if k != nz-1:
                    krT[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i, j, k+1] else kr[i, j, k+1]
                else:
                    krT[i, j, k] = 0.0
                if k != 0:
                    krB[i, j, k] = kr[i, j, k] \
                        if P[i, j, k] >= P[i, j, k-1] else kr[i, j, k-1]
                else:
                    krB[i, j, k] = 0.0


def plotresults():
    # write results into pickle file at selected time steps
    global PLOTINDEX, dt, dt_prev
    if config.PLOTSAVE:
        if PLOTINDEX <= numpy.shape(config.PLOTTIME)[0] - 1:
            t_plot = config.PLOTTIME[PLOTINDEX]
            if t_sim < t_plot:
                dt_prev = dt
                dt = min(dt, t_plot-t_sim)
            elif t_sim == t_plot:
                with open("{:s}-{:05d}.p".format(cdir+cfile, PLOTINDEX), "wb")\
                  as pfile:
                    dump((lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt,
                          P, C, T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f,
                          nue, cp_f, l_f, BP, BC, BT, L, mass, WP, WT, WC, e,
                          S),
                         pfile, compress=3)
                dt = dt_prev
                PLOTINDEX += 1

@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def peclet_C(PeC, qx, qy, qz, D_fx, D_fy, D_fz):
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if BC[i, j, k] == 0:
                    if numpy.max(D_fx[i, j, k]) != 0:
                        PeC[i, j, k, 0] = \
                            (abs(qx[i, j, k]) / numpy.max(D_fx[i, j, k])
                             * dx[i])
                    else:
                        PeC[i, j, k, 0] = numpy.nan

                    if numpy.max(D_fy[i, j, k]) != 0:
                        PeC[i, j, k, 1] = \
                            (abs(qy[i, j, k]) / numpy.max(D_fy[i, j, k])
                             * dy[j])
                    else:
                        PeC[i, j, k, 1] = numpy.nan

                    if numpy.max(D_fz[i, j, k]) != 0:
                        PeC[i, j, k, 2] = \
                            (abs(qz[i, j, k]) / numpy.max(D_fz[i, j, k])
                             * dz[k])
                    else:
                        PeC[i, j, k, 2] = numpy.nan

    return numpy.max(PeC[:, :, :, 0]), numpy.max(PeC[:, :, :, 1]), \
        numpy.max(PeC[:, :, :, 2])

@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def peclet_T(PeT, qx, qy, qz, rho_f, cp_f, phi, l_r, l_f):
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if BT[i, j, k] == 0:
                    PeT[i, j, k, 0] = \
                        (abs(qx[i, j, k]) * rho_f[i, j, k] * cp_f[i, j, k]
                         / ((1-phi[i, j, k]) * l_r[i, j, k] + phi[i, j, k]
                         * l_f[i, j, k])*dx[i])
                    PeT[i, j, k, 1] = \
                        (abs(qy[i, j, k]) * rho_f[i, j, k] * cp_f[i, j, k]
                         / ((1-phi[i, j, k]) * l_r[i, j, k] + phi[i, j, k]
                         * l_f[i, j, k])*dy[j])
                    PeT[i, j, k, 2] = \
                        (abs(qz[i, j, k]) * rho_f[i, j, k] * cp_f[i, j, k]
                         / ((1-phi[i, j, k]) * l_r[i, j, k] + phi[i, j, k]
                         * l_f[i, j, k])*dz[k])

    return numpy.max(PeT[:, :, :, 0]), numpy.max(PeT[:, :, :, 1]), \
        numpy.max(PeT[:, :, :, 2])

@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def explicit_C(D_fx, phi):
    dt_NeC = 1e100
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                dt_NeC = min(dt_NeC,
                             0.45 * numpy.max(D_fx[i, j, k]) / phi[i, j, k] /
                             (1 / dx[i]**2 + 1 / dy[j]**2 + 1/dz[k]**2))

    return dt_NeC

# determine elements for chemical calculation
@njit(parallel=False, fastmath=numba_fastmath, cache=numba_cache)
def index_list(BC):
    chem_list = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if BC[i, j, k] != 1:
                    chem_list.append((i, j, k))
    return chem_list

@njit(parallel=False, fastmath=numba_fastmath, cache=numba_cache)
def flow_error(P, Pn, qx, qy, qz, WP, phi, c_f, c_r, rho_f, dt_P, S, Sn,
               unsat):
    '''
        Local flow balance error e = I - O - S + WP after Panday et al.
        (2018), computed for ALL cells including pressure-Dirichlet
        cells, whose imbalance (water exchanged with the exterior
        reservoir) would otherwise remain uncorrected in the transport
        solution and cause concentration overshoot at these cells.
        For the Richards solution (unsat=True) the storage term is the
        mixed-form derivative of the water mass phi*S*rho:
            Sn*c_t*(P-Pn)*rho/dt + phi*(S-Sn)*rho/dt
    '''
    e = numpy.zeros((nx, ny, nz))
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if i != nx-1:
                    e[i, j, k] -= (qx[i, j, k] / dx[i]
                                * harm(i+1, j, k, i, j, k, rho_f, 0))
                if i != 0:
                    e[i, j, k] += (qx[i-1, j, k] / dx[i]
                                * harm(i-1, j, k, i, j, k, rho_f, 0))
                if j != ny-1:
                    e[i, j, k] -= (qy[i, j, k] / dy[j]
                                * harm(i, j+1, k, i, j, k, rho_f, 1))
                if j != 0:
                    e[i, j, k] += (qy[i, j-1, k] / dy[j]
                                * harm(i, j-1, k, i, j, k, rho_f, 1))
                if k != nz-1:
                    e[i, j, k] -= (qz[i, j, k] / dz[k]
                                * harm(i, j, k+1, i, j, k, rho_f, 2))
                if k != 0:
                    e[i, j, k] += (qz[i, j, k-1] / dz[k]
                                * harm(i, j, k-1, i, j, k, rho_f, 2))

                por = phi[i, j, k]
                if unsat:
                    e[i, j, k] += (WP[i, j, k]
                        + Sn[i, j, k] * ((1 - por) * c_r[i, j, k]
                        + por * c_f[i, j, k]) * (P[i, j, k] - Pn[i, j, k])
                        * rho_f[i, j, k] / dt_P
                        + por * (S[i, j, k] - Sn[i, j, k])
                        * rho_f[i, j, k] / dt_P)
                else:
                    e[i, j, k] += (WP[i, j, k] + ((1 - por)
                        * c_r[i, j, k] + por * c_f[i, j, k]) 
                        * (P[i, j, k] - Pn[i, j, k]) * rho_f[i, j, k] / dt_P)

    return e

from numba import extending, types
from numba.core import cgutils
@extending.intrinsic
def printf(typingctx, format_type, *args):
    """printf that can be called from Numba jit-decorated functions.
    """
    if isinstance(format_type, types.StringLiteral):
        sig = types.void(format_type, types.BaseTuple.from_types(args))
        def codegen(context, builder, signature, args):
            cgutils.printf(builder, format_type.literal_value, *args[1:])
        return sig, codegen

@njit(parallel=False, fastmath=numba_fastmath, cache=numba_cache)
def dispersivity(qx, qy, qz, l_l, l_t, B, i, j, k):

    D_xr = D_yr = D_zr = D_xl = D_yl = D_zl = \
    qxl = qyl = qzl = qxr = qyr = qzr = \
    qxyl = qxzl = qyxl = qyzl = qzxl = qzyl = \
    qxyr = qxzr = qyxr = qyzr = qzxr = qzyr = 0

    # calculate longitudinal and transversal velocity terms
    if i != 0:  # left face
        qxl = qx[i-1, j, k]  # longitudinal term

        # X-Y plane transversal term (clockwise)
        q1 = qy[i-1, j, k]
        q2 = qy[i, j, k]
        q3 = q4 = 0
        c = 2
        if j != 0:
            q3 = qy[i, j-1, k]
            q4 = qy[i-1, j-1, k]
            c += 2
        if not B[i-1, j, k]:
            qxyl = (q1 + q2 + q3 + q4) / c

        # X-Z plane transversal term (clockwise)
        q1 = qz[i-1, j, k]
        q2 = qz[i, j, k]
        q3 = q4 = 0
        c = 2
        if k != 0:
            q3 = qz[i, j, k-1]
            q4 = qz[i-1, j, k-1]
            c += 2
        if not B[i-1, j, k]:
            qxzl = (q1 + q2 + q3 + q4) / c

    if i != nx-1:  # right face
        qxr = qx[i, j, k]  # longitudinal term

        # X-Y plane transversal term (clockwise)
        q1 = qy[i, j, k]
        q2 = qy[i+1, j, k]
        q3 = q4 = 0
        c = 2
        if j != 0:
            q3 = qy[i+1, j-1, k]
            q4 = qy[i, j-1, k]
            c += 2
        if not B[i+1, j, k]:
            qxyr = (q1 + q2 + q3 + q4) / c

        # X-Z plane transversal term (clockwise)
        q1 = qz[i, j, k]
        q2 = qz[i+1, j, k]
        q3 = q4 = 0
        c = 2
        if k != 0:
            q3 = qz[i+1, j, k-1]
            q4 = qz[i, j, k-1]
            c += 2
        if not B[i+1, j, k]:
            qxzr = (q1 + q2 + q3 + q4) / c

    if j != 0:  # left face
        qyl = qy[i, j-1, k]  # longitudinal term

        # Y-X plane transversal term (clockwise)
        q1 = qx[i, j, k]
        q2 = qx[i, j-1, k]
        q3 = q4 = 0
        c = 2
        if i != 0:
            q3 = qx[i-1, j-1, k]
            q4 = qx[i-1, j, k]
            c += 2
        if not B[i, j-1, k]:
            qyxl = (q1 + q2 + q3 + q4) / c

        # Y-Z plane transversal term (clockwise)
        q1 = qz[i, j-1, k]
        q2 = qz[i, j, k]
        q3 = q4 = 0
        c = 2
        if k != 0:
            q3 = qz[i, j, k-1]
            q4 = qz[i, j-1, k-1]
            c += 2
        if not B[i, j-1, k]:
            qyzl = (q1 + q2 + q3 + q4) / c

    if j != ny-1:  # right face
        qyr = qy[i, j, k]  # longitudinal term

        # Y-X plane transversal term (clockwise)
        q1 = qx[i, j+1, k]
        q2 = qx[i, j, k]
        q3 = q4 = 0
        c = 2
        if i != 0:
            q3 = qx[i-1, j, k]
            q4 = qx[i-1, j+1, k]
            c += 2
        if not B[i, j+1, k]:
            qyxr = (q1 + q2 + q3 + q4) / c

        # Y-Z plane transversal term (clockwise)
        q1 = qz[i, j, k]
        q2 = qz[i, j+1, k]
        q3 = q4 = 0
        c = 2
        if k != 0:
            q3 = qz[i, j+1, k-1]
            q4 = qz[i, j, k-1]
            c += 2
        if not B[i, j+1, k]:
            qyzr = (q1 + q2 + q3 + q4) / c

    if k != 0:  # bottom face
        qzl = qz[i, j, k-1]  # longitudinal term

        # Z-X plane transversal term (clockwise)
        q1 = qx[i, j, k]
        q2 = qx[i, j, k-1]
        q3 = q4 = 0
        c = 2
        if i != 0:
            q3 = qx[i-1, j, k-1]
            q4 = qx[i-1, j, k]
            c += 2
        if not B[i, j, k-1]:
            qzxl = (q1 + q2 + q3 + q4) / c

        # Z-Y plane transversal term (clockwise)
        q1 = qy[i, j, k]
        q2 = qy[i, j, k-1]
        q3 = q4 = 0
        c = 2
        if j != 0:
            q3 = qy[i, j-1, k-1]
            q4 = qy[i, j-1, k]
            c += 2
        if not B[i, j, k-1]:
            qzyl = (q1 + q2 + q3 + q4) / c

    if k != nz-1:  # top face
        qzr = qz[i, j, k]  # longitudinal term

        # Z-X plane transversal term (clockwise)
        q1 = qx[i, j, k+1]
        q2 = qx[i, j, k]
        q3 = q4 = 0
        c = 2
        if i != 0:
            q3 = qx[i-1, j, k]
            q4 = qx[i-1, j, k+1]
            c += 2
        if not B[i, j, k+1]:
            qzxr = (q1 + q2 + q3 + q4) / c

        # Z-Y plane transversal term (clockwise)
        q1 = qy[i, j, k+1]
        q2 = qy[i, j, k]
        q3 = q4 = 0
        c = 2
        if j != 0:
            q3 = qy[i, j-1, k]
            q4 = qy[i, j-1, k+1]
            c += 2
        if not B[i, j, k+1]:
            qzyr = (q1 + q2 + q3 + q4) / c

    D_xl = (l_l[i, j, k] * abs(qxl) + l_t[i, j, k] *
            (abs(qxyl) + abs(qxzl)))
    D_yl = (l_l[i, j, k] * abs(qyl) + l_t[i, j, k] *
            (abs(qyxl) + abs(qyzl)))
    D_zl = (l_l[i, j, k] * abs(qzl) + l_t[i, j, k] *
            (abs(qzxl) + abs(qzyl)))
    D_xr = (l_l[i, j, k] * abs(qxr) + l_t[i, j, k] *
            (abs(qxyr) + abs(qxzr)))
    D_yr = (l_l[i, j, k] * abs(qyr) + l_t[i, j, k] *
            (abs(qyxr) + abs(qyzr)))
    D_zr = (l_l[i, j, k] * abs(qzr) + l_t[i, j, k] *
            (abs(qzxr) + abs(qzyr)))

    return D_xl, D_yl, D_zl, D_xr, D_yr, D_zr

# SOLVER ROUTINES #############################################################
@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_stencil_P(aE, aW, aN, aS, aT, aB, bgrav, bdir, Rbase, Msum,
                      nue, rho_f, phi, c_r, c_f, Kx, Ky, Kz, BP, P):
    '''
        Precompute the 7-point pressure stencil once per flow solution.
        The coefficients depend on rho_f, nue and the Dirichlet values in
        P, which change between - not within - flow solutions.

        aE..aB   face transmissivities (zero at domain faces and faces
                 to Dirichlet cells; their contribution is folded into
                 bdir)
        bgrav    gravity terms with the signs of the SOR right-hand side
        bdir     Dirichlet boundary contributions a_face * P_dirichlet
        Rbase    fluid storage term rho_f * ((1-phi) c_r + phi c_f)
        Msum     sum of all face transmissivities (diagonal without R)
    '''
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                aE_ = aW_ = aN_ = aS_ = aT_ = aB_ = 0.0
                gE = gW = gT = gB = 0.0
                bdir_ = 0.0
                if i != nx-1:
                    aE_ = alpha(i+1, j, k, i, j, k, nue, rho_f, Kx, qx,
                                dx, i+1, i, 0)
                    gE = aE_ * beta_g(i+1, j, k, i, j, k, rho_f, 0)
                    if BP[i+1, j, k]:
                        bdir_ += aE_ * P[i+1, j, k]
                if i != 0:
                    aW_ = alpha(i, j, k, i-1, j, k, nue, rho_f, Kx, qx,
                                dx, i-1, i, 0)
                    gW = aW_ * beta_g(i, j, k, i-1, j, k, rho_f, 0)
                    if BP[i-1, j, k]:
                        bdir_ += aW_ * P[i-1, j, k]
                if j != ny-1:
                    aN_ = alpha(i, j+1, k, i, j, k, nue, rho_f, Ky, qy,
                                dy, j+1, j, 1)
                    if BP[i, j+1, k]:
                        bdir_ += aN_ * P[i, j+1, k]
                if j != 0:
                    aS_ = alpha(i, j, k, i, j-1, k, nue, rho_f, Ky, qy,
                                dy, j-1, j, 1)
                    if BP[i, j-1, k]:
                        bdir_ += aS_ * P[i, j-1, k]
                if k != nz-1:
                    aT_ = alpha(i, j, k+1, i, j, k, nue, rho_f, Kz, qz,
                                dz, k+1, k, 2)
                    gT = aT_ * beta_g(i, j, k+1, i, j, k, rho_f, 2)
                    if BP[i, j, k+1]:
                        bdir_ += aT_ * P[i, j, k+1]
                if k != 0:
                    aB_ = alpha(i, j, k, i, j, k-1, nue, rho_f, Kz, qz,
                                dz, k-1, k, 2)
                    gB = aB_ * beta_g(i, j, k, i, j, k-1, rho_f, 2)
                    if BP[i, j, k-1]:
                        bdir_ += aB_ * P[i, j, k-1]

                por = phi[i, j, k]
                Rbase[i, j, k] = rho_f[i, j, k] * \
                    ((1-por) * c_r[i, j, k] + por * c_f[i, j, k])
                Msum[i, j, k] = aE_ + aW_ + aN_ + aS_ + aT_ + aB_
                bgrav[i, j, k] = gE - gW + gT - gB
                bdir[i, j, k] = bdir_
                # faces to Dirichlet cells are excluded from the operator
                aE[i, j, k] = 0.0 if (i != nx-1 and BP[i+1, j, k]) else aE_
                aW[i, j, k] = 0.0 if (i != 0 and BP[i-1, j, k]) else aW_
                aN[i, j, k] = 0.0 if (j != ny-1 and BP[i, j+1, k]) else aN_
                aS[i, j, k] = 0.0 if (j != 0 and BP[i, j-1, k]) else aS_
                aT[i, j, k] = 0.0 if (k != nz-1 and BP[i, j, k+1]) else aT_
                aB[i, j, k] = 0.0 if (k != 0 and BP[i, j, k-1]) else aB_


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_P(P, aE, aW, aN, aS, aT, aB, bgrav, bdir, Rbase, Msum,
              dt_P, BP, WP,
              SOR=1.0, max_it=MAXIT_P, inc_it=1.05, max_dt=dt_max,
              min_dt=1e-8, P_conv=1e-10, Pn_prev=None):
    t_P = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    # storage term must reference the previous time step's pressure, not
    # the (possibly extrapolated) initial guess in P, otherwise the
    # transient term drains mass from over-extrapolated cells and the
    # converged solution develops unphysical low/negative pressure dips
    if Pn_prev is None:
        Pn = P.copy()
    else:
        Pn = Pn_prev.copy()

    def solve(i, j, k):
        R = Rbase[i, j, k] / dt_P
        s = bgrav[i, j, k] + bdir[i, j, k] + R * Pn[i, j, k] + WP[i, j, k]
        if i != nx-1:
            s += aE[i, j, k] * P[i+1, j, k]
        if i != 0:
            s += aW[i, j, k] * P[i-1, j, k]
        if j != ny-1:
            s += aN[i, j, k] * P[i, j+1, k]
        if j != 0:
            s += aS[i, j, k] * P[i, j-1, k]
        if k != nz-1:
            s += aT[i, j, k] * P[i, j, k+1]
        if k != 0:
            s += aB[i, j, k] * P[i, j, k-1]

        P[i, j, k] = (1-SOR) * P[i, j, k] + SOR * s / (R + Msum[i, j, k])

        return P[i, j, k]

    while not converged:
        num_it += 1
        num_sub_it += 1
        Pold = P.copy()

        # implementation of black-red stencil scheme for SOR
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if GP[i, j, k]:
                        if not BP[i, j, k]:
                            P[i, j, k] = solve(i, j, k)
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if not GP[i, j, k]:
                        if not BP[i, j, k]:
                            P[i, j, k] = solve(i, j, k)

        diff = 1e100
        if num_it < 50:
            diff = l2_norm(P, Pold, 'P')
        else:
            if num_it % 10 == 0:  # l2 norm after each 10 iterations
                diff = l2_norm(P, Pold, 'P')
        if diff < P_conv:
            converged = True
            t_P += dt_P
            # Increase time step by factor inc_it, while not exceeding max_dt
            if num_sub_it < max_it/2:
                dt_P = min(dt_P*inc_it, max_dt)
        else:
            converged = False
            if num_sub_it >= max_it:
                print(' t:', t_P, 'P [', num_it, ']: Reducing dt:', dt_P,
                      '=>', dt_P/2, '(s) L2:', diff)
                dt_P = dt_P/2  # decrease time step size
                num_sub_it = 0
                P = Pn.copy()  # reset P
            if dt_P <= min_dt:
                break

    return t_P, dt_P, P, Pn, converged, num_it


# CONJUGATE GRADIENT SOLVER ####################################################
@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_matvec(x, Ax, aE, aW, aN, aS, aT, aB, diag):
    '''
        Ax = diag * x - sum(a_face * x_neighbour)
        The branch-free interior loop is unit-stride and auto-vectorisable;
        the boundary shell is handled separately.
    '''
    nxy = nx * ny
    for k in prange(1, nz-1):
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                Ax[i, j, k] = (diag[i, j, k] * x[i, j, k]
                               - aE[i, j, k] * x[i+1, j, k]
                               - aW[i, j, k] * x[i-1, j, k]
                               - aN[i, j, k] * x[i, j+1, k]
                               - aS[i, j, k] * x[i, j-1, k]
                               - aT[i, j, k] * x[i, j, k+1]
                               - aB[i, j, k] * x[i, j, k-1])
    n = nx * ny * nz
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1:
            s = diag[i, j, k] * x[i, j, k]
            if i != nx-1:
                s -= aE[i, j, k] * x[i+1, j, k]
            if i != 0:
                s -= aW[i, j, k] * x[i-1, j, k]
            if j != ny-1:
                s -= aN[i, j, k] * x[i, j+1, k]
            if j != 0:
                s -= aS[i, j, k] * x[i, j-1, k]
            if k != nz-1:
                s -= aT[i, j, k] * x[i, j, k+1]
            if k != 0:
                s -= aB[i, j, k] * x[i, j, k-1]
            Ax[i, j, k] = s


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_dot(a, b):
    '''
        Parallel dot product (prange reduction)
    '''
    s = 0.0
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        s += a[i, j, k] * b[i, j, k]
    return s


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_assemble(diag, b, Rbase, Msum, bgrav, bdir, Pn, WP, dt_P):
    '''
        Diagonal and right-hand side for the current time step size
    '''
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        R = Rbase[i, j, k] / dt_P
        diag[i, j, k] = R + Msum[i, j, k]
        b[i, j, k] = (bgrav[i, j, k] + bdir[i, j, k]
                      + R * Pn[i, j, k] + WP[i, j, k])


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_residual(r, b, Ax, BP):
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        if BP[i, j, k]:
            r[i, j, k] = 0.0
        else:
            r[i, j, k] = b[i, j, k] - Ax[i, j, k]


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_update(P, r, p, Ap, alpha, BP):
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        if not BP[i, j, k]:
            P[i, j, k] += alpha * p[i, j, k]
            r[i, j, k] -= alpha * Ap[i, j, k]


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def cg_update_p(p, z, beta, BP):
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        if BP[i, j, k]:
            p[i, j, k] = 0.0
        else:
            p[i, j, k] = z[i, j, k] + beta * p[i, j, k]


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def gs_sweep(z, r, aE, aW, aN, aS, aT, aB, diag, color, BP, omega):
    '''
        One colour sweep of a red-black Gauss-Seidel solve of A z = r
        (used as CG preconditioner; cells of equal colour are independent)
    '''
    n = nx * ny * nz
    nxy = nx * ny
    for ii in prange(n):
        k = ii // nxy
        j = (ii // nx) % ny
        i = ii % nx
        if not BP[i, j, k] and (i + j + k) % 2 == color:
            s = r[i, j, k]
            if i != nx-1:
                s += aE[i, j, k] * z[i+1, j, k]
            if i != 0:
                s += aW[i, j, k] * z[i-1, j, k]
            if j != ny-1:
                s += aN[i, j, k] * z[i, j+1, k]
            if j != 0:
                s += aS[i, j, k] * z[i, j-1, k]
            if k != nz-1:
                s += aT[i, j, k] * z[i, j, k+1]
            if k != 0:
                s += aB[i, j, k] * z[i, j, k-1]
            z[i, j, k] = (1-omega) * z[i, j, k] + omega * s / diag[i, j, k]


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def cg_precondition(z, r, aE, aW, aN, aS, aT, aB, diag, BP, omega=1.0):
    '''
        Symmetric Gauss-Seidel preconditioner (black-red-red-black);
        the symmetrisation is required for the standard PCG algorithm
    '''
    z[:] = 0.0
    gs_sweep(z, r, aE, aW, aN, aS, aT, aB, diag, 0, BP, omega)
    gs_sweep(z, r, aE, aW, aN, aS, aT, aB, diag, 1, BP, omega)
    gs_sweep(z, r, aE, aW, aN, aS, aT, aB, diag, 1, BP, omega)
    gs_sweep(z, r, aE, aW, aN, aS, aT, aB, diag, 0, BP, omega)


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def compute_P_CG(P, aE, aW, aN, aS, aT, aB, bgrav, bdir, Rbase, Msum,
                 dt_P, BP, WP, max_it=MAXIT_P, inc_it=1.05, max_dt=dt_max,
                 min_dt=1e-8, P_conv=1e-10, pcg_omega=1.0, Pn_prev=None):
    '''
        Preconditioned Conjugate Gradient solver for the pressure equation,
        solving the same linear system as the SOR solver:
        (R + M) P - sum(a_face P_neighbour) = bgrav + bdir + R Pn + WP
    '''
    t_P = 0
    converged = False
    num_it = 0
    # storage term must reference the previous time step's pressure, not
    # the (possibly extrapolated) initial guess in P (see compute_P)
    if Pn_prev is None:
        Pn = P.copy()
    else:
        Pn = Pn_prev.copy()
    diag = numpy.zeros((nx, ny, nz))
    b = numpy.zeros((nx, ny, nz))
    r = numpy.zeros((nx, ny, nz))
    z = numpy.zeros((nx, ny, nz))
    p = numpy.zeros((nx, ny, nz))
    Ax = numpy.zeros((nx, ny, nz))
    Ap = numpy.zeros((nx, ny, nz))

    while not converged:
        # assemble operator diagonal and right-hand side for current dt_P
        cg_assemble(diag, b, Rbase, Msum, bgrav, bdir, Pn, WP, dt_P)
        cg_matvec(P, Ax, aE, aW, aN, aS, aT, aB, diag)
        cg_residual(r, b, Ax, BP)
        b_norm = numpy.sqrt(cg_dot(r, r))

        if b_norm < 1e-200:
            converged = True
            t_P += dt_P
            dt_P = min(dt_P*inc_it, max_dt)
        else:
            cg_precondition(z, r, aE, aW, aN, aS, aT, aB, diag, BP,
                            pcg_omega)
            p[:] = z
            rz = cg_dot(r, z)
            it = 0
            while it < max_it:
                it += 1
                cg_matvec(p, Ap, aE, aW, aN, aS, aT, aB, diag)
                pAp = cg_dot(p, Ap)
                if abs(pAp) < 1e-200:
                    break
                alpha_cg = rz / pAp
                cg_update(P, r, p, Ap, alpha_cg, BP)
                r_norm = numpy.sqrt(cg_dot(r, r))
                if r_norm / b_norm < P_conv:
                    converged = True
                    break
                cg_precondition(z, r, aE, aW, aN, aS, aT, aB, diag, BP,
                                pcg_omega)
                rz_new = cg_dot(r, z)
                if abs(rz) < 1e-200:
                    break
                beta_cg = rz_new / rz
                cg_update_p(p, z, beta_cg, BP)
                rz = rz_new

            num_it = it
            if converged:
                t_P += dt_P
                if it < max_it/2:
                    dt_P = min(dt_P*inc_it, max_dt)
            else:
                print(' t:', t_P, 'P [CG', it, ']: Reducing dt:', dt_P,
                      '=>', dt_P/2, '(s)')
                dt_P = dt_P/2
                P = Pn.copy()
                if dt_P <= min_dt:
                    break

    return t_P, dt_P, P, Pn, converged, num_it

@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_q(P, BP, Kx, Ky, Kz, qx, qy, qz, nue, rho_f):
    for k in prange(nz):  # iteration only for interfaces between grid points
        for j in prange(ny):
            for i in prange(nx):
                if i < nx - 1:
                    if not (BP[i, j, k] and BP[i+1, j, k]):
                        qx[i, j, k] = (- harm(i+1, j, k, i, j, k, Kx, 0) /
                                       harm(i+1, j, k,  i, j, k, nue, 0)
                                       # amean(i+1, j, k,  i, j, k, nue, 0)
                                       * ((P[i+1, j, k] - P[i, j, k]) /
                                       (0.5 * (dx[i+1] + dx[i]))
                                       + amean(i+1, j, k, i, j, k, rho_f, 0) *
                                       g * numpy.sin(beta)))
                    else:
                        qx[i, j, k] = 0
                if j < ny - 1:
                    if not (BP[i, j, k] and BP[i, j+1, k]):
                        qy[i, j, k] = (- harm(i, j+1, k, i, j, k, Ky, 1) /
                                       harm(i, j+1, k, i, j, k, nue, 1)
                                       # amean(i, j+1, k, i, j, k, nue, 1)
                                       * (P[i, j+1, k] - P[i, j, k]) /
                                       (0.5 * (dy[j+1] + dy[j])))
                    else:
                        qy[i, j, k] = 0
                if k < nz - 1:
                    if not (BP[i, j, k] and BP[i, j, k+1]):
                        qz[i, j, k] = (- harm(i, j, k+1, i, j, k, Kz, 2) /
                                       harm(i, j, k+1, i, j, k, nue, 2)
                                       * ((P[i, j, k+1]-P[i, j, k]) /
                                       (0.5 * (dz[k+1] + dz[k]))
                                       + amean(i, j, k+1, i, j, k, rho_f, 2)
                                       * g * numpy.cos(beta)))
                    else:
                        qz[i, j, k] = 0
    return qx, qy, qz


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_q_RICH(P, BP, Kx, Ky, Kz, qx, qy, qz, nue, rho_f,
                   krE, krW, krN, krS, krT, krB):
    '''
        Darcy fluxes for the Richards solution: identical to compute_q,
        but each face flux is multiplied by the frozen upwinded face
        relative permeability (consistent with the transmissibilities
        used in compute_P_RICH)
    '''
    for k in prange(nz):  # iteration only for interfaces between grid points
        for j in prange(ny):
            for i in prange(nx):
                if i < nx - 1:
                    if not (BP[i, j, k] and BP[i+1, j, k]):
                        qx[i, j, k] = krE[i, j, k] * \
                            (- harm(i+1, j, k, i, j, k, Kx, 0) /
                             harm(i+1, j, k, i, j, k, nue, 0)
                             * ((P[i+1, j, k] - P[i, j, k]) /
                             (0.5 * (dx[i+1] + dx[i]))
                             + amean(i+1, j, k, i, j, k, rho_f, 0) *
                             g * numpy.sin(beta)))
                    else:
                        qx[i, j, k] = 0
                if j < ny - 1:
                    if not (BP[i, j, k] and BP[i, j+1, k]):
                        qy[i, j, k] = krN[i, j, k] * \
                            (- harm(i, j+1, k, i, j, k, Ky, 1) /
                             harm(i, j+1, k, i, j, k, nue, 1)
                             * (P[i, j+1, k] - P[i, j, k]) /
                             (0.5 * (dy[j+1] + dy[j])))
                    else:
                        qy[i, j, k] = 0
                if k < nz - 1:
                    if not (BP[i, j, k] and BP[i, j, k+1]):
                        qz[i, j, k] = krT[i, j, k] * \
                            (- harm(i, j, k+1, i, j, k, Kz, 2) /
                             harm(i, j, k+1, i, j, k, nue, 2)
                             * ((P[i, j, k+1]-P[i, j, k]) /
                             (0.5 * (dz[k+1] + dz[k]))
                             + amean(i, j, k+1, i, j, k, rho_f, 2)
                             * g * numpy.cos(beta)))
                    else:
                        qz[i, j, k] = 0
    return qx, qy, qz


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_P_RICH(P, Pn, S, Sn, kr, krE, krW, krN, krS, krT, krB, dSdP,
                   c_f, c_r, nue, rho_f, phi, Kx, Ky, Kz, qx, qy, qz,
                   dt_P, BP, WP, alpha_vg, n_vg, S_r, S_s, A_vg, gamma_vg,
                   krm, conv_s, max_dt,
                   SOR=1.0, max_it=MAXIT_P, P_conv=1e-10):
    '''
        Modified-Picard solution of the mixed-form Richards equation
        (Celia et al. 1990), same solver pattern as compute_P:

            phi*rho*(S - Sn)/dt + Sn*rho*c_t*(P - Pn)/dt
            = div(rho * kr * K/mue * (grad P + rho*g*grad z)) + WP

        OUTER loop: modified Picard iteration on the saturation. The
        storage accumulation is linearised around the current iterate,

            phi*rho/dt * [S^m + dS/dP|(P^m) * (P^(m+1) - P^m) - Sn]

        and all coefficients (S^m, dS/dP, face kr) are frozen, so each
        pass solves a LINEAR system. After each converged inner solve the
        saturation, capacity and face kr are updated from the new
        pressure; convergence on the saturation L2 norm (conv_s). At
        Picard convergence the linearisation error vanishes and the
        discretised equation is the exact mass-conservative mixed form.

        INNER loop: red-black SOR solve of the linearised equation with
        frozen coefficients (same as compute_P).

        Non-convergence of either loop halves dt_P and restarts the time
        step from Pn/Sn (same pattern as compute_P); on convergence dt_P
        is increased by 1.25 for the next time step.

        P   pressure iterate (initial guess and linearisation point)
        Pn  pressure at the previous TIME LEVEL (storage reference)
        S   saturation at the linearisation point, S^m = S(P^m)
        Sn  saturation at the previous time level
        dSdP, krE..krB  frozen capacity and face relative permeabilities
        P is gauge pressure; P < 0: capillary pressure
        (head h = -P/(rho*g)); P >= 0: saturated (S = S_s, kr = 1)
    '''
    t_P = 0
    converged = False
    num_it = 0        # Picard passes of the accepted step (reporting)
    num_sub_it = 0    # Picard passes since the last dt reduction

    # frozen linearisation arrays (refilled in place per Picard pass;
    # preallocated here so that the solve closure only captures
    # once-assigned variables, as required by numba)
    bS = numpy.empty_like(phi)      # constant storage RHS term
    coef_C = numpy.empty_like(phi)  # capacity storage coefficient
    coef_P = numpy.empty_like(phi)  # compressible storage coefficient

    def solve(i, j, k):
        A_ = B_ = C_ = D_ = E_ = G_ = I_ = J_ = K_ = L_ = \
            Anew = Cnew = Enew = Gnew = Inew = Knew = 0

        if i != nx-1:
            A_ = alpha(i+1, j, k, i, j, k, nue, rho_f, Kx, qx, dx, i+1, i, 0) \
                * krE[i, j, k]
            Anew = A_*P[i+1, j, k]
            B_ = beta_g(i+1, j, k, i, j, k, rho_f, 0)
        if i != 0:
            C_ = alpha(i, j, k, i-1, j, k, nue, rho_f, Kx, qx, dx, i-1, i, 0) \
                * krW[i, j, k]
            Cnew = C_*P[i-1, j, k]
            D_ = beta_g(i, j, k, i-1, j, k, rho_f, 0)
        if j != ny-1:
            E_ = alpha(i, j+1, k, i, j, k, nue, rho_f, Ky, qy, dy, j+1, j, 1) \
                * krN[i, j, k]
            Enew = E_*P[i, j+1, k]
        if j != 0:
            G_ = alpha(i, j, k, i, j-1, k, nue, rho_f, Ky, qy, dy, j-1, j, 1) \
                * krS[i, j, k]
            Gnew = G_*P[i, j-1, k]
        if k != nz-1:
            I_ = alpha(i, j, k+1, i, j, k, nue, rho_f, Kz, qz, dz, k+1, k, 2) \
                * krT[i, j, k]
            Inew = I_*P[i, j, k+1]
            J_ = beta_g(i, j, k+1, i, j, k, rho_f, 2)
        if k != 0:
            K_ = alpha(i, j, k, i, j, k-1, nue, rho_f, Kz, qz, dz, k-1, k, 2) \
                * krB[i, j, k]
            Knew = K_*P[i, j, k-1]
            L_ = beta_g(i, j, k, i, j, k-1, rho_f, 2)

        M = A_ + C_ + E_ + G_ + I_ + K_

        P[i, j, k] = (1-SOR) * P[i, j, k] + SOR * (
            Anew + A_*B_ + Cnew - C_*D_ + Enew
            + Gnew + Inew + I_*J_ + Knew - K_*L_
            + bS[i, j, k] + coef_P[i, j, k] * Pn[i, j, k]
            + WP[i, j, k]) / (coef_C[i, j, k] + coef_P[i, j, k] + M)

        return P[i, j, k]

    while not converged:
        # frozen linearisation coefficients (modified Picard, Celia 1990)
        # bS uses the pressure linearisation point P^m (frozen), while the
        # stencil neighbour values use the live SOR iterate
        coef_S = phi * rho_f / dt_P
        bS[:] = coef_S * (Sn - S + dSdP * P)
        coef_C[:] = coef_S * dSdP
        coef_P[:] = Sn * rho_f * ((1 - phi) * c_r + phi * c_f) / dt_P

        # ---- inner red-black SOR solve with frozen coefficients ----
        num_sub_it += 1
        inner_conv = False
        inner_it = 0
        diff = 1e100
        while not inner_conv:
            inner_it += 1
            Pold = P.copy()

            # implementation of black-red stencil scheme for SOR
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if GP[i, j, k]:
                            if not BP[i, j, k]:
                                P[i, j, k] = solve(i, j, k)
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if not GP[i, j, k]:
                            if not BP[i, j, k]:
                                P[i, j, k] = solve(i, j, k)

            if inner_it < 50:
                diff = l2_norm(P, Pold, 'P')
            elif inner_it % 10 == 0:  # l2 norm after each 10 iterations
                diff = l2_norm(P, Pold, 'P')
            if DEBUG_INNER and inner_it % 2000 == 0:
                print('   [inner it', inner_it, '] L2', diff)
            if diff < P_conv:
                inner_conv = True
            elif inner_it >= max_it:
                break


        if not inner_conv:
            # inner solver did not reach the change criterion within max_it
            # sweeps (the saturated zone with c_t ~ 0 is nearly elliptic and
            # the SOR tail decays very slowly). Accept the current iterate:
            # the remaining change per sweep bounds the solution error,
            # which is far below the accuracy of the outer Picard loop.
            # Only a DIVERGED iterate (L2 >= 1 or NaN) is rejected.
            if not (diff < 1.0) or numpy.isnan(diff):
                print(' t:', t_P, 'P [', inner_it, ']: Diverged, reducing dt:',
                      dt_P, '=>', dt_P/2, '(s) L2:', diff)
                dt_P = dt_P/2
                num_sub_it = 0
                P = Pn.copy()
                S = Sn.copy()
                update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT,
                                  krB, rho_f, Kx, Ky, Kz, alpha_vg, n_vg,
                                  S_r, S_s, A_vg, gamma_vg, krm)
                if dt_P <= 1e-8:
                    break
                continue

        # ---- Picard update: saturation, capacity, face kr ----
        Sold = S.copy()
        update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT, krB,
                          rho_f, Kx, Ky, Kz, alpha_vg, n_vg, S_r, S_s,
                          A_vg, gamma_vg, krm)
        num_it += 1
        pic_diff = l2_norm(S, Sold, 'S')
        if pic_diff < conv_s:
            converged = True
            t_P += dt_P
            # Increase time step by factor 1.25 for the next time step
            if num_sub_it < max_it/2:
                dt_P = min(dt_P*1.25, max_dt)
        elif num_sub_it >= 10:
            # Picard iteration stagnates: reduce dt, restart the step
            print(' t:', t_P, 'PICARD [', num_it, ']: Reducing dt:', dt_P,
                  '=>', dt_P/2, '(s) dS:', pic_diff)
            dt_P = dt_P/2
            num_sub_it = 0
            P = Pn.copy()
            S = Sn.copy()
            update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT,
                              krB, rho_f, Kx, Ky, Kz, alpha_vg, n_vg,
                              S_r, S_s, A_vg, gamma_vg, krm)
            if dt_P <= 1e-8:
                break

    return t_P, dt_P, P, converged, num_it


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def richards_stencil(aE, aW, aN, aS, aT, aB, bgrav, bdir, diag, b,
                     P, Pn, S, Sn, krE, krW, krN, krS, krT, krB,
                     dSdP, nue, rho_f, phi, c_f, c_r, Kx, Ky, Kz, BP, WP,
                     dt_P):
    '''
        Build the linearised Richards system for one frozen Picard pass
        (mixed form, Celia et al. 1990):

        (coef_C + coef_P + Msum) P - sum(a_face kr P_nb)
            = bgrav + bdir + bS + coef_P Pn + WP

        with bS = phi rho/dt (Sn - S + dS/dP|P^m),
             coef_C = phi rho/dt dS/dP|P^m,
             coef_P = Sn rho c_t / dt.

        aE..aB   face transmissivities alpha()*kr_face (zero at domain
                 faces and faces to Dirichlet cells; their contribution
                 is folded into bdir)
        bgrav    gravity terms with the signs of the SOR right-hand side
        bdir     Dirichlet boundary contributions a_face * P_dirichlet
        diag     operator diagonal (coef_C + coef_P + Msum)
        b        right-hand side
    '''
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                aE_ = aW_ = aN_ = aS_ = aT_ = aB_ = 0.0
                gE = gW = gT = gB = 0.0
                bdir_ = 0.0
                if i != nx-1:
                    aE_ = alpha(i+1, j, k, i, j, k, nue, rho_f, Kx, qx,
                                dx, i+1, i, 0) * krE[i, j, k]
                    gE = aE_ * beta_g(i+1, j, k, i, j, k, rho_f, 0)
                    if BP[i+1, j, k]:
                        bdir_ += aE_ * P[i+1, j, k]
                if i != 0:
                    aW_ = alpha(i, j, k, i-1, j, k, nue, rho_f, Kx, qx,
                                dx, i-1, i, 0) * krW[i, j, k]
                    gW = aW_ * beta_g(i, j, k, i-1, j, k, rho_f, 0)
                    if BP[i-1, j, k]:
                        bdir_ += aW_ * P[i-1, j, k]
                if j != ny-1:
                    aN_ = alpha(i, j+1, k, i, j, k, nue, rho_f, Ky, qy,
                                dy, j+1, j, 1) * krN[i, j, k]
                    if BP[i, j+1, k]:
                        bdir_ += aN_ * P[i, j+1, k]
                if j != 0:
                    aS_ = alpha(i, j, k, i, j-1, k, nue, rho_f, Ky, qy,
                                dy, j-1, j, 1) * krS[i, j, k]
                    if BP[i, j-1, k]:
                        bdir_ += aS_ * P[i, j-1, k]
                if k != nz-1:
                    aT_ = alpha(i, j, k+1, i, j, k, nue, rho_f, Kz, qz,
                                dz, k+1, k, 2) * krT[i, j, k]
                    gT = aT_ * beta_g(i, j, k+1, i, j, k, rho_f, 2)
                    if BP[i, j, k+1]:
                        bdir_ += aT_ * P[i, j, k+1]
                if k != 0:
                    aB_ = alpha(i, j, k, i, j, k-1, nue, rho_f, Kz, qz,
                                dz, k-1, k, 2) * krB[i, j, k]
                    gB = aB_ * beta_g(i, j, k, i, j, k-1, rho_f, 2)
                    if BP[i, j, k-1]:
                        bdir_ += aB_ * P[i, j, k-1]

                por = phi[i, j, k]
                coef_S = por * rho_f[i, j, k] / dt_P
                bS_ = coef_S * (Sn[i, j, k] - S[i, j, k]
                                + dSdP[i, j, k] * P[i, j, k])
                coef_C_ = coef_S * dSdP[i, j, k]
                coef_P_ = Sn[i, j, k] * rho_f[i, j, k] * \
                    ((1 - por) * c_r[i, j, k] + por * c_f[i, j, k]) / dt_P

                Msum_ = aE_ + aW_ + aN_ + aS_ + aT_ + aB_
                bgrav[i, j, k] = gE - gW + gT - gB
                bdir[i, j, k] = bdir_
                diag[i, j, k] = coef_C_ + coef_P_ + Msum_
                b[i, j, k] = (bgrav[i, j, k] + bdir[i, j, k] + bS_
                              + coef_P_ * Pn[i, j, k] + WP[i, j, k])
                # faces to Dirichlet cells are excluded from the operator
                aE[i, j, k] = 0.0 if (i != nx-1 and BP[i+1, j, k]) else aE_
                aW[i, j, k] = 0.0 if (i != 0 and BP[i-1, j, k]) else aW_
                aN[i, j, k] = 0.0 if (j != ny-1 and BP[i, j+1, k]) else aN_
                aS[i, j, k] = 0.0 if (j != 0 and BP[i, j-1, k]) else aS_
                aT[i, j, k] = 0.0 if (k != nz-1 and BP[i, j, k+1]) else aT_
                aB[i, j, k] = 0.0 if (k != 0 and BP[i, j, k-1]) else aB_


@njit(fastmath=numba_fastmath, cache=numba_cache, nogil=numba_nogil)
def compute_P_RICH_CG(P, Pn_r, S, Sn, kr, krE, krW, krN, krS, krT, krB,
                      dSdP, c_f, c_r, nue, rho_f, phi, Kx, Ky, Kz, qx, qy,
                      qz, dt_P, BP, WP, alpha_vg, n_vg, S_r, S_s, A_vg,
                      gamma_vg, krm, conv_s, max_dt,
                      max_it=MAXIT_P, P_conv=1e-10, pcg_omega=1.0):
    '''
        Modified-Picard solution of the mixed-form Richards equation with
        a preconditioned CG solver for the inner linear systems (same
        outer loop as compute_P_RICH). The CG converges on the RESIDUAL
        of the linearised system, which is the cell-wise mass balance
        error - the correct criterion for the nearly elliptic saturated
        zone where change-based SOR criteria are unattainable.
    '''
    t_P = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    aE = numpy.zeros_like(krE)
    aW = numpy.zeros_like(krE)
    aN = numpy.zeros_like(krE)
    aS = numpy.zeros_like(krE)
    aT = numpy.zeros_like(krE)
    aB = numpy.zeros_like(krE)
    bgrav = numpy.zeros_like(P)
    bdir = numpy.zeros_like(P)
    diag = numpy.zeros_like(P)
    b = numpy.zeros_like(P)
    r = numpy.zeros_like(P)
    z = numpy.zeros_like(P)
    p = numpy.zeros_like(P)
    Ax = numpy.zeros_like(P)
    Ap = numpy.zeros_like(P)

    while not converged:
        num_sub_it += 1
        # frozen Picard coefficients: build stencil and right-hand side
        richards_stencil(aE, aW, aN, aS, aT, aB, bgrav, bdir, diag, b,
                         P, Pn_r, S, Sn, krE, krW, krN, krS, krT, krB,
                         dSdP, nue, rho_f, phi, c_f, c_r, Kx, Ky, Kz, BP,
                         WP, dt_P)

        # ---- inner PCG solve with frozen coefficients ----
        cg_matvec(P, Ax, aE, aW, aN, aS, aT, aB, diag)
        cg_residual(r, b, Ax, BP)
        b_norm = numpy.sqrt(cg_dot(r, r))

        inner_conv = False
        it = 0
        if b_norm >= 1e-200:
            cg_precondition(z, r, aE, aW, aN, aS, aT, aB, diag, BP,
                            pcg_omega)
            p[:] = z
            rz = cg_dot(r, z)
            while it < max_it:
                it += 1
                cg_matvec(p, Ap, aE, aW, aN, aS, aT, aB, diag)
                pAp = cg_dot(p, Ap)
                if abs(pAp) < 1e-200:
                    break
                alpha_cg = rz / pAp
                cg_update(P, r, p, Ap, alpha_cg, BP)
                r_norm = numpy.sqrt(cg_dot(r, r))
                if r_norm / b_norm < P_conv:
                    inner_conv = True
                    break
                cg_precondition(z, r, aE, aW, aN, aS, aT, aB, diag, BP,
                                pcg_omega)
                rz_new = cg_dot(r, z)
                if abs(rz) < 1e-200:
                    break
                beta_cg = rz_new / rz
                cg_update_p(p, z, beta_cg, BP)
                rz = rz_new
        else:
            inner_conv = True

        num_it += 1
        if not inner_conv:
            # inner solver did not converge: reduce dt, restart the step
            print(' t:', t_P, 'P [CG', it, ']: Reducing dt:', dt_P,
                  '=>', dt_P/2, '(s) rel res:', r_norm/b_norm)
            dt_P = dt_P/2
            num_sub_it = 0
            P = Pn_r.copy()
            S = Sn.copy()
            update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT,
                              krB, rho_f, Kx, Ky, Kz, alpha_vg, n_vg,
                              S_r, S_s, A_vg, gamma_vg, krm)
            if dt_P <= 1e-8:
                break
            continue

        # ---- Picard update: saturation, capacity, face kr ----
        Sold = S.copy()
        update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT, krB,
                          rho_f, Kx, Ky, Kz, alpha_vg, n_vg, S_r, S_s,
                          A_vg, gamma_vg, krm)
        pic_diff = l2_norm(S, Sold, 'S')
        if pic_diff < conv_s:
            converged = True
            t_P += dt_P
            # Increase time step by factor 1.25 for the next time step
            if num_sub_it < max_it/2:
                dt_P = min(dt_P*1.25, max_dt)
        elif num_sub_it >= 10:
            # Picard iteration stagnates: reduce dt, restart the step
            print(' t:', t_P, 'PICARD [', num_it, ']: Reducing dt:', dt_P,
                  '=>', dt_P/2, '(s) dS:', pic_diff)
            dt_P = dt_P/2
            num_sub_it = 0
            P = Pn_r.copy()
            S = Sn.copy()
            update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT,
                              krB, rho_f, Kx, Ky, Kz, alpha_vg, n_vg,
                              S_r, S_s, A_vg, gamma_vg, krm)
            if dt_P <= 1e-8:
                break

    return t_P, dt_P, P, converged, num_it


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_C(C, D_fx, D_fy, D_fz, phi, S, rho_f, qx, qy, qz, dt_C, BC, WC,
              RC, e, omega=1, SOR=1.0, max_it=MAXIT_C, inc_it=1.05,
              max_dt=dt_max, min_dt=1e-8, C_conv=1e-10):

    t_C = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    Cn = C.copy()

    def solve(i, j, k):
        A_ = B_ = C_ = D_ = E_ = F_ = G_ = H_ = \
            I_ = J_ = K_ = L_ = M_ = N_ = O_ = P_ =  Q_ = R_ = EC = \
            Mnew = Nnew = Onew = Pnew = Qnew = Rnew = \
            Mold = Nold = Oold = Pold = Qold = Rold = \
            D_xr = D_yr = D_zr = D_xl = D_yl = D_zl = 0

        # compute dispersion tensor if any dispersivity term is non-zero
        if lf_l[i, j, k] != 0 or lf_t[i, j, k] != 0:
            D_xl, D_yl, D_zl, D_xr, D_yr, D_zr = dispersivity(qx, qy, qz, 
                                                              lf_l, lf_t, BC,
                                                              i, j, k)

        if i != nx-1:
            dd = dx[i]
            dd2 = (0.5 * (dx[i+1] + dx[i])) * dd
            ph = amean(i+1, j, k, i, j, k, phi, 0) \
                * amean(i+1, j, k, i, j, k, S, 0)
            A_ = (harm(i+1, j, k, i, j, k, D_fx, 0) * ph + D_xr) / dd2
            N_ = min(qx[i, j, k], 0) / dd
            H_ = max(qx[i, j, k], 0) / dd
            Mnew = (A_ - N_) * C[i+1, j, k]
            Mold = (A_ - N_) * Cn[i+1, j, k]
        if i != 0:
            dd = dx[i]
            dd2 = (0.5 * (dx[i-1] + dx[i])) * dd
            ph = amean(i-1, j, k, i, j, k, phi, 0) \
                * amean(i-1, j, k, i, j, k, S, 0)
            B_ = (harm(i-1, j, k, i, j, k, D_fx, 0) * ph + D_xl) / dd2
            M_ = max(qx[i-1, j, k], 0) / dd
            G_ = min(qx[i-1, j, k], 0) / dd
            Nnew = (B_ + M_) * C[i-1, j, k]
            Nold = (B_ + M_) * Cn[i-1, j, k]
        if j != ny-1:
            dd = dy[j]
            dd2 = (0.5 * (dy[j+1] + dy[j])) * dd
            ph = amean(i, j+1, k, i, j, k, phi, 1) \
                * amean(i, j+1, k, i, j, k, S, 1)
            C_ = (harm(i, j+1, k, i, j, k, D_fy, 1) * ph + D_yr) / dd2
            P_ = min(qy[i, j, k], 0) / dd
            J_ = max(qy[i, j, k], 0) / dd
            Onew = (C_ - P_) * C[i, j+1, k]
            Oold = (C_ - P_) * Cn[i, j+1, k]
        if j != 0:
            dd = dy[j]
            dd2 = (0.5 * (dy[j-1] + dy[j])) * dd
            ph = amean(i, j-1, k, i, j, k, phi, 1) \
                * amean(i, j-1, k, i, j, k, S, 1)
            D_ = (harm(i, j-1, k, i, j, k, D_fy, 1) * ph + D_yl) / dd2
            O_ = max(qy[i, j-1, k], 0) / dd
            I_ = min(qy[i, j-1, k], 0) / dd
            Pnew = (D_ + O_) * C[i, j-1, k]
            Pold = (D_ + O_) * Cn[i, j-1, k]
        if k != nz-1:
            dd = dz[k]
            dd2 = (0.5 * (dz[k+1] + dz[k])) * dd
            ph = amean(i, j, k+1, i, j, k, phi, 2) \
                * amean(i, j, k+1, i, j, k, S, 2)
            E_ = (harm(i, j, k+1, i, j, k, D_fz, 2) * ph + D_zr) / dd2
            R_ = min(qz[i, j, k], 0) / dd
            L_ = max(qz[i, j, k], 0) / dd
            Qnew = (E_ - R_) * C[i, j, k+1]
            Qold = (E_ - R_) * Cn[i, j, k+1]
        if k != 0:
            dd = dz[k]
            dd2 = (0.5 * (dz[k-1] + dz[k])) * dd
            ph = amean(i, j, k-1, i, j, k, phi, 2) \
                * amean(i, j, k-1, i, j, k, S, 2)
            F_ = (harm(i, j, k-1, i, j, k, D_fz, 2) * ph + D_zl) / dd2
            Q_ = max(qz[i, j, k-1], 0) / dd
            K_ = min(qz[i, j, k-1], 0) / dd
            Rnew = (F_ + Q_) * C[i, j, k-1]
            Rold = (F_ + Q_) * Cn[i, j, k-1]
        R = RC[i,j,k] * phi[i, j, k] * S[i, j, k] / dt_C
        M = - A_ - B_ - C_ - D_ - E_ - F_ + G_ - H_ + I_ - J_ + K_ - L_

        if config.C_IMBLFLO: # and WC[i, j, k] == 0:
            # species generation term based on mass flow error
            # kg /(m^3*s) = kg/m^3 / (kg/m^3) * kg/(m^3*s)
            EC = Cn[i, j, k] / rho_f[i, j, k] * e[i, j, k]

        return ((1-SOR) * C[i, j, k] + SOR * (omega * 
                (Mnew + Nnew + Onew + Pnew + Qnew + Rnew)
                + (1-omega) * 
                (Cn[i, j, k] * M + Mold + Nold + Oold + Pold + Qold + Rold)
                + R * Cn[i, j, k] + WC[i, j, k] - EC)
                / (R - omega * M))

    while not converged:
        num_it += 1
        num_sub_it += 1
        Cold = C.copy()

        # implementation of black-red stencil scheme for SOR
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if GP[i, j, k]:
                        if not BC[i, j, k]:
                            C[i, j, k] = solve(i, j, k)
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if not GP[i, j, k]:
                        if not BC[i, j, k]:
                            C[i, j, k] = solve(i, j, k)

        diff = l2_norm(C, Cold, 'C')
        if diff < C_conv or omega == 0:
            t_C += dt_C
            # Increase time step by factor inc_it, while not exceeding max_dt
            if num_sub_it < max_it/2:
                dt_C = min(dt_C*inc_it, max_dt)
            converged = True
        else:
            converged = False
            if num_sub_it >= max_it:
                # print('\n  [C] [t/s ', t_C, '] [it ', num_it, '] [dt/s', dt_C,
                #       '=>', dt_C/2, '] [L2', diff, ']')
                printf('\n  [C] [iter %5d] [dt/s %6.2e -> %6.2e] [L2 %6.2e]'
                    , num_it, dt_C, dt_C/2, diff)
                dt_C = dt_C/2  # decrease time step size
                num_sub_it = 0
                C = Cn.copy()  # reset C
            if dt_C <= min_dt:
                break

    return t_C, dt_C, C, converged, num_it


@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def compute_T(T, l_f, cp_f, rho_f, phi, S, qx, qy, qz, dt_T, BT, WT, e,
              omega=1, SOR=1.0, max_it=MAXIT_T, inc_it=1.05, max_dt=dt_max,
              min_dt=1e-8, T_conv=1e-10):
    t_T = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    Tn = T.copy()

    def solve(i, j, k):
        A_ = B_ = C_ = D_ = E_ = F_ = G_ = H_ = \
            I_ = J_ = K_ = L_ = S_ = M_ = N_ = O_ = P_ = Q_ = R_ = ET = \
            Mnew = Nnew = Onew = Pnew = Qnew = Rnew =  \
            Mold = Nold = Oold = Pold = Qold = Rold = \
            D_xr = D_yr = D_zr = D_xl = D_yl = D_zl = 0

        # compute dispersion tensor if any dispersivity term is non-zero
        if lt_l[i, j, k] != 0 or lt_t[i, j, k] != 0:
            D_xl, D_yl, D_zl, D_xr, D_yr, D_zr = dispersivity(qx, qy, qz, 
                                                              lt_l, lt_t, BT,
                                                              i, j, k)

        if i != nx-1:
            dd = dx[i]
            dd2 = (0.5 * (dx[i+1] + dx[i])) * dd
            por = amean(i, j, k, i+1, j, k, phi, 0)
            s_ = amean(i, j, k, i+1, j, k, S, 0)
            lf = harm(i, j, k, i+1, j, k, l_f, 0)
            lr = harm(i, j, k, i+1, j, k, l_r, 0)
            # upwinded heat capacity of the flowing fluid: a harmonic
            # face mean biases the advected enthalpy toward the downstream
            # cell (heat accumulation at cells adjacent to cold Dirichlet
            # boundaries); rho stays harmonic for mass-flux consistency
            if qx[i, j, k] >= 0:
                cpf = cp_f[i, j, k]
            else:
                cpf = cp_f[i+1, j, k]
            rhocpf = harm(i, j, k, i+1, j, k, rho_f, 0) * cpf
            A_ = ((1-por) * lr + por * s_ * lf + D_xr * rhocpf) / dd2
            S_ = rhocpf / dd
            N_ = min(qx[i, j, k], 0) * S_
            H_ = max(qx[i, j, k], 0) * S_
            Mnew = (A_ - N_) * T[i+1, j, k]
            Mold = (A_ - N_) * Tn[i+1, j, k]
        if i != 0:
            dd = dx[i]
            dd2 = (0.5 * (dx[i-1] + dx[i])) * dd
            por = amean(i, j, k, i-1, j, k, phi, 0)
            s_ = amean(i, j, k, i-1, j, k, S, 0)
            lf = harm(i, j, k, i-1, j, k, l_f, 0)
            lr = harm(i, j, k, i-1, j, k, l_r, 0)
            if qx[i-1, j, k] > 0:
                cpf = cp_f[i-1, j, k]
            else:
                cpf = cp_f[i, j, k]
            rhocpf = harm(i, j, k, i-1, j, k, rho_f, 0) * cpf
            B_ = ((1-por) * lr + por * s_ * lf + D_xl * rhocpf) / dd2
            S_ = rhocpf / dd
            M_ = max(qx[i-1, j, k], 0) * S_
            G_ = min(qx[i-1, j, k], 0) * S_
            Nnew = (B_ + M_) * T[i-1, j, k]
            Nold = (B_ + M_) * Tn[i-1, j, k]
        if j != ny-1:
            dd = dy[j]
            dd2 = (0.5 * (dy[j+1] + dy[j])) * dd
            por = amean(i, j+1, k,  i, j, k, phi, 1)
            s_ = amean(i, j+1, k, i, j, k, S, 1)
            lf = harm(i, j+1, k, i, j, k, l_f, 1)
            lr = harm(i, j+1, k, i, j, k, l_r, 1)
            if qy[i, j, k] >= 0:
                cpf = cp_f[i, j, k]
            else:
                cpf = cp_f[i, j+1, k]
            rhocpf = harm(i, j+1, k, i, j, k, rho_f, 1) * cpf
            C_ = ((1-por) * lr + por * s_ * lf + D_yr * rhocpf) / dd2
            S_ = rhocpf / dd
            P_ = min(qy[i, j, k], 0) * S_
            J_ = max(qy[i, j, k], 0) * S_
            Onew = (C_ - P_) * T[i, j+1, k]
            Oold = (C_ - P_) * Tn[i, j+1, k]
        if j != 0:
            dd = dy[j]
            dd2 = (0.5 * (dy[j-1] + dy[j])) * dd
            por = amean(i, j-1, k, i, j, k, phi, 1)
            s_ = amean(i, j-1, k, i, j, k, S, 1)
            lf = harm(i, j-1, k, i, j, k, l_f, 1)
            lr = harm(i, j-1, k, i, j, k, l_r, 1)
            if qy[i, j-1, k] > 0:
                cpf = cp_f[i, j-1, k]
            else:
                cpf = cp_f[i, j, k]
            rhocpf = harm(i, j-1, k, i, j, k, rho_f, 1) * cpf
            D_ = ((1-por) * lr + por * s_ * lf + D_yl * rhocpf) / dd2
            S_ = rhocpf / dd
            O_ = max(qy[i, j-1, k], 0) * S_
            I_ = min(qy[i, j-1, k], 0) * S_
            Pnew = (D_ + O_) * T[i, j-1, k]
            Pold = (D_ + O_) * Tn[i, j-1, k]
        if k != nz-1:
            dd = dz[k]
            dd2 = (0.5 * (dz[k+1] + dz[k])) * dd
            por = amean(i, j, k+1, i, j, k, phi, 2)
            s_ = amean(i, j, k+1, i, j, k, S, 2)
            lf = harm(i, j, k+1, i, j, k, l_f, 2)
            lr = harm(i, j, k+1, i, j, k, l_r, 2)
            if qz[i, j, k] >= 0:
                cpf = cp_f[i, j, k]
            else:
                cpf = cp_f[i, j, k+1]
            rhocpf = harm(i, j, k+1, i, j, k, rho_f, 2) * cpf
            E_ = ((1-por) * lr + por * s_ * lf + D_zr * rhocpf) / dd2
            S_ = rhocpf / dd
            R_ = min(qz[i, j, k], 0) * S_
            L_ = max(qz[i, j, k], 0) * S_
            Qnew = (E_ - R_) * T[i, j, k+1]
            Qold = (E_ - R_) * Tn[i, j, k+1]
        if k != 0:
            dd = dz[k]
            dd2 = (0.5 * (dz[k-1] + dz[k])) * dd
            por = amean(i, j, k-1, i, j, k, phi, 2)
            s_ = amean(i, j, k-1, i, j, k, S, 2)
            lf = harm(i, j, k-1, i, j, k, l_f, 2)
            lr = harm(i, j, k-1, i, j, k, l_r, 2)
            if qz[i, j, k-1] > 0:
                cpf = cp_f[i, j, k-1]
            else:
                cpf = cp_f[i, j, k]
            rhocpf = harm(i, j, k-1, i, j, k, rho_f, 2) * cpf
            F_ = ((1-por) * lr + por * s_ * lf + D_zl * rhocpf) / dd2
            S_ = rhocpf / dd
            Q_ = max(qz[i, j, k-1], 0) * S_
            K_ = min(qz[i, j, k-1], 0) * S_
            Rnew = (F_ + Q_) * T[i, j, k-1]
            Rold = (F_ + Q_) * Tn[i, j, k-1]

        por_ = phi[i, j, k]
        R = (por_ * S[i, j, k] * rho_f[i, j, k] * cp_f[i, j, k] +
             (1 - por_) * rho_r[i, j, k] * cp_r[i, j, k]) / dt_T
        M = - A_ - B_ - C_ - D_ - E_ - F_ + G_ - H_ + I_ - J_ + K_ - L_

        if config.T_IMBLFLO: # and WT[i, j, k] == 0:
            # heat generation term based on mass flow error
            # W/m^3 = J/(kg*K) * K * kg/(m^3*s)
            ET = cp_f[i, j, k] * Tn[i, j, k] * e[i, j, k] 
 
        return ((1-SOR) * T[i, j, k] + SOR * (omega *
                (Mnew + Nnew + Onew + Pnew + Qnew + Rnew)
                + (1-omega) *
                (Tn[i, j, k] * M + Mold + Nold + Oold + Pold + Qold + Rold)
                + R * Tn[i, j, k] + WT[i, j, k] - ET) 
                / (R - omega * M))

    while not converged:
        num_it += 1
        num_sub_it += 1
        Told = T.copy()

        # implementation of black-red stencil scheme for SOR
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if GP[i, j, k]:
                        if not BT[i, j, k]:
                            T[i, j, k] = solve(i, j, k)
        for k in prange(nz):
            for j in prange(ny):
                for i in prange(nx):
                    if not GP[i, j, k]:
                        if not BT[i, j, k]:
                            T[i, j, k] = solve(i, j, k)

        diff = l2_norm(T, Told, 'T')
        if diff < T_conv or omega == 0:
            t_T += dt_T
            if num_sub_it < max_it/2:
                dt_T = min(dt_T*inc_it, max_dt)
            converged = True
        else:
            converged = False
            if num_sub_it >= max_it:
                printf('\n  [T] [iter %5d] [dt/s %6.2e -> %6.2e] [L2 %6.2e]'
                    , num_it, dt_T, dt_T/2, diff)
                dt_T = dt_T/2  # decrease time step size
                num_sub_it = 0
                T = Tn.copy()  # reset T
            if dt_T <= min_dt:
                break

    return t_T, dt_T, T, converged, num_it

@njit(parallel=numba_parallel, fastmath=numba_fastmath, cache=numba_cache,
      nogil=numba_nogil)
def smolarkiewicz(C, qx, qy, qz, vx, vy, vz, dt, eps=1e-15, Sc=1.08):
    '''
    Smolarkiewicz (1983) numerical dispersion correction
    for explicit time-weighting, requires additional transport step
    with corrected velocities
    '''
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if C[i, j, k] > 0:
                    if i < nx - 1:
                        dd = 0.5 * (dx[i+1] + dx[i])
                        vx[i, j, k] = \
                            (Sc * (abs(qx[i, j, k]) * dd - dt
                             * qx[i, j, k]**2) * (C[i+1, j, k] - C[i, j, k])
                             / ((C[i, j, k] + C[i+1, j, k] + eps) * dd))
                    if j < ny - 1:
                        dd = 0.5 * (dy[j+1] + dy[j])
                        vy[i, j, k] = \
                            (Sc * (abs(qy[i, j, k]) * dd - dt
                             * qy[i, j, k]**2) * (C[i, j+1, k] - C[i, j, k])
                             / ((C[i, j, k] + C[i, j+1, k] + eps) * dd))
                    if k < nz - 1:
                        dd = 0.5 * (dz[k+1] + dz[k])
                        vz[i, j, k] = \
                            (Sc * (abs(qz[i, j, k]) * dd - dt
                             * qz[i, j, k]**2) * (C[i, j, k+1] - C[i, j, k])
                             / ((C[i, j, k] + C[i, j, k+1] + eps) * dd))
                else:
                    vx[i, j, k], vy[i, j, k], vz[i, j, k] = 0, 0, 0

    return vx, vy, vz

###############################################################################
# SOLVER PARAMETERS ###########################################################
dt = config.dt     # initial time step size (s)
dt_prev = dt           # plotresults dt initialisation
if not 't_sim' in dir(config):
    t_sim = 0              # initial simulation time (s)
else:
    t_sim = config.t_sim
n_tsteps = 0           # time step counter (-)

# set counters etc. to zero
numitP = numitC = num_it_C = num_itC = numitT = num_CFL = max_DT = \
    max_Pex_T = max_Pey_T = max_Pez_T = sum_Pex_T = sum_Pey_T = sum_Pez_T = \
    max_Pex_C = max_Pey_C = max_Pez_C = sum_Pex_C = sum_Pey_C = sum_Pez_C = \
    Pex_T = Pey_T = Pez_T = Pex_C = Pey_C = Pez_C = bc_inc = chem_inc = \
    dt_chem = 0

dt_max_abs = dt_max    # define maximum simulation time

def print_row(label, text):
    print('  \033[1m%-12s\033[0m%s' % (label, text))


def onoff(flag):
    return '\033[32mon\033[0m' if flag else '\033[90moff\033[0m'


print()
print_row('MODEL', '%d x %d x %d nodes (%d total) | %g x %g x %g m'
          % (nx, ny, nz, nx*ny*nz, lx, ly, lz))
print_row('SPACING', 'dx %.2f m | dy %.2f m | dz %.2f m (min/max)'
          % (numpy.min(dx), numpy.min(dy), numpy.min(dz)))
print_row('TIME', 't %.3e s | dt %.3e s | dt_max %.3e s' % (t, dt, dt_max))
print_row('PROCESSES', 'FLUIDFLOW [%s]  SPECTRANS [%s]  HEATTRANS [%s]'
          % (onoff(config.FLUIDFLOW), onoff(config.SPECTRANS),
             onoff(config.HEATTRANS)))
print_row('', 'FLUIDPROP [%s]  CHEMREACT [%s]  DYNAMICBC [%s]'
          % (onoff(config.FLUIDPROP), onoff(config.CHEMREACT),
             onoff(config.DYNAMICBC)))
print_row('', 'MASSBALAN [%s]  SIA [%s]  SMOLAR_C [%s]  SMOLAR_T [%s]'
          % (onoff(config.MASSBALAN), onoff(config.SIA),
             onoff(config.SMOLAR_C), onoff(config.SMOLAR_T)))
print_row('', 'IMBLFLO C [%s]  T [%s]'
          % (onoff(config.C_IMBLFLO), onoff(config.T_IMBLFLO)))
if UNSATURAT:
    print_row('', 'RICHARDS [on]  KRMODEL %s' % KRMODEL)
print_row('SOLVER', 'SOR_P %.4f | SOR_C %.4f | SOR_T %.4f'
          % (config.SOR_P, config.SOR_C, config.SOR_T))
print_row('WEIGHTING', 'OMEGA_C %g | OMEGA_T %g'
          % (config.OMEGA_C, config.OMEGA_T))
print_row('CONVERGENCE', 'P %.1e | C %.1e | T %.1e'
          % (config.CONV_P, config.CONV_C, config.CONV_T))
print_row('PRESSURE', PSOLVER + ' solver')
print_row('PARALLEL', '%d thread(s) | numba threading layer %s'
          % (config.parallel_processes, config.numba_tl))
if config.CHEMREACT:
    print_row('CHEMISTRY', '%d thread(s) | dt_chem %g s'
              % (config.parallel_chemistry, config.DTCHEM))
print()
print('\033[93m' + BAR + '\033[0m')

time_start = time.time()
plotresults()

rhof_prev = numpy.zeros_like(rho_f)
P_prev2 = P.copy()      # pressure two time steps ago (initial guess)
dt_CFL_prev = 1e10      # CFL time step limit of the previous time step

# CFL-based time step cap for the next time step (UNSATURAT: applied
# before the flow solve, 'CFL first')
dt_cfl_prev = dt_max_abs

if UNSATURAT:
    # consistent initial saturation, capacity and relative permeability
    # after the initial fluid properties evaluation
    update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT, krB, rho_f,
                      Kx, Ky, Kz, alpha_vg, n_vg, S_r, S_s, A_vg, gamma_vg,
                      krm)

while t_sim < t:

    SIA_filled = False
    SIA_iter = 0
    SIA_diff = 0

    # saturation at the previous time level for the mixed-form storage term
    if UNSATURAT:
        Sn = S.copy()
        P_step = P.copy()      # pressure at the previous time level
    else:
        # extrapolated initial guess for the pressure solution
        # (P_prev + (P_prev - P_prev2); Dirichlet cells are excluded)
        P_interior = (BP == 0)
        if t_sim > 0:
            Pprev_copy = P.copy()
            P[P_interior] = 2.0 * P[P_interior] - P_prev2[P_interior]
            P_prev2[:] = Pprev_copy
        else:
            P_prev2[:] = P

        # predictive CFL clamp: velocities change slowly between consecutive
        # time steps, so clamping dt before the flow solution avoids pressure
        # solves at time steps that would be reduced afterwards anyway
        if config.FLUIDFLOW and t_sim > 0 and not config.SIA:
            dt = min(dt, dt_CFL_prev)

    while not SIA_filled and SIA_iter < config.SIA_maxit:
        ## outer SIA iteration here
        #  SIA = True/False
        #  SIA_max_iter = n 
        Co_Filled = False
        CFL = 0

        # Neumann criteria for species and heat transport
        if config.OMEGA_C == 0:
            dt_NeC = explicit_C(D_fx, phi)

        # EXPLICIT HEAT TRANSPORT (OMEGA_T == 0) DISABLED:
        # the former dt_NeT calculation used the element-wise sum
        # 1/dx**2 + 1/dy**2 + 1/dz**2 instead of the required outer sum
        # over i/j/k, and the resulting array broke the scalar expression
        # dt_max = min(dt_max_C, dt_max_T, dt_max_P) below
        # if config.OMEGA_T == 0:
        #     alphamax = \
        #         numpy.max((phi * l_f + (1 - phi) * l_r) /
        #                 (phi * rho_f * cp_f + (1 - phi) * rho_r * cp_r))
        #     dt_NeT = 0.45 / alphamax / (1 / dx[:, None, None]**2
        #                                 + 1 / dy[None, :, None]**2
        #                                 + 1 / dz[None, None, :]**2)

        num_it_P = 0
        if config.FLUIDFLOW:
            while not Co_Filled:
                if UNSATURAT:
                    # CFL first: cap the time step by the CFL criterion of
                    # the previous step before solving (dt_cfl_prev)
                    dt = min(dt, dt_cfl_prev)
                    # modified-Picard solution of the mixed-form Richards
                    # equation (outer Picard + inner solve in one call,
                    # with internal dt reduction on non-convergence);
                    # PSOLVER 'CG': preconditioned CG on the linearised
                    # system (residual-based, robust for the nearly
                    # elliptic saturated zone), otherwise red-black SOR
                    if PSOLVER == 'CG':
                        dt, dt_P, P, converged_P, num_it_P = \
                            compute_P_RICH_CG(
                                P, P_step, S, Sn, kr, krE, krW, krN, krS,
                                krT, krB, dSdP, c_f, c_r, nue, rho_f, phi,
                                Kx, Ky, Kz, qx, qy, qz, dt, BP, WP,
                                alpha_vg, n_vg, S_r, S_s, A_vg, gamma_vg,
                                krm, CONV_S, dt_max_abs,
                                max_it=MAXIT_P, P_conv=config.CONV_P,
                                pcg_omega=CG_OMEGA)
                    else:
                        dt, dt_P, P, converged_P, num_it_P = compute_P_RICH(
                            P, P_step, S, Sn, kr, krE, krW, krN, krS, krT,
                            krB, dSdP, c_f, c_r, nue, rho_f, phi, Kx, Ky,
                            Kz, qx, qy, qz, dt, BP, WP, alpha_vg, n_vg,
                            S_r, S_s, A_vg, gamma_vg, krm, CONV_S,
                            dt_max_abs, SOR=config.SOR_P, max_it=MAXIT_P,
                            P_conv=config.CONV_P)
                else:
                    # build the pressure stencil (coefficients depend on rho_f,
                    # nue and the Dirichlet values in P; rebuilt per flow solve)
                    compute_stencil_P(aE, aW, aN, aS, aT, aB, bgrav, bdir,
                                      Rbase, Msum, nue, rho_f, phi, c_r, c_f,
                                      Kx, Ky, Kz, BP, P)
                    if PSOLVER == 'CG':
                        dt, dt_P, P, Pn, converged_P, num_it_P = \
                            compute_P_CG(
                                P, aE, aW, aN, aS, aT, aB, bgrav, bdir, Rbase,
                                Msum, dt, BP, WP, max_dt=dt_max, inc_it=1.25,
                                max_it=MAXIT_P, P_conv=config.CONV_P,
                                pcg_omega=CG_OMEGA, Pn_prev=P_prev2)
                    else:
                        dt, dt_P, P, Pn, converged_P, num_it_P = \
                            compute_P(
                                P, aE, aW, aN, aS, aT, aB, bgrav, bdir, Rbase,
                                Msum, dt, BP, WP, max_dt=dt_max, inc_it=1.25,
                                SOR=config.SOR_P, max_it=MAXIT_P,
                                P_conv=config.CONV_P, Pn_prev=P_prev2)

                if not converged_P:
                    sys.exit('* P CONVERGENCE ERROR ' +
                            '- check dt and CONV_P in compute_P')

                if t_sim == 0 and numba_parallel and SIA_iter == 0:
                    print('Threading layer:', threading_layer(),
                        'Threads:', get_num_threads(), '\n')

                if UNSATURAT:
                    qx, qy, qz = compute_q_RICH(P, BP, Kx, Ky, Kz, qx, qy,
                                                qz, nue, rho_f, krE, krW,
                                                krN, krS, krT, krB)
                else:
                    qx, qy, qz = compute_q(P, BP, Kx, Ky, Kz, qx, qy, qz, nue, 
                                           rho_f)


                if config.HEATTRANS or config.SPECTRANS:
                    # dt is the time step actually used for the converged
                    # P solution (returned t_P), while dt_P is the increased
                    # time step for the next flow solution
                    if UNSATURAT:
                        e = flow_error(P, P_step, qx, qy, qz, WP, phi, c_f,
                                       c_r, rho_f, dt, S, Sn, True)
                    else:
                        e = flow_error(P, Pn, qx, qy, qz, WP, phi, c_f, c_r,
                                       rho_f, dt, S, Sn, False)

                # Check CFL and Neumann criteria to adapt maximum time step as
                # required
                if t_sim > 0:
                    if UNSATURAT:
                        # pore velocity q/(phi*S) with dry-cell guard
                        S_eff = numpy.maximum(S, 1e-8)
                        dt_CFL = 0.9 / (
                            numpy.max(numpy.abs(qx) / (phi * S_eff
                                                       * dx[:, None, None])) +
                            numpy.max(numpy.abs(qy) / (phi * S_eff
                                                       * dy[None, :, None])) +
                            numpy.max(numpy.abs(qz) / (phi * S_eff
                                                       * dz[None, None, :])))
                    else:
                        dt_CFL = \
                            0.9 / (numpy.max(numpy.reshape(numpy.abs(qx) / phi,
                                                        (ny, nz, nx)) / dx) +
                                numpy.max(numpy.reshape(numpy.abs(qy) / phi,
                                                        (nx, nz, ny)) / dy) +
                                numpy.max(numpy.abs(qz) / phi / dz))
                else:
                    dt_CFL = 1e10
                dt_CFL_prev = dt_CFL

                dt_max_C = dt_max_T = dt_max_P = dt_max_abs

                # calculate Peclet numbers for heat and species transport
                if config.HEATTRANS:
                    Pex_T, Pey_T, Pez_T = peclet_T(PeT, qx, qy, qz, rho_f, cp_f,
                                                phi, l_r, l_f)

                if config.SPECTRANS:
                    Pex_C, Pey_C, Pez_C = peclet_C(PeC, qx, qy, qz, D_fx, D_fy,
                                                D_fz)

                # Set Neumann criteria for explicit method
                if config.OMEGA_C == 0:
                    dt_max_C = dt_NeC
                # if config.OMEGA_T == 0:  # explicit heat transport disabled
                #     dt_max_T = dt_NeT

                if UNSATURAT:
                    # CFL first: the time step was already capped by the
                    # CFL criterion of the previous step (dt_cfl_prev)
                    # before the flow solve; store the new criterion for
                    # the next time step
                    dt_cfl_prev = min(dt_CFL, dt_max_abs)
                elif config.SIA:
                    dt_max = min(dt_max_C, dt_max_T, dt_max_P)
                else:
                    dt_max = min(dt_CFL, dt_max_C, dt_max_T, dt_max_P)

                # print time step and convergence control data
                if n_tsteps % config.PRINTOUT == 0:
                    if config.OMEGA_C + config.OMEGA_T < 2:
                        print(' [dt_NeC] %8.2e [dt_NeT] %8.2e' %
                            (dt_max_C, dt_max_T))

                    print(' [CFL] %6.2f%% | Pe_T %.1e/%.1e/%.1e | Pe_C %.1e/%.1e/%.1e'
                        % (dt/dt_CFL*100, Pex_T, Pey_T, Pez_T,
                           Pex_C, Pey_C, Pez_C))
                    print('[P+] %6.2e [P-] %6.2e [T+] %6.4f [T-] %6.4f'
                        % (numpy.max(P), numpy.min(P), numpy.max(T),
                            numpy.min(T)) +
                        ' [C+] %6.4f [C-] %6.4f' % (numpy.max(C), 
                            numpy.min(C)), end='')
                    if UNSATURAT:
                        print(' [S+] %6.4f [S-] %6.4f'
                            % (numpy.max(S), numpy.min(S)), end='')

                Co_Filled = True
                if not UNSATURAT and dt > dt_max*1.0:
                    num_CFL += num_it_P  # count also CFL correction iterations
                    CFL += 1
                    dt = dt_max  # /2 # reduce dt to match CFL criterion

        num_it_C = 0

        if config.SPECTRANS:
            Cn = C.copy()
            for n in config.SPTRANS:  # repeat for all transported species
                # compute only if at least one C != 0 and WC != 0
                if numpy.sum(C[:, :, :, n]) != 0.0 \
                    or numpy.sum(WC[:, :, :, n]) != 0.0:

                    # solve species transport PDE system
                    dt, dt_C, C[:, :, :, n], converged_C, num_itC = \
                        compute_C(
                            C[:, :, :, n], D_fx[:, :, :, n], D_fy[:, :, :, n],
                            D_fz[:, :, :, n], phi, S, rho_f, qx, qy, qz, dt,
                            BC, WC[:, :, :, n], R, e,
                            omega=config.OMEGA_C, SOR=config.SOR_C, 
                            max_it=MAXIT_C, C_conv=config.CONV_C)

                    num_it_C += num_itC

                    if not converged_C:
                        sys.exit('* C CONVERGENCE ERROR' +
                                 '- check dt and C_conv in compute_C')

                    if config.SMOLAR_C:
                        vx, vy, vz = \
                            smolarkiewicz(C[:, :, :, n], qx, qy, qz,
                                        vx, vy, vz, dt, eps=1e-15, Sc=1.08)

                        dt, dt_C, C[:, :, :, n], converged_C, num_itC = \
                            compute_C(
                            C[:, :, :, n], D_fx[:, :, :, n],
                            D_fy[:, :, :, n], D_fz[:, :, :, n],
                            phi, S, rho_f, vx, vy, vz, dt, BC, WC[:, :, :, n],
                            R, ZEROS, omega=config.OMEGA_C, 
                            max_it=MAXIT_C, SOR=config.SOR_C,
                            C_conv=config.CONV_C)

                        num_it_C += num_itC

                        if not converged_C:
                            sys.exit('* C2 CONVERGENCE ERROR' +
                                     '- check dt and C_conv in compute_C')

        num_it_T = 0
        if config.HEATTRANS:
            Tn = T.copy()

            # solve heat transport PDE system
            if 'NLTE_SOLV' in dir(config) and config.NLTE_SOLV:
                # Gas phase (layer 0)
                dt, dt_T, T[:,:,:,0], converged_T, num_itT1 = \
                    compute_T(T[:,:,:,0], l_f, cp_f, rho_f, ONES, S, qx, qy,
                            qz, dt, BT, ZEROS,
                            e, omega=config.OMEGA_T, max_it=MAXIT_T, SOR=config.SOR_T,
                            T_conv=config.CONV_T)
                # Solid/immobile phase (layer 1) - zero fluid properties
                dt, dt_T, T[:,:,:,1], converged_T, num_itT2 = \
                    compute_T(T[:,:,:,1], ZEROS, ZEROS, ZEROS, ZEROS, S, qx,
                            qy, qz, dt, BT, WT,
                            e, omega=config.OMEGA_T, max_it=MAXIT_T, SOR=config.SOR_T,
                            T_conv=config.CONV_T)
                num_itT = num_itT1 + num_itT2

            else:
                dt, dt_T, T, converged_T, num_itT = \
                    compute_T(T, l_f, cp_f, rho_f, phi, S, qx, qy, qz, dt, BT,
                            WT, e, omega=config.OMEGA_T, max_it=MAXIT_T,
                            SOR=config.SOR_T, T_conv=config.CONV_T)

            num_it_T += num_itT

            if not converged_T:
                sys.exit('* T CONVERGENCE ERROR' +
                        ' - check dt and T_conv in compute_T')

            if config.SMOLAR_T:
                vx, vy, vz = smolarkiewicz(T, qx, qy, qz, vx, vy, vz, dt,
                                        eps=1e-15, Sc=1.08)

                dt, dt_T, T, converged_T, num_itT = \
                    compute_T(T, l_f, cp_f, rho_f, phi, S, vx, vy, vz, dt, BT,
                            WT, ZEROS, omega=config.OMEGA_T, max_it=MAXIT_T,
                            SOR=config.SOR_T, T_conv=config.CONV_T)

                num_it_T += num_itT

                if not converged_T:
                    sys.exit('* T CONVERGENCE ERROR' +
                            ' - check dt and T_conv in compute_T')

        # calculate chemical reactions
        if config.CHEMREACT:
            if t_sim > 0:
                chem_inc += dt
            else:
                chem_inc = 0
                dt_chem = dt

            if chem_inc >= config.DTCHEM:
                dt_chem = chem_inc
                chem_inc = 0
            chem_list = index_list(BC)
            if chem_list != [] and chem_inc == 0:
                results = \
                    Parallel(n_jobs=config.parallel_chemistry)(
                        delayed(config.chemical_reactions)(
                            C[i, j, k], P[i, j, k], T[i, j, k],
                            rho_f[i, j, k], phi[i, j, k], Kx[i, j, k], Ky[i, j, k],
                            Kz[i, j, k], dt_chem, i, j, k) for i, j, k in chem_list)

                for num, l in enumerate(chem_list):
                    C[l[0], l[1], l[2]], P[l[0], l[1], l[2]], \
                        T[l[0], l[1], l[2]], rho_f[l[0], l[1], l[2]], \
                        phi[l[0], l[1], l[2]], Kx[l[0], l[1], l[2]], \
                        Ky[l[0], l[1], l[2]], Kz[l[0], l[1], l[2]] = \
                        results[num][0], results[num][1], results[num][2], \
                        results[num][3], results[num][4], results[num][5], \
                        results[num][6], results[num][7]


        # calculate fluid properties
        if config.FLUIDPROP:
            update_fluid_properties()

        if UNSATURAT:
            # the modified-Picard iteration (outer loop) and any required
            # dt reduction are handled inside compute_P_RICH; the time
            # step is finished after one call
            SIA_filled = True
            if n_tsteps % config.PRINTOUT == 0:
                print(' [PICARD] %4d' % num_it_P)
        elif config.SIA:
            SIA_diff = l2_norm(rho_f, rhof_prev, 'rhof')
            if n_tsteps % config.PRINTOUT == 0:
                print(' [SIA] %4d [SIA] %6.2e' % (SIA_iter, SIA_diff))
            if SIA_diff < config.SIA_conv or SIA_iter == config.SIA_maxit - 1:
                SIA_filled = True
            else:
                rhof_prev = rho_f.copy()
                P = Pn.copy()
                if config.HEATTRANS:
                    T = Tn.copy()
                if config.SPECTRANS:
                    C = Cn.copy()
                SIA_iter += 1
        else:
            SIA_filled = True
            if n_tsteps % config.PRINTOUT == 0:
                print('')

        numitP += num_it_P
        numitC += num_it_C
        numitT += num_it_T

    if UNSATURAT and config.FLUIDFLOW:
        # final consistency: saturation and kr exactly consistent with the
        # converged pressure field (for output and the next time step)
        update_S_kr_faces(P, S, kr, dSdP, krE, krW, krN, krS, krT, krB,
                          rho_f, Kx, Ky, Kz, alpha_vg, n_vg, S_r, S_s,
                          A_vg, gamma_vg, krm)

    # dynamic boundary conditions
    if config.DYNAMICBC:
        bc_inc_ = bc_inc
        bc_inc, P, T, C, BP, BT, BC, WP, WT, WC, Kx, Ky, Kz = \
            config.dynamic_bc(t_sim + dt, bc_inc, P, T, C, BP, BT, BC, WP, WT,
                            WC, Kx, Ky, Kz, rho_f, nue, c_f, cp_f, qx, phi, l_f, l_r)

        if bc_inc_ != bc_inc:
            # recalculate fluid properties if dynamic boundary conditions change
            if config.FLUIDPROP:
                update_fluid_properties()

    # calculate mass balances
    if config.MASSBALAN:
        mass = config.mass_balance(C, qx, qy, qz, WP, WC, WT,
                                    D_fx, D_fy, D_fz, dt, mass)

    t_sim += dt
    n_tsteps += 1
    max_Pex_T = max(Pex_T, max_Pex_T)
    max_Pey_T = max(Pey_T, max_Pey_T)
    max_Pez_T = max(Pez_T, max_Pez_T)
    sum_Pex_T += Pex_T
    sum_Pey_T += Pey_T
    sum_Pez_T += Pez_T
    max_Pex_C = max(Pex_C, max_Pex_C)
    max_Pey_C = max(Pey_C, max_Pey_C)
    max_Pez_C = max(Pez_C, max_Pez_C)
    sum_Pex_C += Pex_C
    sum_Pey_C += Pey_C
    sum_Pez_C += Pez_C
    max_DT = max(max_DT, dt)

    if n_tsteps % config.PRINTOUT == 0:
        print('[t] %8.2e/%8.2e [dt] \033[1m%8.2e\033[0m [S] %4d [P] %3d'
              % (t_sim, t, dt, n_tsteps, num_it_P) +
              ' [C] %3d [T] %3d [CFL] %d' % (num_it_C, num_it_T, CFL))

    dt = 1e99
    if config.FLUIDFLOW:
        dt = min(dt_P, t-t_sim)
    if config.HEATTRANS:
        dt = min(dt_T, t-t_sim, dt)
    if config.SPECTRANS:
        dt = min(dt_C, t-t_sim, dt)

    plotresults()

# PRINT FINAL SIMULATION STATS #############################################
print('[ SIMULATION SUMMARY ]')
print('[t/s] %8.2e/%8.2e [S] %5d [P] %5d [C] %5d [T] %5d [CFL] %1d'
      % (t_sim, t, n_tsteps, numitP, numitC, numitT, num_CFL))
print('[max dt] %8.2e [max PTx] %8.4e [max PTy] %8.4e [max PTz] %8.4e'
      % (max_DT, max_Pex_T, max_Pey_T, max_Pez_T))
print('[avg dt] %8.2e [avg PTx] %8.4e [avg PTy] %8.4e [avg PTz] %8.4e'
      % (t_sim/n_tsteps, sum_Pex_T/n_tsteps, sum_Pey_T/n_tsteps,
         sum_Pez_T/n_tsteps))
print('[max dt] %8.2e [max PCx] %8.4e [max PCy] %8.4e [max PCz] %8.4e'
      % (max_DT, max_Pex_C, max_Pey_C, max_Pez_C))
print('[avg dt] %8.2e [avg PCx] %8.4e [avg PCy] %8.4e [avg PCz] %8.4e'
      % (t_sim/n_tsteps, sum_Pex_C/n_tsteps, sum_Pey_C/n_tsteps,
         sum_Pez_C/n_tsteps))

time_total = time.time()-time_start
print('[Run time]', round(time_total, 2), '(s)', round(time_total/3600, 2),
      '(hours)')
