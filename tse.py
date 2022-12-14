# Licence: GNU General Public License, Version 3, 29 June 2007
# Copyright © 2022 Helmholtz Centre Potsdam GFZ German Research Centre for
# Geosciences, Potsdam, Germany
# Citation: Kempka, T. (2020): Verification of a Python-based TRANSPORT
# Simulation Environment for density-driven fluid flow and coupled transport of
# heat and chemical species. Advances in Geosciences, 54, 67-77.
# https://doi.org/10.5194/adgeo-54-67-2020

'''
    tse.py
    See config.py for input parameter documentation
'''

# IMPORT LIBRARIES AND CONFIG FILE #############################################
import os
os.environ['NUMBA_THREADING_LAYER']='tbb'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['BLAS_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'

import sys
import time
import numpy
import joblib
from multiprocessing import Pool, cpu_count

# check if config file name is provided
if len(sys.argv) < 2:
    print('\n\tPlease use this command to run TRANSPORTSE:\n' + \
         '\033[1m\tpython -i [path]tse.py [path]<config file name>\033[0m' + \
         '\n\n\t\033[1mExamples:\033[0m' + \
         '\n\tpython -i tse.py config.py' + \
         '\n\tpython -i tse.py simulation/config.py' + \
         '\n\tpython -i tse.py ~/tse/simulation/config.py\n'
            )
    sys.exit()

# import config file defined as first argument
cfile = sys.argv[1].split('/')[-1].split('.py')[0]
cdir = sys.argv[1].split(cfile+'.py')[0]
sys.path.append(cdir)
config = __import__(cfile)

# CHEMISTRY ####################################################################
if config.CHEMREACT:
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod

# MANAGE PARALLELISATION #######################################################
parallel_processes = config.parallel_processes
numba_fastmath = config.numba_fastmath
numba_tl = config.numba_tl
numba_cache = False
numba_nogil = False

if parallel_processes < 2:
    numba_parallel = False
else:
    numba_parallel = True

from numba import njit, prange, threading_layer, set_num_threads, get_num_threads
from numba import config as config_numba

if numba_parallel:
    config_numba.THREADING_LAYER = numba_tl
    set_num_threads(parallel_processes)

# PRINT GREAT BANNER ###########################################################
print('\n\033[93m+------------------------------------------+\033[0m')
print('\n\033[92m\033[1m'\
+'░▀█▀░█▀▄░█▀█░█▀█░█▀▀░█▀█░█▀█░█▀▄░▀█▀░█▀▀░█▀▀\n' \
+'░░█░░█▀▄░█▀█░█░█░▀▀█░█▀▀░█░█░█▀▄░░█░░▀▀█░█▀▀\n' \
+'░░▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀\n' \
+'\033[0m\n' \
+'\033[93m+---| TRANSPORT Simulation Environment |---+' \
+'\n\033[0m')

# MODEL DIMENSIONS AND SPATIAL DISCRETISATION ##################################
lx = config.lx
ly = config.ly
lz = config.lz
nx = config.nx
ny = config.ny
nz = config.nz
dx = config.dx
dy = config.dy
dz = config.dz

## GENERAL SETTINGS ############################################################
g = config.g                       # gravitational acceleration (m/s^2)
beta = numpy.radians(config.beta)  # model slope angle (rad)

# SIMULATION TIME ##############################################################
t = config.t
dt_max = config.dt_max       # maximum time step size (s)

# PLOTTING #####################################################################
PLOTINDEX = 0                # required for time series plots

# MATRIX INITIALISATION ########################################################
## ROCK ########################################################################
phi   = config.phi           # porosity (-)
Kx    = config.Kx            # permeability (m^2)
Ky    = config.Ky            # permeability (m^2)
Kz    = config.Kz            # permeability (m^2)
c_r   = config.c_r           # compressiblity (1/Pa)
cp_r  = config.cp_r          # heat capacity (J/kg/K)
l_r   = config.l_r           # heat conductivity (W/m/K)
rho_r = config.rho_r         # density (kg/m^3)
l_l   = config.l_l           # longitudinal dispersion length (m)
l_t   = config.l_t           # transversal dispersion length (m)
L     = config.L             # lithology (-)
R     = config.R             # retardation coefficient (-)

## FLUID #######################################################################
D_fx  = config.D_fx          # diffusion coefficient (W/m/K)
D_fy  = config.D_fy          # diffusion coefficient (W/m/K)
D_fz  = config.D_fz          # diffusion coefficient (W/m/K)
c_f   = config.c_f           # compressiblity (1/Pa)
cp_f  = config.cp_f          # heat capacity (J/kg/K)
l_f   = config.l_f           # heat conductivity (W/m/K)
rho_f = config.rho_f         # density (kg/m^3)
nue   = config.nue           # viscosity (Pa*s)

## PRIMARY VARIABLES ###########################################################
P = config.P                 # pore pressure (kg/m/s^2 = Pa)
C = config.C                 # concentration (kg/m^3 bulk volume)
T = config.T                 # temperature (°C)

## BOUNDARY CONDITIONS #########################################################
BP = config.BP               # flow boundary conditions
BC = config.BC               # species boundary conditions
BT = config.BT               # thermal boundary conditions

## SINK AND SOURCE TERMS #######################################################
WP = config.WP               # fluid flow (kg/s)
WT = config.WT               # heat flow  (W/m^3)
WC = config.WC               # species flow (kg/m^3/s)

## FLOW VELOCITIES #############################################################
qx = config.qx               # fluid flow velocity (m/s)
qy = config.qy               # fluid flow velocity (m/s)
qz = config.qz               # fluid flow velocity (m/s)

## MASS BALANCE DATA ###########################################################
mass = config.mass           # mass balance vector

## SMOLARKIEWICZ FLOW VELOCITIES ###############################################
vx = numpy.zeros((nx,ny,nz)) # Smolarkiewicz velocity (m/s)
vy = numpy.zeros((nx,ny,nz)) # Smolarkiewicz velocity (m/s)
vz = numpy.zeros((nx,ny,nz)) # Smolarkiewicz velocity (m/s)

## PECLET MATRICES #############################################################
PeC = numpy.zeros((nx,ny,nz,3)) # Smolarkiewicz velocity (m/s)
PeT = numpy.zeros((nx,ny,nz,3)) # Smolarkiewicz velocity (m/s)

## SOR GRID PARTITIONING #######################################################
GP = numpy.zeros((nx,ny,nz), dtype=bool)
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if (i+j+k) % 2 == 0:
                GP[i,j,k] = True

# CALCULATE INITIAL FLUIDPROPERTIES ############################################
@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def run_fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi,
                        Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r):
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                C[i,j,k], T[i,j,k], P[i,j,k], nue[i,j,k], rho_f[i,j,k], \
                    c_f[i,j,k], cp_f[i,j,k], l_f[i,j,k], cp_r[i,j,k], \
                    l_r[i,j,k], phi[i,j,k], Kx[i,j,k], Ky[i,j,k], \
                    Kz[i,j,k], D_fx[i,j,k], D_fy[i,j,k], D_fz[i,j,k], \
                    rho_r[i,j,k] = \
                    config.fluid_properties(C[i,j,k], T[i,j,k],
                        P[i,j,k], nue[i,j,k], rho_f[i,j,k], c_f[i,j,k],
                        cp_f[i,j,k], l_f[i,j,k], cp_r[i,j,k], l_r[i,j,k],
                        phi[i,j,k], Kx[i,j,k], Ky[i,j,k], Kz[i,j,k],
                        D_fx[i,j,k], D_fy[i,j,k], D_fz[i,j,k], rho_r[i,j,k])

    return C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, Kz, \
            D_fx, D_fy, D_fz, rho_r

if config.FLUIDPROP:
    C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, Kz, \
        D_fx, D_fy, D_fz, rho_r = \
        run_fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r,
                            phi, Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r)

# SUPPORT FUNCTIONS ############################################################

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def alpha(i1, j1, k1, i2, j2, k2, nue, rho_f, K, q, d, d1, d2, n):
    '''
        alpha = k * rho_f / (nue * di)
        with rho_f upwinded
        i1 > i2, j1 > j2
    '''
    a = 0
    if q[i2, j2, k2] < 0:
        a = 1
    return (harm(i1, j1, k1, i2, j2, k2, K, n) /
            harm(i1, j1, k1, i2, j2, k2, nue, n) /
            (d[d2] * (d[d1]+d[d2])*0.5)
            * (a * rho_f[i1,j1,k1] + (1-a) * rho_f[i2,j2,k2]))

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def beta_g(i1, j1, k1, i2, j2, k2, rho_f, d):
    # slope in x/z direction can be applied via angle beta (0-1)
    if d == 0:
        return amean(i1, j1, k1, i2, j2, k2, rho_f, 0) * g * \
                0.5 * (dx[i1] + dx[i2]) * numpy.sin(beta)
                #dx[i1] * numpy.sin(beta)
    else:
        return amean(i1, j1, k1, i2, j2, k2, rho_f, 2) * g * \
                0.5 * (dz[k1] + dz[k2]) * numpy.cos(beta)
                #dz[k1] * numpy.cos(beta)

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def l2_norm(New, Old, Process):
    l2 = numpy.sqrt(numpy.sum((New - Old)**2)) / \
           numpy.sqrt(numpy.sum(Old**2)+1e-100)
    if numpy.isnan(l2):
        print('ISNAN:', Process)
        return 1
    else:
        return l2

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def P_diff(New, Old):
    return numpy.max(numpy.abs(New-Old))

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def amean(i1, j1, k1, i2, j2, k2, X, n):
    if n == 0:
        return (X[i1,j1,k1] * dx[i1] + X[i2,j2,k2] * dx[i2]) / (dx[i1] + dx[i2])
    elif n == 1:
        return (X[i1,j1,k1] * dy[j1] + X[i2,j2,k2] * dy[j2]) / (dy[j1] + dy[j2])
    else:
        return (X[i1,j1,k1] * dz[k1] + X[i2,j2,k2] * dz[k2]) / (dz[k1] + dz[k2])

@njit(fastmath=numba_fastmath,cache=numba_cache,nogil=numba_nogil)
def harm(i1, j1, k1, i2, j2, k2, H, n):
    if H[i1,j1,k1] != 0 and H[i2,j2,k2] != 0:
        if n == 0:
            return ((dx[i1] + dx[i2]) /
                    (dx[i1] / H[i1,j1,k1] + dx[i2] / H[i2,j2,k2]))
        elif n == 1:
            return ((dy[j1] + dy[j2]) /
                    (dy[j1] / H[i1,j1,k1] + dy[j2] / H[i2,j2,k2]))
        else:
            return ((dz[k1] + dz[k2]) /
                    (dz[k1] / H[i1,j1,k1] + dz[k2] / H[i2,j2,k2]))
    else:
        return 0


def plotresults():
    # write results into pickle file at selected time steps
    global PLOTINDEX, dt
    if config.PLOTSAVE:
        if PLOTINDEX <= numpy.shape(config.PLOTTIME)[0] - 1:
            t_plot = config.PLOTTIME[PLOTINDEX]
            if t_sim < t_plot:
                dt = min(dt, t_plot-t_sim)
            elif t_sim == t_plot:
                with open("{:s}-{:05d}.p".format(cdir+cfile, PLOTINDEX), "wb") \
                    as pfile:
                    joblib.dump( (lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt,
                                  P, C, T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f,
                                  nue, cp_f, l_f, BP, BC, BT, L, mass,
                                  WP, WT), pfile,
                                  compress=3)
                PLOTINDEX += 1

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def peclet_C(PeC, qx, qy, qz, D_fx):
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if BC[i,j,k] == 0:
                    if D_fx[i,j,k] != 0:
                        PeC[i,j,k,0] = abs(qx[i,j,k]) / D_fx[i,j,k] * dx[i]
                        PeC[i,j,k,1] = abs(qy[i,j,k]) / D_fx[i,j,k] * dy[j]
                        PeC[i,j,k,2] = abs(qz[i,j,k]) / D_fx[i,j,k] * dz[k]
                    else:
                        PeC[i,j,k,0] = numpy.nan
                        PeC[i,j,k,1] = numpy.nan
                        PeC[i,j,k,2] = numpy.nan

    return numpy.max(PeC[:,:,:,0]), numpy.max(PeC[:,:,:,1]), \
            numpy.max(PeC[:,:,:,2])

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def peclet_T(PeT, qx, qy, qz, rho_f, cp_f, phi, l_r, l_f):
    for k in prange(nz):
        for j in prange(ny):
            for i in prange(nx):
                if BT[i,j,k] == 0:
                    PeT[i,j,k,0] = (abs(qx[i,j,k]) * rho_f[i,j,k] * cp_f[i,j,k]
                        / ((1-phi[i,j,k]) * l_r[i,j,k] + phi[i,j,k]
                        * l_f[i,j,k])*dx[i])
                    PeT[i,j,k,1] = (abs(qy[i,j,k]) * rho_f[i,j,k] * cp_f[i,j,k]
                        / ((1-phi[i,j,k]) * l_r[i,j,k] + phi[i,j,k]
                        * l_f[i,j,k])*dy[j])
                    PeT[i,j,k,2] = (abs(qz[i,j,k]) * rho_f[i,j,k] * cp_f[i,j,k]
                        / ((1-phi[i,j,k]) * l_r[i,j,k] + phi[i,j,k]
                        * l_f[i,j,k])*dz[k])

    return numpy.max(PeT[:,:,:,0]), numpy.max(PeT[:,:,:,1]), \
            numpy.max(PeT[:,:,:,2])

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache, \
      nogil=numba_nogil)
def explicit_C(D_fx, phi):
    dt_NeC = 1e100
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if L[i,j,k] != 0 and L[i,j,k] != 6:
                    dt_NeC = min(dt_NeC,
                        0.45 * D_fx[i,j,k] / phi[i,j,k] /
                        (1 / dx[i]**2 + 1 / dy[j]**2 + 1/dz[k]**2))

    return dt_NeC

# determine elements for chemical calculation
@njit(parallel=False, fastmath=numba_fastmath,cache=numba_cache)
def index_list(BC):
    chem_list = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if BC[i,j,k] != 1:
                    chem_list.append((i, j, k))
    return chem_list

# SOLVER ROUTINES ##############################################################

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def compute_P(P, c_f, nue, rho_f, phi, Kx, Ky, Kz, qx, qy, qz, dt_P, BP, WP,
              SOR=1.0, max_it=20000, inc_it=1.05, max_dt=dt_max,
              min_dt=1e-8, P_conv=1e-10):
    t_P = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    Pn = P.copy()

    def solve(i,j,k):
        A_ = B_ = C_ = D_ = E_ = F_ = G_ = H_ = I_ = J_ = K_ = L_ = \
        Anew = Cnew = Enew = Gnew = Inew = Knew = 0

        if i != nx-1:
            A_ = alpha(i+1, j, k, i, j, k, nue, rho_f, Kx, qx, dx, i+1, i, 0)
            Anew = A_*P[i+1, j, k]
            B_ = beta_g(i+1, j, k, i, j, k, rho_f, 0)
        if i != 0:
            C_ = alpha(i, j, k, i-1, j, k, nue, rho_f, Kx, qx, dx, i-1, i, 0)
            Cnew = C_*P[i-1, j, k]
            D_ = beta_g(i, j, k, i-1, j, k, rho_f, 0)
        if j != ny-1:
            E_ = alpha(i, j+1, k, i, j, k, nue, rho_f, Ky, qy, dy, j+1, j, 1)
            Enew = E_*P[i, j+1, k]
            F_ = 0
        if j != 0:
            G_ = alpha(i, j, k, i, j-1, k, nue, rho_f, Ky, qy, dy, j-1, j, 1)
            Gnew = G_*P[i, j-1, k]
            H_ = 0
        if k != nz-1:
            I_ = alpha(i, j, k+1, i, j, k, nue, rho_f, Kz, qz, dz, k+1, k, 2)
            Inew = I_*P[i, j, k+1]
            J_ = beta_g(i, j, k+1, i, j, k, rho_f, 2)
        if k != 0:
            K_ = alpha(i, j, k, i, j, k-1, nue, rho_f, Kz, qz, dz, k-1, k, 2)
            Knew = K_*P[i, j, k-1]
            L_ = beta_g(i, j, k, i, j, k-1, rho_f, 2)

        por = phi[i,j,k]
        R = rho_f[i,j,k] * ((1-por) * c_r[i,j,k] + por * c_f[i,j,k]) / dt_P
        M = A_ + C_ + E_ + G_ + I_ + K_

        return (1-SOR) * P[i,j,k] + SOR * (
            Anew + A_*B_ + Cnew - C_*D_ + Enew + E_*F_
             + Gnew - G_*H_ + Inew + I_*J_ + Knew - K_*L_
            + R*Pn[i,j,k] + WP[i,j,k] * rho_f[i,j,k]) / (R + M)

    while not converged:
        num_it += 1
        num_sub_it += 1
        Pold = P.copy()

        if numba_parallel is False: # sequential SOR
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if not BP[i,j,k]:
                            P[i,j,k] = solve(i,j,k)
        else: # implementation of black-red stencil scheme for SOR
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if GP[i,j,k]:
                            if not BP[i,j,k]:
                                P[i,j,k] = solve(i,j,k)
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if not GP[i,j,k]:
                            if not BP[i,j,k]:
                                P[i,j,k] = solve(i,j,k)

        if num_it < 50:
            diff = l2_norm(P, Pold, 'P')
        else:
            if num_it % 10 == 0: # l2 norm after each 10 iterations
                diff = l2_norm(P, Pold, 'P')
        if diff < P_conv:
            converged = True
            t_P += dt_P
            # Increase time step by factor inc_it, while not exceeding max_dt
            if num_sub_it < max_it/2:
                dt_P = min(dt_P*inc_it, max_dt)
        else:
            converged = False
            if num_sub_it >= max_it or (diff >= 1 and num_sub_it > 1):
                print(' t:', t_P, 'P [', num_it, ']: Reducing dt:', dt_P, \
                      '=>', dt_P/2, '(s) L2:', diff)
                dt_P = dt_P/2 # decrease time step size
                num_sub_it = 0
                P = Pn.copy() # reset P
            if dt_P <= min_dt:
                break

    return t_P, dt_P, P, Pn, converged, num_it

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache, \
      nogil=numba_nogil)
def compute_q(P, Kx, Ky, Kz, qx, qy, qz, nue, rho_f):
    for k in prange(nz): # iteration only for interfaces between grid points
        for j in prange(ny):
            for i in prange(nx):
                if i < nx - 1:
                    qx[i, j, k] = (- harm(i+1, j, k, i, j, k, Kx, 0) /
                                harm(i+1, j, k,  i, j, k, nue, 0)
                                * ((P[i+1, j, k] -P[i, j, k]) /
                                  (0.5 * (dx[i+1] + dx[i]))
                                + amean(i+1, j, k, i, j, k, rho_f, 0) * g *
                                numpy.sin(beta)))
                if j < ny - 1:
                    qy[i, j, k] = (- harm(i, j+1, k, i, j, k, Ky, 1) /
                                harm(i, j+1, k, i, j, k, nue, 1)
                                * (P[i, j+1, k] - P[i, j, k]) /
                                  (0.5 * (dy[j+1] + dy[j])))
                if k < nz - 1:
                    qz[i, j, k] = (- harm(i, j, k+1, i, j, k, Kz, 2) /
                                harm(i, j, k+1, i, j, k, nue, 2)
                                * ((P[i, j, k+1]-P[i, j, k]) /
                                  (0.5 * (dz[k+1] + dz[k]))
                                + amean(i, j, k+1, i, j, k, rho_f, 2) * g *
                                numpy.cos(beta)))
    return qx, qy, qz

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def compute_C(C, D_fx, D_fy, D_fz, phi, rho_f, qx, qy, qz, dt_C, BC, WC, RC,
        omega=1, SOR=1.0, max_it=20000, inc_it=1.00, max_dt=dt_max, min_dt=1e-8,
        C_conv=1e-10, upwind=1):

    t_C = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    Cn = C.copy()

    def solve(i,j,k):
        A_ = B_ = C_ = D_ =  E_ = F_ = G_ = H_ = \
        I_ = J_ = K_ = L_ = M_ = N_ = O_ = P_ = Q_ = R_ = \
        Mnew = Nnew = Onew = Pnew = Qnew = Rnew = \
        Mold = Nold = Oold = Pold = Qold = Rold = \
        D_xr = D_yr = D_zr = D_xl = D_yl = D_zl = 0

        # compute dispersion tensor if any dispersivity term is non-zero
        if l_l[i,j,k] != 0 or l_t[i,j,k] != 0:
            qxl = qyl = qzl = qxr = qyr = qzr = \
            qxyl = qxzl = qyxl = qyzl = qzxl = qzyl = \
            qxyr = qxzr = qyxr = qyzr = qzxr = qzyr = 0

            # calculate longitudinal and transversal velocity terms
            if i != 0: # left face
                qxl = qx[i-1,j,k] # longitudinal term

                # X-Y plane transversal term (clockwise)
                q1 = qy[i-1,j,k]
                q2 = qy[i,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k]
                    q4 = qy[i-1,j-1,k]
                    c += 2
                if not BC[i-1,j,k]:
                    qxyl = (q1 + q2 + q3 + q4) / c

                # X-Z plane transversal term (clockwise)
                q1 = qz[i-1,j,k]
                q2 = qz[i,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j,k-1]
                    q4 = qz[i-1,j,k-1]
                    c += 2
                if not BC[i-1,j,k]:
                    qxzl = (q1 + q2 + q3 + q4) / c

            if i != nx-1: # right face
                qxr = qx[i,j,k] # longitudinal term

                # X-Y plane transversal term (clockwise)
                q1 = qy[i,j,k]
                q2 = qy[i+1,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i+1,j-1,k]
                    q4 = qy[i,j-1,k]
                    c += 2
                if not BC[i+1,j,k]:
                    qxyr = (q1 + q2 + q3 + q4) / c

                # X-Z plane transversal term (clockwise)
                q1 = qz[i,j,k]
                q2 = qz[i+1,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i+1,j,k-1]
                    q4 = qz[i,j,k-1]
                    c += 2
                if not BC[i+1,j,k]:
                    qxzr = (q1 + q2 + q3 + q4) / c

            if j != 0: # left face
                qyl = qy[i,j-1,k] # longitudinal term

                # Y-X plane transversal term (clockwise)
                q1 = qx[i,j,k]
                q2 = qx[i,j-1,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j-1,k]
                    q4 = qx[i-1,j,k]
                    c += 2
                if not BC[i,j-1,k]:
                    qyxl = (q1 + q2 + q3 + q4) / c

                # Y-Z plane transversal term (clockwise)
                q1 = qz[i,j-1,k]
                q2 = qz[i,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j,k-1]
                    q4 = qz[i,j-1,k-1]
                    c += 2
                if not BC[i,j-1,k]:
                    qyzl = (q1 + q2 + q3 + q4) / c

            if j != ny-1: # right face
                qyr = qy[i,j,k] # longitudinal term

                # Y-X plane transversal term (clockwise)
                q1 = qx[i,j+1,k]
                q2 = qx[i,j,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k]
                    q4 = qx[i-1,j+1,k]
                    c += 2
                if not BC[i,j+1,k]:
                    qyxr = (q1 + q2 + q3 + q4) / c

                # Y-Z plane transversal term (clockwise)
                q1 = qz[i,j,k]
                q2 = qz[i,j+1,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j+1,k-1]
                    q4 = qz[i,j,k-1]
                    c += 2
                if not BC[i,j+1,k]:
                    qyzr = (q1 + q2 + q3 + q4) / c

            if k != 0: # bottom face
                qzl = qz[i,j,k-1] # longitudinal term

                # Z-X plane transversal term (clockwise)
                q1 = qx[i,j,k]
                q2 = qx[i,j,k-1]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k-1]
                    q4 = qx[i-1,j,k]
                    c += 2
                if not BC[i,j,k-1]:
                    qzxl = (q1 + q2 + q3 + q4) / c

                # Z-Y plane transversal term (clockwise)
                q1 = qy[i,j,k]
                q2 = qy[i,j,k-1]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k-1]
                    q4 = qy[i,j-1,k]
                    c += 2
                if not BC[i,j,k-1]:
                    qzyl = (q1 + q2 + q3 + q4) / c

            if k != nz-1: # top face
                qzr = qz[i,j,k] # longitudinal term

                # Z-X plane transversal term (clockwise)
                q1 = qx[i,j,k+1]
                q2 = qx[i,j,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k]
                    q4 = qx[i-1,j,k+1]
                    c += 2
                if not BC[i,j,k+1]:
                    qzxr = (q1 + q2 + q3 + q4) / c

                # Z-Y plane transversal term (clockwise)
                q1 = qy[i,j,k+1]
                q2 = qy[i,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k]
                    q4 = qy[i,j-1,k+1]
                    c += 2
                if not BC[i,j,k+1]:
                    qzyr = (q1 + q2 + q3 + q4) / c

            D_xl = l_l[i,j,k] * abs(qxl) + l_t[i,j,k] * (abs(qxyl) + abs(qxzl))
            D_yl = l_l[i,j,k] * abs(qyl) + l_t[i,j,k] * (abs(qyxl) + abs(qyzl))
            D_zl = l_l[i,j,k] * abs(qzl) + l_t[i,j,k] * (abs(qzxl) + abs(qzyl))
            D_xr = l_l[i,j,k] * abs(qxr) + l_t[i,j,k] * (abs(qxyr) + abs(qxzr))
            D_yr = l_l[i,j,k] * abs(qyr) + l_t[i,j,k] * (abs(qyxr) + abs(qyzr))
            D_zr = l_l[i,j,k] * abs(qzr) + l_t[i,j,k] * (abs(qzxr) + abs(qzyr))

        if i != nx-1:
            dd = dx[i]
            dd2 = (0.5 * (dx[i+1] + dx[i])) * dd
            ph = amean(i+1, j, k, i, j, k, phi, 0)
            re = harm(i+1, j, k, i, j, k, RC, 0)
            A_ = (harm(i+1, j, k, i, j, k, D_fx, 0) * ph + D_xr ) / dd2 / re
            G_ = max(qx[i,j,k], 0) / dd / re
            H_ = min(qx[i,j,k], 0) / dd / re
            if upwind == 0:
                G_ = 0.5 * qx[i,j,k] / dd / re
                H_ = 0.5 * qx[i,j,k] / dd / re
            Mnew = (A_ - H_) * C[i+1,j,k]
            Mold = (A_ - H_) * Cn[i+1,j,k]
        if i != 0:
            dd = dx[i]
            dd2 = (0.5 * (dx[i-1] + dx[i])) * dd
            ph = amean(i-1, j, k, i, j, k, phi, 0)
            re = harm(i-1, j, k, i, j, k, RC, 0)
            B_ = (harm(i-1, j, k, i, j, k, D_fx, 0) * ph + D_xl ) / dd2 / re
            I_ = max(qx[i-1,j,k], 0) / dd / re
            J_ = min(qx[i-1,j,k], 0) / dd / re
            if upwind == 0:
                I_ = 0.5 * qx[i-1,j,k] / dd / re
                J_ = 0.5 * qx[i-1,j,k] / dd / re
            Nnew = (B_ + I_) * C[i-1,j,k]
            Nold = (B_ + I_) * Cn[i-1,j,k]
        if j != ny-1:
            dd = dy[j]
            dd2 = (0.5 * (dy[j+1] + dy[j])) * dd
            ph = amean(i, j+1, k, i, j, k, phi, 1)
            re = harm(i, j+1, k, i, j, k, RC, 1)
            C_ = (harm(i, j+1, k, i, j, k, D_fy, 1) * ph + D_yr ) / dd2 / re
            K_ = max(qy[i,j,k], 0) / dd / re
            L_ = min(qy[i,j,k], 0) / dd / re
            if upwind == 0:
                K_ = 0.5 * qy[i, j,k] / dd / re
                L_ = 0.5 * qy[i, j,k] / dd / re
            Onew = (C_ - L_) * C[i,j+1,k]
            Oold = (C_ - L_) * Cn[i,j+1,k]
        if j != 0:
            dd = dy[j]
            dd2 = (0.5 * (dy[j-1] + dy[j])) * dd
            ph = amean(i, j-1, k, i, j, k, phi, 1)
            re = harm(i, j-1, k, i, j, k, RC, 1)
            D_ = (harm(i, j-1, k, i, j, k, D_fy, 1) * ph + D_yl ) / dd2 / re
            M_ = max(qy[i,j-1,k], 0) / dd / re
            N_ = min(qy[i,j-1,k], 0) / dd / re
            if upwind == 0:
                M_ = 0.5 * qy[i,j-1,k] / dd / re
                N_ = 0.5 * qy[i,j-1,k] / dd / re
            Pnew = (D_ + M_) * C[i,j-1,k]
            Pold = (D_ + M_) * Cn[i,j-1,k]
        if k != nz-1:
            dd = dz[k]
            dd2 = (0.5 * (dz[k+1] + dz[k])) * dd
            ph = amean(i, j, k+1, i, j, k, phi, 2)
            re = harm(i, j, k+1, i, j, k, RC, 2)
            E_ = (harm(i, j, k+1, i, j, k, D_fz, 2) * ph + D_zr ) / dd2 / re
            O_ = max(qz[i,j,k], 0) / dd / re
            P_ = min(qz[i,j,k], 0) / dd / re
            if upwind == 0:
                O_ = 0.5 * qz[i, j,k] / dd / re
                P_ = 0.5 * qz[i, j,k] / dd / re
            Qnew = (E_ - P_) * C[i,j,k+1]
            Qold = (E_ - P_) * Cn[i,j,k+1]
        if k != 0:
            dd = dz[k]
            dd2 = (0.5 * (dz[k-1] + dz[k])) * dd
            ph = amean(i, j, k-1, i, j, k, phi, 2)
            re = harm(i, j, k-1, i, j, k, RC, 2)
            F_ = (harm(i, j, k-1, i, j, k, D_fz, 2) * ph + D_zl ) / dd2 / re
            Q_ = max(qz[i,j,k-1], 0) / dd / re
            R_ = min(qz[i,j,k-1], 0) / dd / re
            if upwind == 0:
                Q_ = 0.5 * qz[i,j,k-1] / dd / re
                R_ = 0.5 * qz[i,j,k-1] / dd / re
            Rnew = (F_ + Q_) * C[i,j,k-1]
            Rold = (F_ + Q_) * Cn[i,j,k-1]
        R = phi[i,j,k] / dt_C
        M = - A_ - B_ - C_ - D_ - E_ - F_ - G_ + J_ - K_ + N_ - O_ + R_

        return ((1-SOR) * C[i,j,k] + SOR * ((omega *
                    (Mnew + Nnew + Onew + Pnew + Qnew + Rnew))
                    + (1-omega) * (Cn[i,j,k] * M
                    + Mold + Nold + Oold + Pold + Qold + Rold)
                    + R * Cn[i,j,k] + WC[i,j,k]) / (R - omega * M))

    while not converged:
        num_it += 1
        num_sub_it += 1
        Cold = C.copy()

        if numba_parallel is False: # sequential SOR
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if not BC[i,j,k]:
                            C[i,j,k] = solve(i,j,k)
        else: # implementation of black-red stencil scheme for SOR
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if GP[i,j,k]:
                            if not BC[i,j,k]:
                                C[i,j,k] = solve(i,j,k)
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if not GP[i,j,k]:
                            if not BC[i,j,k]:
                                C[i,j,k] = solve(i,j,k)

        diff = l2_norm(C, Cold, 'C')
        if diff < C_conv or omega == 0:
            converged = True
            t_C += dt_C
        else:
            converged = False
            if num_sub_it >= max_it:
                print(' t:', t_C, 'C [', num_it, ']: Reducing dt:', dt_C,
                        '=>', dt_C/2, '(s) L2:', diff)
                dt_C = dt_C/2 # decrease time step size
                num_sub_it = 0
                C = Cn.copy() # reset C
            if dt_C <= min_dt:
                break

    return t_C, dt_C, C, converged, num_it

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,
      nogil=numba_nogil)
def compute_T(T, l_f, cp_f, rho_f, phi, qx, qy, qz, dt_T, BT, WT, omega=1,
              SOR=1.0, max_it=100, inc_it=1.05, max_dt=dt_max, min_dt=1e-8,
              T_conv=1e-10, upwind=1):
    t_T = 0
    converged = False
    num_it = 0
    num_sub_it = 0
    Tn = T.copy()

    def solve(i,j,k):
        A_ = B_ = C_ = D_ = E_ = F_ = G_ = H_ = \
        I_ = J_ = K_ = L_ = M_ = N_ = O_ = P_ = Q_ = R_ = S_ = \
        Mnew = Nnew = Onew = Pnew = Qnew = Rnew =  \
        Mold = Nold = Oold = Pold = Qold = Rold = \
        D_xr = D_yr = D_zr = D_xl = D_yl = D_zl = 0

        # compute dispersion tensor if any dispersivity term is non-zero
        if l_l[i,j,k] != 0 or l_t[i,j,k] != 0:
            qxl = qyl = qzl = qxr = qyr = qzr = \
            qxyl = qxzl = qyxl = qyzl = qzxl = qzyl = \
            qxyr = qxzr = qyxr = qyzr = qzxr = qzyr = 0

            # calculate longitudinal and transversal velocity terms
            if i != 0: # left face
                qxl = qx[i-1,j,k] # longitudinal term

                # X-Y plane transversal term (clockwise)
                q1 = qy[i-1,j,k]
                q2 = qy[i,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k]
                    q4 = qy[i-1,j-1,k]
                    c += 2
                if not BT[i-1,j,k]:
                    qxyl = (q1 + q2 + q3 + q4) / c

                # X-Z plane transversal term (clockwise)
                q1 = qz[i-1,j,k]
                q2 = qz[i,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j,k-1]
                    q4 = qz[i-1,j,k-1]
                    c += 2
                if not BT[i-1,j,k]:
                    qxzl = (q1 + q2 + q3 + q4) / c

            if i != nx-1: # right face
                qxr = qx[i,j,k] # longitudinal term

                # X-Y plane transversal term (clockwise)
                q1 = qy[i,j,k]
                q2 = qy[i+1,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i+1,j-1,k]
                    q4 = qy[i,j-1,k]
                    c += 2
                if not BT[i+1,j,k]:
                    qxyr = (q1 + q2 + q3 + q4) / c

                # X-Z plane transversal term (clockwise)
                q1 = qz[i,j,k]
                q2 = qz[i+1,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i+1,j,k-1]
                    q4 = qz[i,j,k-1]
                    c += 2
                if not BT[i+1,j,k]:
                    qxzr = (q1 + q2 + q3 + q4) / c

            if j != 0: # left face
                qyl = qy[i,j-1,k] # longitudinal term

                # Y-X plane transversal term (clockwise)
                q1 = qx[i,j,k]
                q2 = qx[i,j-1,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j-1,k]
                    q4 = qx[i-1,j,k]
                    c += 2
                if not BT[i,j-1,k]:
                    qyxl = (q1 + q2 + q3 + q4) / c

                # Y-Z plane transversal term (clockwise)
                q1 = qz[i,j-1,k]
                q2 = qz[i,j,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j,k-1]
                    q4 = qz[i,j-1,k-1]
                    c += 2
                if not BT[i,j-1,k]:
                    qyzl = (q1 + q2 + q3 + q4) / c

            if j != ny-1: # right face
                qyr = qy[i,j,k] # longitudinal term

                # Y-X plane transversal term (clockwise)
                q1 = qx[i,j+1,k]
                q2 = qx[i,j,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k]
                    q4 = qx[i-1,j+1,k]
                    c += 2
                if not BT[i,j+1,k]:
                    qyxr = (q1 + q2 + q3 + q4) / c

                # Y-Z plane transversal term (clockwise)
                q1 = qz[i,j,k]
                q2 = qz[i,j+1,k]
                q3 = q4 = 0
                c = 2
                if k != 0:
                    q3 = qz[i,j+1,k-1]
                    q4 = qz[i,j,k-1]
                    c += 2
                if not BT[i,j+1,k]:
                    qyzr = (q1 + q2 + q3 + q4) / c

            if k != 0: # bottom face
                qzl = qz[i,j,k-1] # longitudinal term

                # Z-X plane transversal term (clockwise)
                q1 = qx[i,j,k]
                q2 = qx[i,j,k-1]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k-1]
                    q4 = qx[i-1,j,k]
                    c += 2
                if not BT[i,j,k-1]:
                    qzxl = (q1 + q2 + q3 + q4) / c

                # Z-Y plane transversal term (clockwise)
                q1 = qy[i,j,k]
                q2 = qy[i,j,k-1]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k-1]
                    q4 = qy[i,j-1,k]
                    c += 2
                if not BT[i,j,k-1]:
                    qzyl = (q1 + q2 + q3 + q4) / c

            if k != nz-1: # top face
                qzr = qz[i,j,k] # longitudinal term

                # Z-X plane transversal term (clockwise)
                q1 = qx[i,j,k+1]
                q2 = qx[i,j,k]
                q3 = q4 = 0
                c = 2
                if i != 0:
                    q3 = qx[i-1,j,k]
                    q4 = qx[i-1,j,k+1]
                    c += 2
                if not BT[i,j,k+1]:
                    qzxr = (q1 + q2 + q3 + q4) / c

                # Z-Y plane transversal term (clockwise)
                q1 = qy[i,j,k+1]
                q2 = qy[i,j,k]
                q3 = q4 = 0
                c = 2
                if j != 0:
                    q3 = qy[i,j-1,k]
                    q4 = qy[i,j-1,k+1]
                    c += 2
                if not BT[i,j,k+1]:
                    qzyr = (q1 + q2 + q3 + q4) / c

            D_xl = l_l[i,j,k] * abs(qxl) + l_t[i,j,k] * (abs(qxyl) + abs(qxzl))
            D_yl = l_l[i,j,k] * abs(qyl) + l_t[i,j,k] * (abs(qyxl) + abs(qyzl))
            D_zl = l_l[i,j,k] * abs(qzl) + l_t[i,j,k] * (abs(qzxl) + abs(qzyl))
            D_xr = l_l[i,j,k] * abs(qxr) + l_t[i,j,k] * (abs(qxyr) + abs(qxzr))
            D_yr = l_l[i,j,k] * abs(qyr) + l_t[i,j,k] * (abs(qyxr) + abs(qyzr))
            D_zr = l_l[i,j,k] * abs(qzr) + l_t[i,j,k] * (abs(qzxr) + abs(qzyr))

        if i != nx-1:
            dd = dx[i]
            dd2 = (0.5 * (dx[i+1] + dx[i])) * dd
            por = amean(i, j, k, i+1, j, k, phi, 0)
            lf = harm(i, j, k, i+1, j, k, l_f, 0)
            lr = harm(i, j, k, i+1, j, k, l_r, 0)
            cpf = harm(i, j, k, i+1, j, k, cp_f, 0)
            rhocpf = harm(i, j, k, i+1, j, k, rho_f, 0) * cpf
            A_ = ((1-por) * lr + por * lf + D_xr * rhocpf) / dd2
            S_ = rhocpf / dd
            G_ = max(qx[i,j,k], 0) * S_
            H_ = min(qx[i,j,k], 0) * S_
            if upwind == 0:
                G_ = 0.5 * qx[i,j,k] * S_
                H_ = 0.5 * qx[i,j,k] * S_
            Mnew = (A_ - H_) * T[i+1,j,k]
            Mold = (A_ - H_) * Tn[i+1,j,k]
        if i != 0:
            dd = dx[i]
            dd2 = (0.5 * (dx[i-1] + dx[i])) * dd
            por = amean(i, j, k, i-1, j, k, phi, 0)
            lf = harm(i, j, k, i-1, j, k, l_f, 0)
            lr = harm(i, j, k, i-1, j, k, l_r, 0)
            cpf = harm(i, j, k, i-1, j, k, cp_f, 0)
            rhocpf = harm(i, j, k, i-1, j, k, rho_f, 0) * cpf
            B_ = ((1-por) * lr + por * lf + D_xl * rhocpf) / dd2
            S_ = rhocpf / dd
            I_ = max(qx[i-1,j, k], 0) * S_
            J_ = min(qx[i-1,j, k], 0) * S_
            if upwind == 0:
                I_ = 0.5 * qx[i-1,j,k] * S_
                J_ = 0.5 * qx[i-1,j,k] * S_
            Nnew = (B_ + I_) * T[i-1,j,k]
            Nold = (B_ + I_) * Tn[i-1,j,k]
        if j != ny-1:
            dd = dy[j]
            dd2 = (0.5 * (dy[j+1] + dy[j])) * dd
            por = amean(i, j+1, k,  i, j, k, phi, 1)
            lf = harm(i, j+1, k,  i, j, k, l_f, 1)
            lr = harm(i, j+1, k,  i, j, k, l_r, 1)
            cpf = harm(i, j+1, k, i, j, k, cp_f, 1)
            rhocpf = harm(i, j+1, k, i, j, k, rho_f, 1) * cpf
            C_ = ((1-por) * lr + por * lf + D_yr * rhocpf) / dd2
            S_ = rhocpf / dd
            K_ = max(qy[i,j,k], 0) * S_
            L_ = min(qy[i,j,k], 0) * S_
            if upwind == 0:
                K_ = 0.5 * qy[i,j,k] * S_
                L_ = 0.5 * qy[i,j,k] * S_
            Onew = (C_ - L_) * T[i,j+1,k]
            Oold = (C_ - L_) * Tn[i,j+1,k]
        if j != 0:
            dd = dy[j]
            dd2 = (0.5 * (dy[j-1] + dy[j])) * dd
            por = amean(i, j-1, k, i, j, k, phi, 1)
            lf = harm(i, j-1, k, i, j, k, l_f, 1)
            lr = harm(i, j-1, k, i, j, k, l_r, 1)
            cpf = harm(i, j-1, k, i, j, k, cp_f, 1)
            rhocpf = harm(i, j-1, k, i, j, k, rho_f, 1) * cpf
            D_ = ((1-por) * lr + por * lf + D_yl * rhocpf) / dd2
            S_ = rhocpf / dd
            M_ = max(qy[i,j-1,k], 0) * S_
            N_ = min(qy[i,j-1,k], 0) * S_
            if upwind == 0:
                M_ = 0.5 * qy[i,j-1,k] * S_
                N_ = 0.5 * qy[i,j-1,k] * S_
            Pnew = (D_ + M_) * T[i, j-1,k]
            Pold = (D_ + M_) * Tn[i, j-1,k]
        if k != nz-1:
            dd = dz[k]
            dd2 = (0.5 * (dz[k+1] + dz[k])) * dd
            por = amean(i, j, k+1,  i, j, k, phi, 2)
            lf = harm(i, j, k+1,  i, j, k, l_f, 2)
            lr = harm(i, j, k+1,  i, j, k, l_r, 2)
            cpf = harm(i, j, k+1, i, j, k, cp_f, 2)
            rhocpf = harm(i, j, k+1, i, j, k, rho_f, 2) * cpf
            E_ = ((1-por) * lr + por * lf + D_zr * rhocpf) / dd2
            S_ =  rhocpf / dd
            O_ = max(qz[i,j,k], 0) * S_
            P_ = min(qz[i,j,k], 0) * S_
            if upwind == 0:
                O_ = 0.5 * qz[i,j,k] * S_
                P_ = 0.5 * qz[i,j,k] * S_
            Qnew = (E_ - P_) * T[i,j,k+1]
            Qold = (E_ - P_) * Tn[i,j,k+1]
        if k != 0:
            dd = dz[k]
            dd2 = (0.5 * (dz[k-1] + dz[k])) * dd
            por = amean(i, j, k-1, i, j, k, phi, 2)
            lf = harm(i, j, k-1, i, j, k, l_f, 2)
            lr = harm(i, j, k-1, i, j, k, l_r, 2)
            cpf = harm(i, j, k-1, i, j, k, cp_f, 2)
            rhocpf = harm(i, j, k-1, i, j, k, rho_f, 2) * cpf
            F_ = ((1-por) * lr + por * lf + D_zl * rhocpf) / dd2
            S_ = rhocpf / dd
            Q_ = max(qz[i,j,k-1], 0) * S_
            R_ = min(qz[i,j,k-1], 0) * S_
            if upwind == 0:
                Q_ = 0.5 * qz[i,j,k-1] * S_
                R_ = 0.5 * qz[i,j,k-1] * S_
            Rnew = (F_ + Q_) * T[i,j,k-1]
            Rold = (F_ + Q_) * Tn[i,j,k-1]
        por_ = phi[i,j,k]
        R = (por_ * rho_f[i,j,k] * cp_f[i,j,k] +
              (1-por_) * rho_r[i,j,k] * cp_r[i,j,k]) / dt_T
        M = - A_ - B_ - C_ - D_ - E_ - F_ - G_ + J_ - K_ + N_ - O_ + R_

        return (1-SOR) * T[i,j,k] + SOR * (omega *
                (Mnew + Nnew + Onew + Pnew + Qnew + Rnew)
                + (1-omega) * (Tn[i,j,k] * M
                + Mold + Nold + Oold + Pold + Qold + Rold)
                + R * Tn[i,j,k] + WT[i,j,k]) / (R - omega * M)

    while not converged:
        num_it += 1
        num_sub_it += 1
        Told = T.copy()

        if numba_parallel is False: # sequential SOR
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if not BT[i,j,k]:
                            T[i,j,k] = solve(i,j,k)
        else: # implementation of black-red stencil scheme for SOR
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if GP[i,j,k]:
                            if not BT[i,j,k]:
                                T[i,j,k] = solve(i,j,k)
            for k in prange(nz):
                for j in prange(ny):
                    for i in prange(nx):
                        if not GP[i,j,k]:
                            if not BT[i,j,k]:
                                T[i,j,k] = solve(i,j,k)

        diff = l2_norm(T, Told, 'T')
        if diff < T_conv or omega == 0:
            converged = True
            t_T += dt_T
            if num_sub_it < max_it/2:
                dt_T = min(dt_T*inc_it, max_dt)
        else:
            converged = False
            if num_sub_it >= max_it:
                print(' t:', t_T, 'T [', num_it, ']: Reducing dt:', dt_T, \
                      '=>', dt_T/2, '(s) L2:', diff)
                dt_T = dt_T/2 # decrease time step size
                num_sub_it = 0
                T = Tn.copy() # reset T
            if dt_T <= min_dt:
                break

    return t_T, dt_T, T, converged, num_it

@njit(parallel=numba_parallel, fastmath=numba_fastmath,cache=numba_cache,\
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
                if C[i,j,k] > 0:
                    if i < nx - 1:
                        dd = 0.5 * (dx[i+1] + dx[i])
                        vx[i,j,k] = (Sc * (abs(qx[i,j,k]) * dd - dt
                            * qx[i,j,k]**2) * (C[i+1,j,k] - C[i,j,k])
                            / ((C[i,j,k] + C[i+1,j,k] + eps) * dd))
                    if j < ny - 1:
                        dd = 0.5 * (dy[j+1] + dy[j])
                        vy[i,j,k] = (Sc * (abs(qy[i,j,k]) * dd - dt
                            * qy[i,j,k]**2) * (C[i,j+1,k] - C[i,j,k])
                            / ((C[i,j,k] + C[i,j+1,k] + eps) * dd))
                    if k < nz - 1:
                        dd = 0.5 * (dz[k+1] + dz[k])
                        vz[i,j,k] = (Sc * (abs(qz[i,j,k]) * dd - dt
                            * qz[i,j,k]**2) * (C[i,j,k+1] - C[i,j,k])
                            / ((C[i,j,k] + C[i,j,k+1] + eps) * dd))
                else:
                    vx[i,j,k], vy[i,j,k], vz[i,j,k] = 0, 0, 0

    return vx, vy, vz

################################################################################
# SOLVER PARAMETERS ############################################################
t      = config.t      # simulation time (s)
dt     = config.dt     # initial time step size (s)
t_sim = 0              # initial simulation time (s)
n_tsteps = 0           # time step counter (-)

# set counters etc. to zero
numitP = numitC = num_it_C = num_itC = numitT = num_CFL = max_DT = \
max_Pex_T = max_Pey_T = max_Pez_T = sum_Pex_T = sum_Pey_T = sum_Pez_T = \
max_Pex_C = max_Pey_C = max_Pez_C = sum_Pex_C = sum_Pey_C = sum_Pez_C = \
Pex_T = Pey_T = Pez_T = Pex_C = Pey_C = Pez_C = bc_inc = chem_inc = 0

dt_max_abs = dt_max    # define maximum simulation time

print('\033[1mFLUIDFLOW:\033[0m', config.FLUIDFLOW,
      '\033[1mSPECTRANS:\033[0m', config.SPECTRANS,
      '\033[1mHEATRANS\033[0m:', config.HEATTRANS,
      '\033[1mFLUIDPROP\033[0m:', config.FLUIDPROP)
print('lx: %.2f m ly: %.2f m lz: %.2f m\nnx: %d ny: %d nz: %d\n'
      % (lx, ly, lz, nx, ny, nz) + \
      'min dx: %.2f m dy %.2f m dz %.2f m\n' \
      % (numpy.min(dx), numpy.min(dy), numpy.min(dz)) + \
      'max dx: %.2f m dy %.2f m dz %.2f m' \
      % (numpy.max(dx), numpy.max(dy), numpy.max(dz)))
print('\033[1mOMEGA_C:\033[0m', config.OMEGA_C,
      '\033[1mOMEGA_T:\033[0m', config.OMEGA_T)
print('\033[1mSOR_P:\033[0m', round(config.SOR_P,4),
      '\033[1mSOR_C:\033[0m', round(config.SOR_C,4),
      '\033[1mSOR_T:\033[0m', round(config.SOR_T,4), '\n')

time_start = time.time()
plotresults()
while t_sim < t:
    Co_Filled = False
    CFL = 0

    # Neumann criteria for species and heat transport
    if config.OMEGA_C == 0:
        dt_NeC = explicit_C(D_fx, phi)
        print('dt_NeC',dt_NeC )

    if config.OMEGA_T == 0:
        alphamax = numpy.max((phi * l_f + (1 - phi) * l_r) /
                (phi * rho_f * cp_f + (1 - phi) * rho_r * cp_r))
        dt_NeT = 0.45 / alphamax / (1 / dx**2 + 1 / dy**2 + 1 / dz**2)

    num_it_P = 0
    if config.FLUIDFLOW:
        while not Co_Filled:
            dt, dt_P, P, Pn, converged_P, num_it_P = compute_P(P, c_f,
                nue, rho_f, phi, Kx, Ky, Kz, qx, qy, qz, dt, BP, WP,
                max_dt=dt_max, inc_it=1.25, SOR=config.SOR_P,
                max_it=20000, P_conv=config.CONV_P)

            if not converged_P:
                sys.exit('* P CONVERGENCE ERROR ' +
                         '- check dt and p_conv in compute_p')

            if t_sim == 0 and numba_parallel:
                print('Threading layer:', threading_layer(), \
                      'Threads:', get_num_threads(), '\n')

            qx, qy, qz  = compute_q(P, Kx, Ky, Kz, qx, qy, qz, nue, rho_f)

            # Check CFL and Neumann criteria to adapt maximum time step as
            # required
            if t_sim > 0:
                dt_CFL = 0.9/(numpy.max(numpy.reshape(numpy.abs(qx)/phi, \
                            (ny,nz,nx)) /dx) + \
                            numpy.max(numpy.reshape(numpy.abs(qy)/phi, \
                            (nx,nz,ny)) / dy) + \
                            numpy.max(numpy.abs(qz)/phi/dz))
            else:
                dt_CFL = 1e10

            dt_max_C = dt_max_T = dt_max_P = dt_max_abs

            # calculate Peclet numbers for heat and species transport
            if config.HEATTRANS:
                Pex_T, Pey_T, Pez_T = peclet_T(PeC, qx, qy, qz, rho_f, cp_f,
                        phi, l_r, l_f)

            if config.SPECTRANS:
                Pex_C, Pey_C, Pez_C =  peclet_C(PeT, qx, qy, qz, D_fx)

            # Set Neumann criteria for explicit method
            if config.OMEGA_C == 0:
                dt_max_C = dt_NeC
            if config.OMEGA_T == 0:
                dt_max_T = dt_NeT

            dt_max = min(dt_CFL, dt_max_C, dt_max_T, dt_max_P)

            # print time step and convergence control data
            if n_tsteps % config.PRINTOUT == 0:
                if config.OMEGA_C + config.OMEGA_T < 2:
                    print(' [dt_NeC] %8.2e [dt_NeT] %8.2e' %
                          ( dt_max_C, dt_max_T) )

                print(' [%%CFL] %6.2f [PTx] %6.4f [PTy] %6.4f [PTz] %6.4f'
                      % (dt/dt_CFL*100, Pex_T, Pey_T, Pez_T) +
                      ' [PCx] %6.4f [PCy] %6.4f [PCz] %6.4f' % (Pex_C, Pey_C,
                                                                Pez_C))
                print(' [P+] %6.2e [P-] %6.2e [T+] %6.4f [T-] %6.4f [C+] %6.4f'
                      % (numpy.max(P), numpy.min(P), numpy.max(T), numpy.min(T),
                         numpy.max(C)) +
                      ' [C-] %6.4f' % (numpy.min(C)))

            Co_Filled = True
            if dt > dt_max*1.0:
                numitP += num_it_P # count also CFL correction iterations
                num_CFL += num_it_P
                CFL += 1
                dt = dt_max #/2 # reduce dt to match Co-Kriterium
                #P = Pn.copy() # reset P to previous time step

    num_it_C = 0

    if config.SPECTRANS:
        for n in config.SPTRANS: # repeat for all transported species
            # compute only if at least one C != 0 and WC != 0
            if numpy.sum(C[:,:,:,n]) != 0.0 or numpy.sum(WC[:,:,:]) != 0.0:
                dt, dt_C, C[:,:,:,n], converged_C, num_itC = compute_C(
                    C[:,:,:,n], D_fx, D_fy, D_fz, phi, rho_f, qx, qy, qz, dt,
                    BC, WC, R, omega=config.OMEGA_C, SOR=config.SOR_C,
                    C_conv=config.CONV_C, upwind=1)

                num_it_C += num_itC

                if not converged_C:
                    sys.exit('* C CONVERGENCE ERROR' +
                            '- check dt and C_conv in compute_C', n)

                if config.SMOLAR_C:
                    vx, vy, vz = smolarkiewicz(C[:,:,:,n], qx, qy, qz,
                                           vx, vy, vz, dt, eps=1e-15, Sc=1.08)

                    dt, dt_C, C[:,:,:,n], converged_C, num_itC = compute_C(
                        C[:,:,:,n], D_fx, D_fy, D_fz, phi, rho_f, vx, vy, vz,
                        dt, BC, WC, R, omega=config.OMEGA_C, SOR=config.SOR_C,
                        C_conv=config.CONV_C)

                    num_it_C += num_itC

                    if not converged_C:
                        sys.exit('* C2 CONVERGENCE ERROR'+
                            '- check dt and C_conv in compute_C', n)

    num_it_T = 0
    if config.HEATTRANS:
        dt, dt_T, T, converged_T, num_itT = compute_T(T, l_f, cp_f,
                rho_f, phi, qx, qy, qz, dt, BT, WT,
                omega=config.OMEGA_T, SOR=config.SOR_T,
                T_conv=config.CONV_T)

        num_it_T += num_itT

        if not converged_T:
            sys.exit('* T CONVERGENCE ERROR' +
                     ' - check dt and T_conv in compute_T')

        if config.SMOLAR_T:
            vx, vy, vz  = smolarkiewicz(T, qx, qy, qz, vx, vy, vz, dt,
                                        eps=1e-15, Sc=1.08)

            dt, dt_T, T, converged_T, num_itT = compute_T(T, l_f,
                    cp_f, rho_f, phi, vx, vy, vz, dt, BT, WT,
                    omega=config.OMEGA_T, SOR=config.SOR_T,
                    T_conv=config.CONV_T)

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
            config.dt_chem = dt

        if chem_inc >= config.DTCHEM:
            config.dt_chem = chem_inc
            chem_inc = 0
        chem_list = index_list(BC)
        if chem_list != [] and chem_inc == 0:
            with Pool(config.parallel_chemistry) as p:
                results = p.starmap(config.chemical_reactions, chem_list)
            for num, l in enumerate(chem_list):
                C[l[0],l[1],l[2]], rho_f[l[0],l[1],l[2]], \
                phi[l[0],l[1],l[2]], Kx[l[0],l[1],l[2]], \
                Ky[l[0],l[1],l[2]], Kz[l[0],l[1],l[2]] = \
                        results[num][0], results[num][1], \
                        results[num][2], results[num][3], \
                        results[num][4], results[num][5]

    # dynamic boundary conditions
    if config.DYNAMICBC:
        bc_inc, P, T, C, BP, BT, BC, WP, WT, WC = \
            config.dynamic_bc(t_sim+dt, bc_inc, P, T, C, BP, BT, BC, WP, WT, WC)

    # calculate mass balances
    if config.MASSBALAN:
        mass = config.mass_balance(C, qx, qy, qz, dt, mass)

    # calculate fluid properties
    if config.FLUIDPROP:
        C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, \
            Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r = \
            run_fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r,
                                phi, Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r)

    t_sim += dt
    n_tsteps += 1
    numitP += num_it_P
    numitC += num_it_C
    numitT += num_it_T
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
        print('[t/s] %8.2e/%8.2e [dt/s] \033[1m%8.2e\033[0m [S] %8d [P] %5d'
          % (t_sim, t, dt, n_tsteps, num_it_P) +
          ' [C] %5d [T] %5d [CFL] %1d' % (num_it_C, num_it_T, CFL))

    if config.FLUIDFLOW:
        dt = min(dt_P, t-t_sim)
    elif config.HEATTRANS:
        dt = min(dt_T, t-t_sim)
    elif config.SPECTRANS:
        dt = min(dt_C, t-t_sim)

    plotresults()

# PRINT FINAL SIMULATION STATS #################################################
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
