'''
    TRANSPORTSE configuration file
'''

import os
import sys
import numpy
import joblib
from numba import njit

# PARALLELISATION AND TUNING ###################################################
parallel_processes = 1   # number of threads
parallel_chemistry = 1   # number of threads
numba_fastmath = False   # turn off strict IEEE 754 compliance
numba_tl = 'omp'         # numba threading layer (tbb, omp, workqueue)

# MODEL DIMENSIONS AND SPATIAL DISCRETISATION ##################################
lx = 51                          # model length in x-direction (m)
ly = 51                          # model length in y-direction (m)
lz = 30                          # model length in z-direction (m)
nx = 51                          # number of nodes in x-direction (-)
ny = 51                          # number of nodes in y-direction (-)
nz = 1                           # number of nodes in z-direction (-)
dx = numpy.ones((nx)) * lx / nx  # nodal distance x-direction (m)
dy = numpy.ones((ny)) * ly / ny  # nodal distance y-direction (m)
dz = numpy.ones((nz)) * lz / nz  # nodal distance z-direction (m)

# PARAMETRISATION ##############################################################
g = 9.80665        # gravitational acceleration (m/s^2)
beta =  0          # model slope angle (°)

## SIMULATION TIME #############################################################
t = 86400 * 6      # simulation time (s)

## PROCESSES ###################################################################
FLUIDFLOW = True   # fluid flow (qx, qy must be given if set to False)
SPECTRANS = False  # species transport
HEATTRANS = True   # heat transport
FLUIDPROP = False  # fluid properties function
CHEMREACT = False  # chemical reactions
DYNAMICBC = False  # Use dynamic boundary conditions
MASSBALAN = False  # calculate mass balances
DTCHEM    = 1      # chemical reactions time stepping

## NUMBER OF SPECIES ###########################################################
IONS = 1			             # Species
MINERALS = 1                     # Minerals
                                 #
SPECIES = IONS + MINERALS        # total number of species
SPTRANS = numpy.array([0])       # list of species to be transported

# MATRIX INITIALISATION ########################################################
## ROCK ########################################################################
phi   = numpy.zeros((nx, ny, nz))         # porosity (-)
Kx    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
Ky    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
Kz    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
c_r   = numpy.ones((nx, ny, nz)) * 1e-99  # compressiblity (1/Pa)
cp_r  = numpy.zeros((nx, ny, nz))         # heat capacity (J/kg/K)
l_r   = numpy.zeros((nx, ny, nz))         # heat conductivity (W/m/K)
rho_r = numpy.zeros((nx, ny, nz))         # average density (kg/m^3)
l_l   = numpy.zeros((nx, ny, nz))         # longitudinal dispersion length (m)
l_t   = numpy.zeros((nx, ny, nz))         # transversal dispersion length (m)
L     = numpy.zeros((nx, ny, nz))         # lithology marker (-)
R     = numpy.ones((nx, ny, nz))          # retardation coefficient (-)
mass  = numpy.zeros((3))                  # mass balance matrix

## FLUID #######################################################################
D_fx  = numpy.zeros((nx, ny, nz))         # diffusion coefficient (m^2/s)
D_fy  = numpy.zeros((nx, ny, nz))         # diffusion coefficient (m^2/s)
D_fz  = numpy.zeros((nx, ny, nz))         # diffusion coefficient (m^2/s)
c_f   = numpy.ones((nx, ny, nz)) * 1e-99  # compressiblity (1/Pa)
cp_f  = numpy.zeros((nx, ny, nz))         # heat capacity (J/kg/K)
l_f   = numpy.zeros((nx, ny, nz))         # heat conductivity (W/m/K)
rho_f = numpy.zeros((nx, ny, nz))         # density (kg/m^3)
nue   = numpy.zeros((nx, ny, nz))         # viscosity (Pa*s)

## PRIMARY VARIABLES ###########################################################
P = numpy.zeros((nx, ny, nz))           # pore pressure (kg/m/s^2 = Pa)
C = numpy.zeros((nx, ny, nz, SPECIES))  # concentration and mineral amount
                                        # (unit arbitrary, e.g. mol/kgw)
T = numpy.zeros((nx, ny, nz))           # temperature (°C)

## BOUNDARY CONDITIONS #########################################################
BP = numpy.zeros((nx, ny, nz))   # flow boundary conditions
BC = numpy.zeros((nx, ny, nz))   # species boundary conditions
BT = numpy.zeros((nx, ny, nz))   # thermal boundary conditions

## SINK AND SOURCE TERMS #######################################################
WP = numpy.zeros((nx, ny, nz))   # fluid flow (m^3/s)
WT = numpy.zeros((nx, ny, nz))   # heat flow  (W/m^3)
WC = numpy.zeros((nx, ny, nz))   # species flow (kg/m^3/s)

## FLOW VELOCITIES #############################################################
qx = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qy = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qz = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)

# PARAMETERS ###################################################################
# define injection well location
wx = int(nx/2)
wy = int(ny/2)

phi   += 0.15
Kx    += 5.0e-13
Ky    += 5.0e-13
cp_r  += 1000.0
cp_f  += 4150.0
l_r   += 2.0
l_f   += 0.6
nue   += 1e-3
rho_f += 1e3
rho_r += 2500
D_fx  += 5.0e-9
D_fy  = D_fx.copy()
l_l   += 0.5
l_t   += 0.5

# BOUNDARY CONDITIONS ##########################################################

## FLUID FLOW
BP[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
BP[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
BP[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
BP[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)

# Injection rate at injection well (50 m^3/h)
WP[wx,wy,:] = 50 / 3600 / dx[wx] / dy[wy] / dz[0]

## HEAT TRANSPORT
BT[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
BT[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
BT[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
BT[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)
BT[wx,wy,:] = 1      # Constant temperature at well

# INITIAL CONDITIONS ###########################################################
P[:,:,:] = 1.5e7             # Initial reservoir pressure
T[:,:,:] = 100               # Initial reservoir temperature
T[wx,wy,:] = 20              # Temperature at injection well

# USER FLUID PROPERTY FUNCTION #################################################
@njit
def fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi,
                     Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r):
    '''
        nue     viscosity (Pa*s)
        rho_f   density (kg/m^3)
        c_f     isothermal compressibility (1/Pa)
        cp_f    heat capacity (J/kg/K)
        l_f     thermal conductivity (W/m/K)
        D_f     diffusion coefficient (m^2/s)
        ...
    '''

    return C, T , P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, Kz, \
            D_fx, D_fy, D_fz, rho_r
 
#===============================================================================
# USER CHEMICAL REACTIONS FUNCTION #############################################

# LOAD PHREEQC module
if CHEMREACT:
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    if sys.platform == 'linux': # fix phreeqc_dll.py os.platform entry
        print("Fixing phreeqpy module...")
        os.system('sed -i s/linux2/linux/g '+str(phreeqc_mod).split("'")[3])

    phreeqc = phreeqc_mod.IPhreeqc()
    phreeqc.load_database(r"phreeqc.dat")

def chemical_reactions(i,j,k):

    return C[i,j,k], rho_f[i,j,k], phi[i,j,k], Kx[i,j,k], Ky[i,j,k], Kz[i,j,k]

#===============================================================================

# SOLVER PARAMETERS ############################################################
## TIME STEP SETTINGS ##########################################################
dt =     .001  # initial time step size
dt_max = 86400

## IMPLICIT OR EXPLICIT SOLUTION (0 <= OMEGA <= 1)
OMEGA_C = 1
OMEGA_T = 1

## SUCCESSIVE OVER RELAXATION (0 <= SOR <= 2, SOR = 1 -> GAUSS-SEIDEL/JACOBI)
## SOR < 0: UNDERRELAXATION (SLOWER CONVERGENCE, MAY BE USEFUL FOR TRANSPORT)
## SOR > 1: OVERRELAXATION (FASTER CONVERGENCE)
## OPTIMISED SOR:
## SOR_P = 2/(1 + NUMPY.PI/NX) # NX SHOULD BE THE NUMBER OF ACTIVE FLOW ELEMENTS
SOR_P = 2/(1+ numpy.pi/nx)
SOR_C = 1.0
SOR_T = 1.0

## CONVERGENCE CRITERION FOR L2-NORM ###########################################
CONV_P = 1e-5
CONV_C = 1e-5
CONV_T = 1e-5

## SMOLARKIEWICZ ANTI-DIFFUSION STEP ###########################################
SMOLAR_C = False
SMOLAR_T = False

# OUTPUT PARAMETERS ############################################################
PRINTOUT = 1        # output interval
PLOTSAVE = True     # save results
PLOTTIME = numpy.linspace(0,1,11) * t
