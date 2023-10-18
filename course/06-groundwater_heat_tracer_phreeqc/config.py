'''
    TRANSPORTSE configuration file
'''

import os
import sys
import numpy
from numba import njit

# PARALLELISATION AND TUNING ###################################################
parallel_processes = 1   # number of threads
parallel_chemistry = 4   # number of threads
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
t = 86400 * 4      # simulation time (s)

## PROCESSES ###################################################################
FLUIDFLOW = True   # fluid flow (qx, qy must be given if set to False)
SPECTRANS = True   # species transport
HEATTRANS = True   # heat transport
FLUIDPROP = False  # fluid properties function
CHEMREACT = True   # chemical reactions
DYNAMICBC = False  # Use dynamic boundary conditions
MASSBALAN = False  # calculate mass balances
DTCHEM    = 1      # chemical reactions time stepping

## NUMBER OF SPECIES ###########################################################
IONS = 2			             # (0) Tracer (1) Ca
MINERALS = 2                     # (2) Anhydrite (3) pH
                                 #
SPECIES = IONS + MINERALS        # total number of species
SPTRANS = numpy.array([0,1])     # list of species to be transported

# MATRIX INITIALISATION ########################################################
## ROCK ########################################################################
phi   = numpy.zeros((nx, ny, nz))         # porosity (-)
Kx    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
Ky    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
Kz    = numpy.zeros((nx, ny, nz))         # permeability (m^2)
c_r   = numpy.zeros((nx, ny, nz))         # compressiblity (1/Pa)
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
c_f   = numpy.zeros((nx, ny, nz))         # compressiblity (1/Pa)
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
WP = numpy.zeros((nx, ny, nz))   # fluid flow (1/s)
WT = numpy.zeros((nx, ny, nz))   # heat flow  (W/m^3)
WC = numpy.zeros((nx, ny, nz))   # species flow (kg/m^3/s)

## FLOW VELOCITIES #############################################################
qx = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qy = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qz = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)

# PARAMETERS ###################################################################

## EQUATION OF STATE ###########################################################
# read temperature, density and  viscosity from phreeqc output
data = numpy.loadtxt('transportse-eos.csv', delimiter='\t', skiprows=2, 
        usecols=(8,9,10))

# fit polynomial functions using temperature
c_dens = numpy.polyfit(data[:,0], data[:,1], 4)
c_visc = numpy.polyfit(data[:,0], data[:,2], 4)

# define properties as polynomial functions of temperature
p_dens = numpy.poly1d(c_dens)
p_visc = numpy.poly1d(c_visc)

## CHEMICAL SURROGATE MODEL PARAMETERS #########################################
# read temperature and dissolved Ca from phreeqc output
data = numpy.loadtxt('anhydrite_solubility-phreeqc.csv', delimiter='\t',
        skiprows=2, usecols=(8,9))

# fit polynomial functions using temperature
c_anhy = numpy.polyfit(data[:,0], data[:,1], 4)

# define properties as polynomial functions of temperature
p_anhy = numpy.poly1d(c_anhy)

# define injection well location
wx = int(nx/2)
wy = int(ny/2)

# define parameters
phi   += 0.15
Kx    += 5.0e-13
Ky    += 5.0e-13
cp_r  += 1000.0
cp_f  += 4150.0
c_f   += 4.5e-10
c_r   += 1e-10
l_r   += 2.0
l_f   += 0.6
nue   += p_visc(100)
#nue[wx,wy,:] = p_visc(20)
rho_f += p_dens(100)
#rho_f[wx,wy,:] = p_dens(20)
rho_r += 2500
D_fx  += 5.0e-8
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

## SPECIES TRANSPORT
BC[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
BC[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
BC[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
BC[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)
BC[wx,wy,:] = 1      # Constant concentration at well

# INITIAL CONDITIONS ###########################################################

# PRESSURE
P[:,:,:] = 1.5e7             # Initial reservoir pressure

# TEMPERATURE
T[:,:,:] = 100               # Initial reservoir temperature
T[wx,wy,:] = 20              # Temperature at injection well

# TRACER CONCENTRATION
C[:,:,:,0] = 0                 # Initial concentration
C[wx,wy,:,0] = 50 * 1e-3       # Concentration at well

# Ca (mol/kgw)
C[:,:,:,1] = 0.0288836 #p_anhy(100)       # Initial concentration
C[wx,wy,:,1] = 0               # Concentration at well

# Anhydrite (mol/m^3)
C[:,:,:,2] = 150 
C[wx,wy,:,2] = 0.0

# pH
C[:,:,:,3] = 7 


# USER FLUID PROPERTY FUNCTION #################################################

@njit
def fluid_properties(C, T, P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi,
                     Kx, Ky, Kz, D_fx, D_fy, D_fz, rho_r):
    '''
        nue            viscosity (Pa*s)
        rho_f          density (kg/m^3)
        c_f            isothermal compressibility (1/Pa)
        cp_f           heat capacity (J/kg/K)
        l_f            thermal conductivity (W/m/K)
        D_f{x,y,z}     diffusion coefficient (m^2/s)
        ...
    '''
    #rho_fn = rho_f
    rho_f = c_dens[4] + c_dens[3] * T + c_dens[2] * T**2 + c_dens[1] * T**3 + \
            c_dens[0] * T**4
    nue = c_visc[4] + c_visc[3] * T + c_visc[2] * T**2 + c_visc[1] * T**3 + \
            c_visc[0] * T**4

    return C, T , P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, Kz, \
            D_fx, D_fy, D_fz, rho_r
 
#===============================================================================
# USER CHEMICAL REACTIONS FUNCTION #############################################

# LOAD PHREEQC module
if CHEMREACT:
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    if sys.platform == 'linux': # fix phreeqc_dll.py os.platform entry
        os.system('sed -i s/linux2/linux/g '+str(phreeqc_mod).split("'")[3])

    phreeqc = phreeqc_mod.IPhreeqc()
    phreeqc.load_database(r"phreeqc.dat")

def chemical_reactions(C, P, T, rho_f, phi, Kx, Ky, Kz):

    input_string = """
    SELECTED_OUTPUT 1
        #-high_precision true
    	-reset 		false
        -pH
    	-water 		true
    	-totals 	Ca S Na Cl
        -equilibrium_phases Anhydrite

    USER_PUNCH 1
        -headings Density_(kg/m^3)
        -start
            10 PUNCH RHO * 1000

    SOLUTION 1
    	units     mol/kgw 
        #press     """+ str(P / 101325) +"""
        press     """+ str(1.5e7 / 101325) +"""
    	temp      """+ str(T) +"""
        #density   """+ str(rho_f * 1e-3) +"""
        pH        """+ str(C[3]) +""" charge
    	Ca        """+ str(C[1]) +"""
    	S         """+ str(C[1]) +"""
        Na        1.7
        Cl        1.7
        water     1 # kg

    EQUILIBRIUM_PHASES 1
        # convert from mol/m^3 to mol/kgw
        Anhydrite   0   """+ str(C[2] / rho_f / phi) +"""   

    END"""
    
    try:
        ec = phreeqc.run_string(input_string)
        pout = numpy.array(phreeqc.get_selected_output_row(-1), dtype=float)

        # Extract relevant parameters from the output
        # C matrix:
        # IONS:     C0 Tracer
        #           C1 Ca (represents Ca and S in surrogate model solution)
        # MINERALS: C2 anhydrite
        #           C3 pH
        # phreeqc out:
        # (0) 'pH', (1) 'mass_H2O', (2) 'Ca(mol/kgw)', (3) 'S(mol/kgw)', 
        # (4) 'Na(mol/kgw)', (5) 'Cl(mol/kgw)', (6) 'Anhydrite', 
        # (7) 'd_Anhydrite', (8) 'Density_(kg/m^3)'

        # IONS (indices need to be adapted to amount of ions and minerals)
        C[1] = pout[2] 
        # MINERALS: back conversion from mol/kgw to mol/m^3
        C[2] = pout[6] * rho_f * phi 

        return C, rho_f, phi, Kx, Ky, Kz
            
    except:
        print('phreeqc error')

        return C, rho_f, phi, Kx, Ky, Kz

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
CONV_P = 1e-9 
CONV_C = 1e-5
CONV_T = 1e-5

## SMOLARKIEWICZ ANTI-DIFFUSION STEP ###########################################
SMOLAR_C = False
SMOLAR_T = False

# OUTPUT PARAMETERS ############################################################
PRINTOUT = 1        # output interval
PLOTSAVE = True     # save results
PLOTTIME = numpy.linspace(0,1,11) * t
