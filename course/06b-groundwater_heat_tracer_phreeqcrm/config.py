'''
    TRANSPORTSE configuration file

    Example 6b: All processes, chemistry via PhreeqcRM instead of phreeqpy

    Same model as example 6 (51 x 51 x 1 geothermal reservoir, 4 days, 50 m3/h
    injection of 20 degC water into a 100 degC reservoir saturated with respect
    to anhydrite), but the reaction step is done by the PhreeqcRM reaction
    module (USGS) instead of one IPhreeqc instance called per cell:

      example 6  : tse.py loops over all cells and calls config.chemical_reactions()
                   once per cell and time step (~2.2e6 calls for this run), each
                   call building and parsing a complete PHREEQC input string.
      example 6b : tse.py calls config.chemical_reactions_all() once per time step;
                   PhreeqcRM runs all 2601 cells internally and parallelises with
                   OpenMP (parallel_chemistry threads).

    Consequences:
      * the reaction step is one vectorised call instead of 2601 single calls,
      * pH is no longer an input that has to be transported and fed back: the
        module keeps the full solution state per cell, so C[3] becomes an
        output-only field (pH is read from the selected output),
      * anhydrite is tracked by the module as well (EQUILIBRIUM_PHASES), C[2]
        is therefore also output-only,
      * only the dissolved components that the transport equation actually moves
        (here Ca, with S tied to it as in example 6) are written back into the
        module before every RunCells() call,
      * requires phreeqcrm (project dependency) and the tse.py hook for
        config.chemical_reactions_all().

    pH AND REDOX
    PhreeqcRM transports the complete master set (H, O, Charge, Ca, Cl, Na, S),
    TSE only the solutes in SPTRANS. Two measures keep the redox state defined:
      * sulfur is transported as S(6) (sulphate, species 2), and the module's
        total oxygen is kept consistent with it (O = O_base + 4 S(6)),
      * the oxygen total carries a small dissolved O2 concentration
        (O2_MIN, 1e-4 mol/L = 3.2 mg/L, i.e. oxic pore water). Without it
        O(0) sits exactly on the redox boundary and the module reduces part of
        the S(6) to sulphide; the missing anion charge then shows up as a
        slightly basic pH (7.07 instead of 7.00) in the cells where the
        anhydrite has been leached completely.
    pH is an output here (C[4], read from the selected output); example 6 does
    the same in C[3].
'''

import os
import sys
import numpy
from numba import njit
import phreeqcrm

# base directory of this config file (data files are loaded relative to it,
# independent of the current working directory)
CFDIR = os.path.dirname(os.path.abspath(__file__))

# PARALLELISATION AND TUNING ###################################################
parallel_processes = 1   # number of threads
parallel_chemistry = 4   # number of threads used by PhreeqcRM (OpenMP)
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
CHEMREACT = True   # chemical reactions (PhreeqcRM)
DYNAMICBC = False  # Use dynamic boundary conditions
MASSBALAN = False  # calculate mass balances
DTCHEM    = 1      # chemical reactions time stepping
C_IMBLFLO = True
T_IMBLFLO = True
SIA       = False
SIA_maxit = 5
SIA_conv  = 1e-4

## NUMBER OF SPECIES ###########################################################
IONS = 3			             # (0) Tracer (1) Ca (2) S(6)
MINERALS = 2                     # (3) Anhydrite (4) pH
                                 #
SPECIES = IONS + MINERALS        # total number of species
SPTRANS = numpy.array([0,1,2])   # list of species to be transported

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
lf_l   = numpy.zeros((nx, ny, nz))        # longitudinal dispersion length (m)
lt_l   = numpy.zeros((nx, ny, nz))        # longitudinal dispersion length (m)
lf_t   = numpy.zeros((nx, ny, nz))        # transversal dispersion length (m)
lt_t   = numpy.zeros((nx, ny, nz))        # transversal dispersion length (m)
L     = numpy.zeros((nx, ny, nz))         # lithology marker (-)
R     = numpy.ones((nx, ny, nz))          # retardation coefficient (-)
mass  = numpy.zeros((3))                  # mass balance matrix

## FLUID #######################################################################
D_fx  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
D_fy  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
D_fz  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
c_f   = numpy.zeros((nx, ny, nz))           # compressiblity (1/Pa)
cp_f  = numpy.zeros((nx, ny, nz))           # heat capacity (J/kg/K)
l_f   = numpy.zeros((nx, ny, nz))           # heat conductivity (W/m/K)
rho_f = numpy.zeros((nx, ny, nz))           # density (kg/m^3)
nue   = numpy.zeros((nx, ny, nz))           # viscosity (Pa*s)

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
WP = numpy.zeros((nx, ny, nz))            # fluid flow (1/s)
WT = numpy.zeros((nx, ny, nz))            # heat flow  (W/m^3)
WC = numpy.zeros((nx, ny, nz, SPECIES))   # species flow (kg/m^3/s)

## FLOW VELOCITIES #############################################################
qx = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qy = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qz = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)

# PARAMETERS ###################################################################

## EQUATION OF STATE ###########################################################
# read temperature, density and  viscosity from phreeqc output
data = numpy.loadtxt(os.path.join(CFDIR, 'transportse-eos.csv'),
        delimiter='\t', skiprows=2, usecols=(8,9,10))

# fit polynomial functions using temperature
c_dens = numpy.polyfit(data[:,0], data[:,1], 4)
c_visc = numpy.polyfit(data[:,0], data[:,2], 4)

# define properties as polynomial functions of temperature
p_dens = numpy.poly1d(c_dens)
p_visc = numpy.poly1d(c_visc)

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
D_fx  += 5.0e-9
D_fy  = D_fx.copy()
lf_l   += 0.5
lf_t   += 0.5

# BOUNDARY CONDITIONS ##########################################################

## FLUID FLOW
BP[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
BP[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
BP[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
BP[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)

# Injection rate at injection well (50 m^3/h)
WP[wx, wy, :] = 50 / 3600 / dx[wx] / dy[wy] / dz[0] * rho_f[wx, wy, :]

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

# S(6) (mol/kgw), tied to Ca by the anhydrite stoichiometry
C[:,:,:,2] = 0.0288836
C[wx,wy,:,2] = 0

# Anhydrite (mol/m^3)
C[:,:,:,3] = 150 
C[wx,wy,:,3] = 0.0

# pH
C[:,:,:,4] = 7 

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
    rho_f = c_dens[4] + c_dens[3] * T + c_dens[2] * T**2 + c_dens[1] * T**3 + \
            c_dens[0] * T**4
    nue = c_visc[4] + c_visc[3] * T + c_visc[2] * T**2 + c_visc[1] * T**3 + \
            c_visc[0] * T**4

    return C, T , P, nue, rho_f, c_f, cp_f, l_f, cp_r, l_r, phi, Kx, Ky, Kz, \
            D_fx, D_fy, D_fz, rho_r

#===============================================================================
# PHREEQCRM REACTION MODULE ####################################################
# The module holds the complete chemical state of every cell. TRANSPORTSE only
# exchanges the dissolved components it transports (Ca, S) and reads back the
# mineral amount and pH from the selected output.

NXYZ = nx * ny * nz        # number of grid cells
O2_MIN = 1.0e-4            # dissolved O2 in the pore water (mol/L = 3.2 mg/L),
                           # defines the redox state (see pH AND REDOX above)
RV_L = 1.0                 # representative volume per cell (L), keeps the
                           # water volume per cell close to 1 L as recommended
                           # by the PhreeqcRM documentation

rm = phreeqcrm.PhreeqcRM(NXYZ, parallel_chemistry)
rm.SetErrorHandlerMode(1)          # 1 = throw an exception on error
rm.SetComponentH2O(False)          # H2O is not a transport component
rm.SetUnitsSolution(2)             # 2 = mol/L (TSE transports per m^3 water)
# NOTE: the units of EQUILIBRIUM_PHASES are left at the module default, i.e. the
# phase amount is transferred 1:1 as moles per cell. Setting mol/kgw here would
# rescale the initial mineral amount by the water content of a cell.
for _f in ('SetUnitsExchange', 'SetUnitsGasPhase', 'SetUnitsKinetics',
           'SetUnitsSSassemblage', 'SetUnitsSurface'):
    getattr(rm, _f)(1)
rm.SetFilePrefix('rm')             # prefix of the PhreeqcRM output files
rm.SetPrintChemistryOn(False, False, False)  # no per-cell chemistry output
rm.LoadDatabase(os.path.join(CFDIR, 'phreeqc.dat'))

RM_INPUT = """
SELECTED_OUTPUT 1
    -reset false
    -pH true
    -water true
    -totals Ca S Na Cl
    -equilibrium_phases Anhydrite
USER_PUNCH 1
    -headings rho
    -start
    10 PUNCH RHO * 1000
    -end
SOLUTION 1
    units mol/kgw
    press 148.04
    temp 100
    pH 7 charge
    Ca %.7f
    S(6) %.7f
    Na 1.7
    Cl 1.7
    water 1
EQUILIBRIUM_PHASES 1
    Anhydrite 0 %.7e
END"""

# Ca concentration of the initial reservoir solution (TSE units, mol/kgw)
CA_INIT = C[0, 0, 0, 1]


def _init_module(anhy_mol):
    """(Re-)initialise the reaction module with one solution and one anhydrite
       amount per cell; returns nothing, the module keeps the state."""
    rc = rm.RunString(True, True, True,
                      RM_INPUT % (CA_INIT, CA_INIT, anhy_mol))
    if rc != 0:
        raise RuntimeError('PhreeqcRM RunString failed: %s'
                           % rm.ErrorMessage(rc))
    rc = rm.FindComponents()
    if rc < 0:
        raise RuntimeError('PhreeqcRM FindComponents failed: %s'
                           % rm.ErrorMessage(rc))
    # initial condition of every cell: SOLUTION 1 + EQUILIBRIUM_PHASES 1
    ic = numpy.full(7 * NXYZ, -1, dtype=numpy.int32)
    ic[0:NXYZ] = 1
    ic[NXYZ:2 * NXYZ] = 1
    rc = rm.InitialPhreeqc2Module(ic)
    if rc != 0:
        raise RuntimeError('PhreeqcRM InitialPhreeqc2Module failed: %s'
                           % rm.ErrorMessage(rc))


# cell geometry for the module
rm.SetPorosity(phi.ravel())
rm.SetSaturation(numpy.ones(NXYZ))
rm.SetRepresentativeVolume(numpy.full(NXYZ, RV_L))
rm.SetTimeConversion(1.0)          # time unit of the simulation is seconds
rm.SetSelectedOutputOn(True)

# --- calibration against the TSE state (once, at config import) --------------
# The module derives its own water mass per cell from porosity, saturation and
# representative volume, and its own density from the database. Both enter the
# conversion between TSE concentrations and the module's transport array, so
# they are measured here instead of being assumed.
_init_module(C[0, 0, 0, 3] * RV_L * 1e-3)          # placeholder amount
_comps = [str(_c) for _c in rm.GetComponents()]
ICA = _comps.index('Ca')
IS = _comps.index('S')
IOX = _comps.index('O')            # total oxygen, needed for the redox state
NCOMP = len(_comps)
_conc0 = numpy.array(rm.GetConcentrations(), dtype=float)
rm.SetTemperature(T.ravel())
rm.SetPressure(P.ravel() / 101325.0)               # Pa -> atm
rm.SetConcentrations(_conc0)
rc = rm.RunCells()
if rc != 0:
    raise RuntimeError('PhreeqcRM RunCells failed: %s' % rm.ErrorMessage(rc))
_so0 = numpy.array(rm.GetSelectedOutput(), dtype=float)
_head0 = [str(_h) for _h in rm.GetSelectedOutputHeadings()]
IPH = _head0.index('pH')
IANHY = _head0.index('Anhydrite')
IRHO = _head0.index('rho')

# water mass per cell as seen by the module (kg) and the resulting cell volume
MASS_H2O = float(_so0[_head0.index('mass_H2O') * NXYZ])
V_CELL_MOD = MASS_H2O / (rho_f[0, 0, 0] * phi[0, 0, 0])   # m^3, TSE-consistent

# module transport units per TSE concentration unit (mol/L per mol/kgw),
# referenced to the module density of the calibration state: the conversion
# scales with the solution density, so it is corrected per cell below
K_C = float(_conc0[ICA * NXYZ]) / CA_INIT
RHO_REF = float(_so0[IRHO * NXYZ])                # kg/m^3 at calibration
_rho_mod = numpy.full(NXYZ, RHO_REF)              # module density, last step

# anhydrite amount per cell (mol) that corresponds to C[2] (mol/m^3 bulk)
ANHY_MOL = C[0, 0, 0, 3] * V_CELL_MOD

# re-initialise with the calibrated anhydrite amount
_init_module(ANHY_MOL)
conc = numpy.array(rm.GetConcentrations(), dtype=float)
_o_base = conc[IOX * NXYZ:(IOX + 1) * NXYZ] \
    - 4.0 * conc[IS * NXYZ:(IS + 1) * NXYZ]


def chemical_reactions_all(C, P, T, rho_f, phi, Kx, Ky, Kz, dt_chem):
    '''
        Vectorised reaction step: one PhreeqcRM call for all cells.

        The module transports the complete master set (H, O, Charge, Ca, Cl, Na,
        S), TSE moves Ca and S(6). Sulfur is transported as S(6) (sulfate) and
        the total oxygen is kept consistent with it (O = O_base + 4 S(6)), so
        PHREEQC does not re-speciate sulfur as sulfide - otherwise the redox
        state, and with it pH, drifts. O_base is re-derived from the module
        state after every RunCells() call.
    '''
    global _o_base, _rho_mod, IPH, IANHY, IRHO

    # conversion factor per cell: scales with the module's solution density
    k_cell = K_C * _rho_mod / RHO_REF
    ca_rm = C[:, :, :, 1].ravel() * k_cell       # Ca, module units
    s6_rm = C[:, :, :, 2].ravel() * k_cell       # S(6), module units

    conc[ICA * NXYZ:(ICA + 1) * NXYZ] = ca_rm
    conc[IS * NXYZ:(IS + 1) * NXYZ] = s6_rm
    conc[IOX * NXYZ:(IOX + 1) * NXYZ] = _o_base + 4.0 * s6_rm + O2_MIN

    rm.SetTimeStep(dt_chem)
    rm.SetTemperature(T.ravel())
    rm.SetPressure(P.ravel() / 101325.0)
    rm.SetConcentrations(conc)
    rc = rm.RunCells()
    if rc != 0:
        raise RuntimeError('PhreeqcRM RunCells failed: %s'
                           % rm.ErrorMessage(rc))
    conc[:] = rm.GetConcentrations()
    # oxygen not bound to sulfate, for the next step
    _o_base = conc[IOX * NXYZ:(IOX + 1) * NXYZ] \
        - 4.0 * conc[IS * NXYZ:(IS + 1) * NXYZ]

    so = numpy.array(rm.GetSelectedOutput(), dtype=float)
    _rho_mod = so[IRHO * NXYZ:(IRHO + 1) * NXYZ].copy()

    # write back into the cells that are not Dirichlet boundaries (tse.py does
    # not run chemistry there, and neither does example 6)
    inner = (BC != 1)
    C[:, :, :, 1][inner] = (conc[ICA * NXYZ:(ICA + 1) * NXYZ]
                            / k_cell).reshape(nx, ny, nz)[inner]
    C[:, :, :, 2][inner] = (conc[IS * NXYZ:(IS + 1) * NXYZ]
                            / k_cell).reshape(nx, ny, nz)[inner]
    C[:, :, :, 3][inner] = (so[IANHY * NXYZ:(IANHY + 1) * NXYZ]
                            / V_CELL_MOD).reshape(nx, ny, nz)[inner]
    C[:, :, :, 4][inner] = so[IPH * NXYZ:(IPH + 1) * NXYZ] \
        .reshape(nx, ny, nz)[inner]

    return C, P, T, rho_f, phi, Kx, Ky, Kz

#===============================================================================

# SOLVER PARAMETERS ############################################################
## TIME STEP SETTINGS ##########################################################
dt =     0.001  # initial time step size
dt_max = 1e3

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
CONV_P = 1e-7
CONV_C = 1e-7
CONV_T = 1e-7

## SMOLARKIEWICZ ANTI-DIFFUSION STEP ###########################################
SMOLAR_C = False
SMOLAR_T = False

# OUTPUT PARAMETERS ############################################################
PRINTOUT = 1        # output interval
PLOTSAVE = True     # save results
PLOTTIME = numpy.linspace(0,1,11) * t
