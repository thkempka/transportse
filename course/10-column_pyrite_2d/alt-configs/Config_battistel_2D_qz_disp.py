'''
    TRANSPORTSE configuration file - Battistel 2D chamber, scenario "qz" WITH
    the paper's hydrodynamic dispersion closure ("qz_disp")

    Scenario: chamber laid out as in Battistel et al.'s figure (0.8 m long in x,
    0.24 m high in z), pyrite lens at x = 5-21 cm and z = 10-14 cm, inlet on the
    x = 0 plane, D_f = 1e-9 m2/s in all directions, influx pH 5.8, O(0) 5e-4
    mol/kgw (= 2.5e-4 mol/L O2, see the note on O(0) below), imposed Darcy flux
    1.5 m/d in x.  This is the scenario that reproduces Battistel's reference
    profiles best: S peak 1.268e-4 mol/L vs 1.286e-4
    (Battistel, simulation) and 1.255e-4 (experiment), FWHM 4.4 vs 4.5 / 6.0 cm,
    at the comparison plane x = 76 cm.

    This file = ../config.py (which reproduces the archived run to
    machine precision) + ONE physical addition, item (7): the dispersion tensor
    of Battistel et al. (2019), Eqs. [10]-[11].  The archived runs are the "noD"
    ones (l_l = l_t = 0, see the archived figure names), i.e. they had no
    mechanical dispersion at all, while the paper computes longitudinal and
    transverse spreading from the grain size and the flow speed.  Everything
    else is untouched: influx pH 5.8, O(0), pyrite content (12.85 mol/L_bulk =
    Table 1), D_f, grid, flux, theta_g, lambda_, keff.

      (1) D_fx / D_fy / D_fz / WC are now (nx, ny, nz, SPECIES) arrays; the
          current tse.py indexes them per species (3-D arrays used to raise
          "IndexError: too many indices for array").
      (2) the dispersivity arrays were renamed: lf_l / lf_t (fluid) and
          lt_l / lt_t (heat, unused) are derived from l_l / l_t with .copy(),
          exactly as in the migrated 1-D configs (no aliasing).

      (1) D_fx / D_fy / D_fz / WC are now (nx, ny, nz, SPECIES) arrays; the
          current tse.py indexes them per species (3-D arrays used to raise
          "IndexError: too many indices for array").
      (2) the dispersivity arrays were renamed: lf_l / lf_t (fluid) and
          lt_l / lt_t (heat, unused) are derived from l_l / l_t with .copy(),
          exactly as in the migrated 1-D configs (no aliasing).
      (3) flags required by the current tse.py: SIA, SIA_maxit, SIA_conv,
          SIA_conv2, C_IMBLFLO, T_IMBLFLO (all off, as the archived runs).
      (4) chemical_reactions() now takes the (i, j, k) cell indices that
          tse.py passes and returns the full 8-tuple C, P, T, rho_f, phi,
          Kx, Ky, Kz.
      (5) the current tse.py does not call initialise_equilibrium() /
          initialise_influx() any more, so they are applied here cell by cell
          according to the lithology marker L (1 = sand, 2 = pyrite lens,
          3 = inlet), like in the migrated 1-D configs.
      (6) dt_max = 576 s (= dt) instead of 86400 s, and MAXIT_C = 2000 - the
          same corrections as in examples 07-09 (see the comments below).
      (7) NEW in this variant: l_l / l_t are no longer zero but the dispersivities
          that reproduce Eqs. [10]-[11] of Battistel et al. (2019); see the
          derivation in the ROCK section below.

    NOTE on O(0): the PHREEQC redox master species O(0) is ATOMIC oxygen - one
    mole of O(0) accepts 2 electrons, a real O2 molecule accepts 4.  PHREEQC
    therefore reports m_O2 = O(0)/2, so our influx O(0) = 5e-4 mol/kgw is exactly
    the 2.5e-4 mol/L O2 of the paper's Table 1, and the initial O(0) = 6e-5 is
    their 3.0e-5.  Verified with a separate batch check, which
    also shows that the model's dS/dO(0) = 0.258, i.e. dS/dO2 = 0.516 = the
    paper's own 1.72/3.33 from their Eqs. [1]-[3].  Do NOT halve these numbers.
'''

import os
import sys
import numpy
from numba import njit

# base directory of this config file (data files are loaded relative to it,
# independent of the current working directory)
CFDIR = os.path.dirname(os.path.abspath(__file__))

# PARALLELISATION AND TUNING ###################################################
parallel_processes = 4   # number of threads
parallel_chemistry = 16   # number of threads
numba_fastmath = False   # turn off strict IEEE 754 compliance
numba_tl = 'omp'         # numba threading layer (tbb, omp, workqueue)

# MODEL DIMENSIONS AND SPATIAL DISCRETISATION ##################################
lx = 0.8                        # model length in x-direction (m)
ly = 0.05                       # model length in y-direction (m)
lz = 0.24                        # model length in z-direction (m)
nx = 200                         # number of nodes in x-direction (-)
ny = 1                          # number of nodes in y-direction (-)
nz = 60                         # number of nodes in z-direction (-)
dx = numpy.ones((nx)) * lx / nx  # nodal distance x-direction (m)
dy = numpy.ones((ny)) * ly / ny  # nodal distance y-direction (m)
dz = numpy.ones((nz)) * lz / nz  # nodal distance z-direction (m)

# PARAMETRISATION ##############################################################
g = 9.80665        # gravitational acceleration (m/s^2)
beta =  0          # model slope angle (°)


## SIMULATION TIME #############################################################
t = 60 * 60 * 98      # simulation time (s); 98 hours

## PROCESSES ###################################################################
FLUIDFLOW = False   # fluid flow (qx, qy must be given if set to False)
SPECTRANS = True   # species transport
HEATTRANS = False   # heat transport
FLUIDPROP = False  # fluid properties function
CHEMREACT = True   # chemical reactions
DYNAMICBC = False  # Use dynamic boundary conditions
MASSBALAN = False  # calculate mass balances
SIA = False        # semi-implicit algorithm (required by the current tse.py)
SIA_maxit = 10
SIA_conv = 1e-10
SIA_conv2 = 1e-10
C_IMBLFLO = False  # immobile water for solute transport
T_IMBLFLO = False  # immobile water for heat transport
KINETICS  = True   # Calcite dissolution/precipitation kinetics
PUMPING   = False  # PHS simulations
DTCHEM    = 1      # chemical reactions time stepping

## NUMBER OF SPECIES ###########################################################
IONS = 12			             # (0) Tracer (1) pH (2) pe (3) Na (4) Cl (5) S
                                 # (6) O(2) (7) Fe (8) SO42- (9) HCO3- (10) CO2(g)
                                 # (11) O2(g)
MINERALS = 3                     # (12) Pyrite (13) Goethite (14) c_g

SPECIES = IONS + MINERALS        # total number of species
SPTRANS = numpy.array([0,1,2,    # list of species to be transported
                       3,4,5,
                       6,7,8,9,
                       10,11])   
C_sand = numpy.zeros(MINERALS)
C_inclusion = numpy.zeros(MINERALS)

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
D_fx  = numpy.zeros((nx, ny, nz, SPECIES))   # diffusion coefficient (m^2/s), per species
D_fy  = numpy.zeros((nx, ny, nz, SPECIES))   # diffusion coefficient (m^2/s), per species
D_fz  = numpy.zeros((nx, ny, nz, SPECIES))   # diffusion coefficient (m^2/s), per species
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
WC = numpy.zeros((nx, ny, nz, SPECIES))   # species flow (kg/m^3/s), per species

## FLOW VELOCITIES #############################################################
qx = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qy = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qz = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)


# PARAMETERS ###################################################################

# define parameters
#phi   += 0.347              # 1D: phi(water): 0.347 , phi(gas): 0.033
                           # 2D: phi(water): 0.358 , phi(gas): 0.023
Kx    += 5.0e-16
Ky    += 5.0e-16
Kz    += 5.0e-16
cp_r  += 1000.0
cp_f  += 4150.0
c_f   += 4.5e-10
c_r   += 1e-10
l_r   += 2.0
l_f   += 0.6
nue   += 1e-5
#R     += 1.7 #1.2      
#nue[wx,wy,:] = p_visc(20)
rho_f += 1000
#rho_f[wx,wy,:] = p_dens(20)
rho_r += 2650
D_fx  += 1e-9
D_fy  = D_fx.copy()
D_fz  = D_fx.copy()

# --- hydrodynamic dispersion, Battistel et al. (2019) Eqs. [10] and [11] -----
# The paper does not use a fitted dispersivity; it computes the tensor entries
#     D_L = D_P + 0.5 d v
#     D_T = D_P + D_aq / (Pe^2 + 2 + 4 sqrt(2)),      Pe = v d / D_aq
# with v = 1.5 m/d as the SEEPAGE (pore) velocity, d = 0.125-0.250 mm sand and
# D_aq = 2e-9 m2/s (their Table 2), giving Pe = 1.63.
#
# tse.py adds the mechanical part as lf_l * |q| / lf_t * |q|, where q is the
# SPECIFIC DISCHARGE (q = 6.215e-6 m/s here), while the paper writes alpha * v
# with the pore velocity v = q / theta_w.  The dispersivities must therefore be
# divided by theta_w = 0.358, which gives
#     alpha_L = 0.5 d                 = 9.375e-5 m  ->  l_l = 2.619e-4 m
#     alpha_T = D_aq/(v(Pe^2+2+4sqrt2)) = 1.118e-5 m ->  l_t = 3.123e-5 m
# and hence a mechanical spreading of l_l*q = 1.63e-9 (= 0.5 d v) and l_t*q =
# 1.94e-10 m2/s.  With the unchanged D_f * theta_w = 3.58e-10 this yields
#     D_L = 1.99e-9   and   D_T = 5.5e-10 m2/s   (isotropically: 1e-9 before).
# D_f itself is deliberately left at 1e-9: it carries the pore-diffusion part
# and is the value the archived runs were tuned with.
d_grain = 0.1875e-3                      # m, mid-point of the 125-250 um sand
D_aq    = 2e-9                           # m2/s, aqueous diffusion coeff. (Table 2)
v_seep  = 1.5 / 86400.0                  # m/s, seepage velocity (paper, 1.5 m/d)
theta_w_disp = 0.358                     # -, needed once here, set again below
Pe_gr   = v_seep * d_grain / D_aq        # = 1.63
l_l   += 0.5 * d_grain / theta_w_disp                    # alpha_L / theta_w
l_t   += D_aq / (v_seep * (Pe_gr**2 + 2 + 4 * numpy.sqrt(2))) / theta_w_disp

# --- naming of the dispersivity arrays in the current tse.py (as in 1D) ------
lf_l  = l_l.copy()                     # longitudinal dispersion length (m)
lf_t  = l_t.copy()                     # transversal dispersion length (m)
lt_l  = numpy.zeros((nx, ny, nz))      # longitudinal dispersion, heat (unused)
lt_t  = numpy.zeros((nx, ny, nz))      # transversal dispersion, heat (unused)
# NOTE: both dispersivities are zero here ("noD" in the names of the archived
# figures), so mixing across the 0.24 m height is D_f = 1e-9 m2/s plus upwind
# numerical diffusion (~0.5*|q|*dz = 3e-8 m2/s for the 4 mm cells of this
# rotated grid).  The upright scenarios (1 mm cells across) therefore mix less
# and give a narrower profile.  D_fz = D_fx.copy() below is
# what the upright configs lack (they leave D_fz = 0, which is the direction of
# the flow there).


theta_w =  0.358  
theta_g =  0.022 
phi += theta_w
lambda_ =  4.26e-6
H = 31.25

# GEOMETRY AND LITHOLOGY #######################################################
x_coords = numpy.linspace(0, lx, nx)  # Shape (nx,)
z_coords = numpy.linspace(0, lz, nz)  # Shape (nz,)

# Create meshgrid for 2D coordinates

X, Z = numpy.meshgrid(x_coords, z_coords, indexing="ij")  # X, Z shapes: (nx, nz)

# Create mask and expand dimensions
mask = (Z >= 0.1) & (Z <= 0.14) & (X >= 0.05) & (X <= 0.21)  # Shape (nx, nz)
mask = numpy.expand_dims(mask, axis=1)  # Shape (nx, 1, nz)
#Apply mask to update L
L[mask] = 2
L[L != 2] = 1               # Sand 
#mask_inlet = (Z >= 0.1) & (Z <= 0.14) & (X == 0)  # Shape (nx, nz)
#mask_inlet = numpy.expand_dims(mask_inlet, axis=1)  # Shape (nx, 1, nz)
#L[mask_inlet] = 3                # Inlet
L[0,:,:] = 3                # Inlet
#mask_outlet = (Z >= 0.1) & (Z <= 0.14) & (X == 0.8)  # Shape (nx, nz)
#mask_outlet = numpy.expand_dims(mask_outlet, axis=1)  # Shape (nx, 1, nz)
#L[mask_outlet] = 4                # Outlet


# BOUNDARY CONDITIONS ##########################################################


## FLUID FLOW
#BP[0,:,:] = 0        # Left boundary closed (Dirichlet-BC)
BP[-1,:,:] = 1       # Right boundary closed (Dirichlet-BC)
#BP[:,:,0] = 0        # Bottom boundary closed (Dirichlet-BC)
#BP[:,:,-1] = 1        # Top boundary open (Dirichlet-BC)


# ## HEAT TRANSPORT
# BT[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
# BT[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
# BT[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
# BT[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)

## SPECIES TRANSPORT
BC[L == 3] = 1        # Inlet
#BC[0,:,:] = 0        # Left boundary closed (Dirichlet-BC)
BC[-1,:,:] = 1       # Right boundary closed (Dirichlet-BC)
#BC[:,:,0] = 0        # Bottom boundary closed (Dirichlet-BC)
#BC[:,:,-1] = 1        # Top boundary open (Dirichlet-BC)

# INITIAL CONDITIONS ###########################################################

## Injection rate WP #v_seep = 0.0625 m/h; A=0.00022698 m^2
#A_column = (0.017/2)**2 * numpy.pi          ## diameter column: 1.7 cm

## v_seep ist 1.5 m/d --> porosität berücksichtigen und auf Säuleneinlass umrechnen
#WP[:,:,0] = phi * 1.5 / 86400 * A_column / dx[0] / dy[0] / dz[0] # 1/s 


## Fluidflow false --> qz muss manuell definiert werden;
## Porosität berücksichtigen da 1.5 m/d sickergeschwindigkeit -> 
qx += phi * 1.5 / 86400 

# PRESSURE
P[:,:,:] = 1e5             # Initial air pressure

# # TEMPERATURE
T[:,:,:] = 20               # Initial reservoir temperature
C_sand[0] = 0.0             # Pyrite (mol/m^3) initial concentration sand
C_inclusion[0] = 12.85 * 1000 # Pyrite (mol/m^3) initial concentration inclusion
                               # = 12.85 mol/L_bulk, exactly Table 1 of Battistel
                               # et al. (2019) for the 2-D chamber (1:1 pyrite:sand
                               # by mass); verified, nothing to adjust here
C_sand[1] = 0.0             # Goethite (mol/m^3) initial concentration sand
C_inclusion[1] = 0.0        # Goethite (mol/m^3) initial concentration inclusion


C[:,:,:,14] = 0.00006 #2.291e-04 #0.000001 #2.291e-04

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
    db_path = os.path.join(CFDIR, "phreeqc.dat")
    if not os.path.exists(db_path):
        # abort here instead of silently continuing without chemistry
        raise FileNotFoundError(
            f"The PHREEQC database file '{db_path}' was not found.")
    phreeqc.load_database(db_path)


    def initialise_equilibrium(C, C_mineral, P, T, rho_f, phi):
        input_string = """
        SELECTED_OUTPUT 1
            -high_precision true
        	-reset 		false
            -pH
            -pe
        	-water 		true
        	-totals 	Na Cl S O(0) Fe
            -molalities SO4-2 HCO3-
            -equilibrium_phases Pyrite
            -saturation_indices CO2(g) O2(g)
    
        SOLUTION 1
        	units     mol/kgw
            press     """+ str(P / 101325) +"""
        	temp     """+ str(T) +"""
            pH         6.77
        	Na         0.005
            Cl         0.005
            water       1 # kg
            O(0)       0.00006
            
        EQUILIBRIUM_PHASES 1
            # convert minerals from mol/m^3 to mol/kgw
             Pyrite    0   """+ str(C_mineral[0] / rho_f / phi) +"""
    
        END"""
    
        try:
            ec = phreeqc.run_string(input_string)
            pout = numpy.array(phreeqc.get_selected_output_row(-1), dtype=float)
            #print(C_mineral[0])
            # Extract relevant parameters from the output
            # C matrix:
            # IONS:     C0 Tracer
            #           C1 pH
            #           C2 pe 
            #           C3 Na 
            #           C4 Cl 
            #           C5 S 
            #           C6 O(0) 
            #           C7 Fe
            #           C8 SO4
            #           C9 HCO3
            #           C10 pCO2 
            #           C11 pO2 
            # MINERALS: C12 Pyrite 
            # (0) 'pH', (1) 'pe', (2) 'mass_H2O', (3) 'Na(mol/kgw)', (4) 'Cl(4)(mol/kgw)', 
            # (5) 'S(mol/kgw)', (6) 'O(0)(mol/kgw)', (7) 'Fe(mol/kgw)', (8) 'SO4(mol/kgw)',
            # (9) 'HCO3(mol/kgw)', (10) 'Pyrite', (11) 'd_Pyrite',
            # (12) 'CO2(g)', (13) 'O2(g)'
            # IONS (indices need to be adapted to amount of ions and minerals)
            C[1] = pout[0]        # pH in pore space
            C[2] = pout[1]        # pe in pore space
            C[3] = pout[3]        # Na+ in pore space
            C[4] = pout[4]        # Cl- in pore space
            C[5] = pout[5]        # S in pore space
            C[6] = pout[6]        # O(0) in pore space
            C[7] = pout[7]        # Fe in pore space
            C[8] = pout[8]        # SO4-2 in pore space
            C[9] = pout[9]        # HCO3- in pore space
            C[10] = pout[12]      # pCO2 in pore space
            C[11] = pout[13]      # pO2 in pore space
            C[12] = pout[10] * rho_f * phi  # Pyrite in pore space
    
            return C
    
        except:
            print(input_string)
            print('phreeqc error while initialising')
            print(phreeqc.get_error_string())
            return C
    
    
    def initialise_influx(C, P, T):
        input_string = """
        SELECTED_OUTPUT 1
            -high_precision true
        	-reset 		false
            -pH
            -pe
        	-water 		true
        	-totals 	Na Cl S O(0) Fe
            -molalities SO4-2 HCO3-
            -saturation_indices CO2(g) O2(g)
    
        SOLUTION 1
        	units     mol/kgw
            press      """+ str(P / 101325) +"""
        	temp     """+ str(T) +"""
            pH         5.8
        	Na         0.005
            Cl         0.005
            water      1 # kg
            O(0)       0.0005
            
        END"""
    
        try:
            ec = phreeqc.run_string(input_string)
            pout = numpy.array(phreeqc.get_selected_output_row(-1), dtype=float)
            #print(pout)
            # Extract relevant parameters from the output
            # C matrix:
            # IONS:     C0 Tracer
            #           C1 pH
            #           C2 pe 
            #           C3 Na 
            #           C4 Cl 
            #           C5 S 
            #           C6 O(0) 
            #           C7 Fe
            #           C8 SO4
            #           C9 HCO3
            #           C10 pCO2 
            #           C11 pO2 
            # (0) 'pH', (1) 'pe', (2) 'mass_H2O', (3) 'Na(mol/kgw)', (4) 'Cl(4)(mol/kgw)', 
            # (5) 'S(mol/kgw)', (6) 'O(0)(mol/kgw)', (7) 'Fe(mol/kgw)', (8) 'SO4(mol/kgw)',
            # (9) 'HCO3(mol/kgw)', (10) 'CO2(g)', (11) 'O2(g)'
            # IONS (indices need to be adapted to amount of ions and minerals)
            C[0] = 50 * 1e-3
            C[1] = pout[0]      # pH in influx
            C[2] = pout[1]      # pe in influx
            C[3] = pout[3]      # Na in influx
            C[4] = pout[4]      # Cl in influx
            C[5] = pout[5]      # S in influx
            C[6] = pout[6]      # O(0) in influx
            C[7] = pout[7]      # Fe in influx
            C[8] = pout[8]      # SO4-2 in influx
            C[9] = pout[9]      # HCO3- in influx
            C[10] = pout[10]    # pCO2 in influx
            C[11] = pout[11]    # pO2 in influx
    
    
            return C
    
        except:
            print(input_string)
            print('phreeqc error while initialising')
            print(phreeqc.get_error_string())
            return C
    
    
    
    
    def chemical_reactions(C, P, T, rho_f, phi, Kx, Ky, Kz, dt_chem,
                       i, j, k):
        #print(C[5])
        input_string = """
        SELECTED_OUTPUT 1
            -high_precision true
        	-reset 		false
            -pH
            -pe
        	-water 		true
        	-totals 	Na Cl S O(0) Fe
            -molalities SO4-2 HCO3-
            -equilibrium_phases Pyrite Goethite
            -saturation_indices      CO2(g) O2(g)
            -kinetic_reactants Pyrite Goethite
    
        USER_PUNCH 1
            -headings Density_(kg/m^3)
            -start
                10 PUNCH RHO * 1000
    
        SOLUTION 1
        	units     mol/kgw
            pe		   """+ str(C[2]) +"""
            press      """+ str(P / 101325) +"""
        	temp       """+ str(T) +"""
            #density   """+ str(rho_f * 1e-3) +"""
            pH         """+ str(C[1]) +""" 
        	Na         """+ str(C[3]) +"""
        	Cl         """+ str(C[4]) +"""
            #S(6)       """+ str(round(C[8], 8)) +"""
            S           """+ str(round(C[5], 9)) +"""
            water       1 # kg
            O(0)       """+ str(C[6]) +"""
            Fe         """+ str(C[7]) +"""
    
    #######
    #Pyrite
    #######
           # Williamson, M.A. and Rimstidt, J.D., 1994,
    KINETICS 1
    Pyrite
            -formula FeS1.72
            -m0     """+ str((C[12] / rho_f / phi)) +"""
			-parms  2.8

    Goethite
        -tol    1e-8
        -m     """+ str(C[13] / rho_f / phi) +"""
        -parms  1.5e-13 for 2D
        
    -steps  """+ str(dt_chem) +""" seconds in 1 steps
    #INCREMENTAL_REACTIONS True
    
    RATES
    Pyrite
      -start
    1   REM        Williamson and Rimstidt, 1994
    2   REM        Geochimica et Cosmochimica Acta, v. 58, p. 5443-5454,
    3   REM        rate equation is mol m^-2 s^-1.
    4   REM        PARM(1) = log10(specific area), log10(m^2 per mole pyrite)
    5   REM        PARM(2) = exp for (M/M0)
    6   REM        PARM(3) = exp for O2
    7   REM        PARM(4) = exp for H+
    
    10  A = PARM(1) * m0
    30  if SI("Pyrite") > 0 THEN GOTO 100
    50  fH = mol("H+")
    60  if mol("O2") < 1e-6 then goto 100
    70  rO2 = 10^-8.19 * mol("O2")^0.5 * fH^-0.11
    80  rate = A * (m/m0)^0.67 * (rO2) * (1-SR("Pyrite"))
    100 save rate * time
      -end
    
    Goethite
      -start
    1   REM        M. Battistel et al., 2018
    2   REM        PARM(1) = FeOOH effective rate constant, keff [mol/L/s]
    
    10  REM Precipitation of Goethite
    20  if (SI("Goethite") <= 0) THEN GOTO 200
    40  omega_goe = 10^(SI("Goethite"))
    50  RATE = PARM(1) * (1-omega_goe)
    60  moles = RATE * TIME
    200 SAVE moles
      -end
    #EQUILIBRIUM_PHASES
    #    Pyrite  0 """+ str((C[12] / rho_f / phi)) +"""
    #    Goethite 0 """+ str(C[13] / rho_f / phi) +"""
    END"""
    
        try:
            ec = phreeqc.run_string(input_string)
            pout = numpy.array(phreeqc.get_selected_output_row(-1), dtype=float)
            #print(phreeqc.get_selected_output_array())
            #print(phreeqc.get_selected_output_row(-1))
            # Extract relevant parameters from the output
            # C matrix:
            # IONS:     C0 Tracer
            #           C1 pH
            #           C2 pe 
            #           C3 Na 
            #           C4 Cl 
            #           C5 S 
            #           C6 O(0) 
            #           C7 Fe
            #           C8 SO4
            #           C9 HCO3
            #           C10 pCO2 
            #           C11 pO2 
            # MINERALS: C12 Pyrite
            #           C13 Goethite
            #           C14 c_g
            # (0) 'pH', (1) 'pe', (2) 'mass_H2O', (3) 'Na(mol/kgw)', (4) 'Cl(mol/kgw)', 
            # (5) 'S(mol/kgw)', (6) 'O(0)(mol/kgw)', (7) 'Fe(mol/kgw)', (8) 'SO4(mol/kgw)',
            # (9) 'HCO3(mol/kgw)', (10) 'Pyrite', (11) 'd_Pyrite', (12) 'Goethite', (13) 'Goethite', 
            # (14) 'CO2(g)', (15) 'O2(g)', (16) 'Kin: Pyrite total', (17) 'Kin: Pyrite delta', 
            # (18) 'Kin: Goethite total', (19) 'Kin: Goethite delta', (20) 'Density_(kg/m^3)'
            d_cg_dt = (lambda_ / theta_g) * (H * pout[6] - C[14])
            d_cw_dt = -(lambda_ / theta_w) * (H * pout[6] - C[14])    
            ## Update concentrations
            C[14] = C[14] + d_cg_dt * dt_chem
            C[6] = pout[6] + d_cw_dt * dt_chem
            #print('pout:', pout[16],
            #      'pout*rhof*phi:', pout[16]*rho_f*phi,
            #      'C12:', C[12],
            #      'Diff C12/rhof/phi - pout:',(C[12]/rho_f/phi) - (pout[16]),
            #      'dPyr:', pout[17])
            ## IONS (indices need to be adapted to amount of ions and minerals)
            C[1] =  pout[0]
            C[2] =  pout[1]
            C[3] =  pout[3]
            C[4] =  pout[4]
            C[5] =  pout[5]
            #C[6] =  pout[6]
            C[7] =  pout[7]
            C[8] =  pout[8]
            C[9] =  pout[9]
            C[10] = pout[14]
            C[11] = pout[15]
            # MINERALS: back conversion from mol/kgw to mol/m^3
            C[12] = pout[16] * rho_f * phi
            C[13] = pout[18] * rho_f * phi
            
            #print(pout[16])
            #print(pout1[0])
            return C, P, T, rho_f, phi, Kx, Ky, Kz
    
        except:
            print('phreeqc error')
            print(phreeqc.get_error_string())
            #print(C)
            return C, P, T, rho_f, phi, Kx, Ky, Kz

    #---------------------------------------------------------------------------
    # INITIAL CHEMICAL EQUILIBRIUM / INFUX COMPOSITION
    # (the current tse.py does not call initialise_equilibrium / initialise_influx
    #  anymore, so they are applied here - same as in the migrated 1-D configs)
    #---------------------------------------------------------------------------
    for _i in range(nx):
        for _j in range(ny):
            for _k in range(nz):
                if L[_i, _j, _k] == 1:
                    C[_i, _j, _k, :] = initialise_equilibrium(
                        C[_i, _j, _k, :], C_sand, P[_i, _j, _k], T[_i, _j, _k],
                        rho_f[_i, _j, _k], phi[_i, _j, _k])
                elif L[_i, _j, _k] == 2:
                    C[_i, _j, _k, :] = initialise_equilibrium(
                        C[_i, _j, _k, :], C_inclusion, P[_i, _j, _k],
                        T[_i, _j, _k], rho_f[_i, _j, _k], phi[_i, _j, _k])
                elif L[_i, _j, _k] == 3:
                    C[_i, _j, _k, :] = initialise_influx(
                        C[_i, _j, _k, :], P[_i, _j, _k], T[_i, _j, _k])

#===============================================================================

# SOLVER PARAMETERS ############################################################
## TIME STEP SETTINGS ##########################################################
dt =     576.0  # initial time step size
# dt_max = 576 s keeps the time stepping of the archived run.  tse_old.py (the
# solver this config was run with) used inc_it = 1.00 in compute_C(), so the
# transport time step never grew and the delivered dt_max = 86400 s was inert.
# tse.py uses inc_it = 1.05, i.e. dt would grow 5 % per converged
# step up to dt_max - and since the gas-water O2 exchange (lambda_/theta_g) is
# integrated explicitly and PHREEQC is stepped with dt (DTCHEM = 1 s is only a
# minimum), that changes the result: with otherwise identical configs, dt_max =
# 86400 s vs 576 s differ by up to 35 % in S and 26 % in O2 after 24 h.
dt_max = 576.0 #86400

# max. iterations of the transport solver: tse_old.py used max_it = 20000
# internally, the current tse.py defaults to 100, which is tight against the
# rather loose CONV_C = 1e-5 of this config -> keep some headroom.
MAXIT_C = 2000

## IMPLICIT OR EXPLICIT SOLUTION (0 <= OMEGA <= 1)
OMEGA_C = 1
OMEGA_T = 1

## SUCCESSIVE OVER RELAXATION (0 <= SOR <= 2, SOR = 1 -> GAUSS-SEIDEL/JACOBI)
## SOR < 0: UNDERRELAXATION (SLOWER CONVERGENCE, MAY BE USEFUL FOR TRANSPORT)
## SOR > 1: OVERRELAXATION (FASTER CONVERGENCE)
## OPTIMISED SOR:
## SOR_P = 2/(1 + NUMPY.PI/NX) # NX SHOULD BE THE NUMBER OF ACTIVE FLOW ELEMENTS
SOR_P = 1.0 #2/(1+ numpy.pi/nx)
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
PLOTTIME = numpy.linspace(0,1,100) * t
