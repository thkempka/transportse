'''
    TRANSPORTSE configuration file

    Example 7: Battistel column - conservative O2 transport (15 h)
'''

import os
import sys
import numpy
from numba import njit

# base directory of this config file (data files are loaded relative to it,
# independent of the current working directory)
CFDIR = os.path.dirname(os.path.abspath(__file__))

# PARALLELISATION AND TUNING ###################################################
parallel_processes = 1   # number of threads
parallel_chemistry = 4   # number of threads
numba_fastmath = False   # turn off strict IEEE 754 compliance
numba_tl = 'omp'         # numba threading layer (tbb, omp, workqueue)

# MODEL DIMENSIONS AND SPATIAL DISCRETISATION ##################################
lx = 0.0151                     # model length in x-direction (m)
ly = 0.0151                     # model length in y-direction (m)
lz = 0.145                      # model length in z-direction (m)
nx = 1                          # number of nodes in x-direction (-)
ny = 1                          # number of nodes in y-direction (-)
nz = 145                         # number of nodes in z-direction (-)
dx = numpy.ones((nx)) * lx / nx  # nodal distance x-direction (m)
dy = numpy.ones((ny)) * ly / ny  # nodal distance y-direction (m)
dz = numpy.ones((nz)) * lz / nz  # nodal distance z-direction (m)

# PARAMETRISATION ##############################################################
g = 9.80665        # gravitational acceleration (m/s^2)
beta =  0          # model slope angle (°)

## SIMULATION TIME #############################################################
t = 60 * 60 * 15      # simulation time (s); 15 hours

## PROCESSES ###################################################################
FLUIDFLOW = False   # fluid flow (qx, qy must be given if set to False)
SPECTRANS = True   # species transport
HEATTRANS = False   # heat transport
FLUIDPROP = False  # fluid properties function
CHEMREACT = True  # chemical reactions
DYNAMICBC = False  # Use dynamic boundary conditions
MASSBALAN = False  # calculate mass balances
KINETICS  = True   # not used by tse.py (pyrite/goethite kinetics are defined
                   # in chemical_reactions() below)
DTCHEM    = 1      # chemical reactions time stepping
SIA = False        # semi-implicit algorithm
SIA_maxit = 10     # max. iterations of the semi-implicit algorithm (-)
SIA_conv = 1e-10   # convergence criterion of the semi-implicit algorithm,
                   # relative L2-norm change of the fluid density (-)
SIA_conv2 = 1e-10
C_IMBLFLO = False  # add the fluid mass balance error term to the species
                   # transport equation (implicit in the fluid flow solution)
T_IMBLFLO = False  # add the fluid mass balance error term to the heat
                   # transport equation (implicit in the fluid flow solution)

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
C_sand = numpy.zeros(MINERALS)      # initial mineral amounts in sand (mol/m^3)
C_inclusion = numpy.zeros(MINERALS) # initial mineral amounts in the pyrite
                                    # inclusion (mol/m^3)

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
lf_l   = numpy.zeros((nx, ny, nz))         # longitudinal dispersion length (m)
lf_t   = numpy.zeros((nx, ny, nz))         # transversal dispersion length (m)
L     = numpy.zeros((nx, ny, nz))         # lithology marker (-)
R     = numpy.ones((nx, ny, nz))          # retardation coefficient (-)
mass  = numpy.zeros((3))                  # mass balance matrix

## FLUID #######################################################################
D_fx  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
D_fy  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
D_fz  = numpy.zeros((nx, ny, nz, SPECIES))  # diffusion coefficient (m^2/s)
c_f   = numpy.zeros((nx, ny, nz))         # compressiblity (1/Pa)
cp_f  = numpy.zeros((nx, ny, nz))         # heat capacity (J/kg/K)
l_f   = numpy.zeros((nx, ny, nz)) # heat conductivity (W/m/K)
lt_l   = numpy.zeros((nx, ny, nz))         # longitudinal dispersion length (m), heat
lt_t   = numpy.zeros((nx, ny, nz))         # transversal dispersion length (m), heat
rho_f = numpy.zeros((nx, ny, nz))         # density (kg/m^3)
nue   = numpy.zeros((nx, ny, nz))         # viscosity (Pa*s)

## PRIMARY VARIABLES ###########################################################
P = numpy.zeros((nx, ny, nz))           # pore pressure (kg/m/s^2 = Pa)
C = numpy.zeros((nx, ny, nz, SPECIES))  # pore water concentrations (1)-(11)
                                        # (mol/kgw, tracer (0): arbitrary unit),
                                        # mineral amounts (12)-(13) (mol/m^3 bulk)
                                        # and (14) c_g: O2 in the gas phase
T = numpy.zeros((nx, ny, nz))           # temperature (°C)

## BOUNDARY CONDITIONS #########################################################
BP = numpy.zeros((nx, ny, nz))   # flow BC flag: 1 = pressure is prescribed
                                 # (Dirichlet), 0 = pressure is solved
BC = numpy.zeros((nx, ny, nz))   # species BC flag: 1 = concentration is prescribed
                                 # (Dirichlet, no PHREEQC call), 0 = transport is solved
BT = numpy.zeros((nx, ny, nz))   # thermal BC flag: 1 = temperature is prescribed
                                 # (Dirichlet), 0 = temperature is solved

## SINK AND SOURCE TERMS #######################################################
WP = numpy.zeros((nx, ny, nz))   # fluid flow (1/s)
WT = numpy.zeros((nx, ny, nz))   # heat flow  (W/m^3)
WC = numpy.zeros((nx, ny, nz, SPECIES))   # species flow (kg/m^3/s)

## FLOW VELOCITIES #############################################################
qx = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qy = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)
qz = numpy.zeros((nx, ny, nz))     # fluid flow velocity (m/s)


# PARAMETERS ###################################################################

Kx    += 5.0e-16         # permeability (m^2)
Ky    += 5.0e-16         # permeability (m^2)
Kz    += 5.0e-16         # permeability (m^2)
cp_r  += 1000.0         # heat capacity (J/kg/K)
cp_f  += 4150.0         # fluid heat capacity (J/kg/K)
c_f   += 4.5e-10        # fluid compressibility (1/Pa)
c_r   += 1e-10          # rock compressibility (1/Pa)
l_r   += 2.0            # rock heat conductivity (W/m/K)
lf_l   += 0.0                 # longitudinal dispersivity (m)
lf_t   += 0.0                 # transversal dispersivity (m)
l_f    += 0.6                 # fluid heat conductivity (W/m/K)
nue   += 1e-5           # fluid viscosity (Pa*s)
#R     += 0       
rho_f += 1000           # fluid density (kg/m^3)
rho_r += 2650           # rock matrix density (kg/m^3)
D_fx  += 2e-9           # diffusion coefficient in x-direction (m^2/s), all species
D_fy  = D_fx.copy()     # diffusion coefficient in y-direction (m^2/s), = D_fx
D_fz  = D_fx.copy()     # diffusion coefficient in z-direction (m^2/s), = D_fx
lt_l   += 0             # longitudinal heat dispersion length (m)
lt_t   += 0             # transversal heat dispersion length (m)

theta_w =  0.3637      # volumetric water content [-]
phi += theta_w         # porosity = water-filled pore space (fully water saturated)
theta_g = 0.0163       # volumetric gas content [-]
lambda_ = 1.27e-5      # Kinetic water-gas mass transfer coefficient [s-1]
H = 31.25              # Henry's coefficient for oxygen [-]

# GEOMETRY AND LITHOLOGY #######################################################

# Center and range in z-direction
center_z = nz // 2     # Index of the center in z-direction
dz_node_count = int(0.02 / dz[0])  # Number of nodes corresponding to 2 cm

# Range in z-direction (inclusive)
z_start = max(0, center_z - dz_node_count)    # first z-node of the pyrite
                                              # inclusion (slice start, incl.)
z_end = min(nz, center_z + dz_node_count + 1) # last z-node of the pyrite
                                              # inclusion + 1 (slice stop)

# Update L values
L[:, :, z_start:z_end] = 2  # Pyrite intrusion in column center (4 cm total)
L[L != 2] = 1               # Sand/porous medium
L[:,:,0] = 3                # Inlet

# BOUNDARY CONDITIONS ##########################################################

## FLUID FLOW
#BP[0,:,:] = 0        # Left boundary closed (no flow, flag 0 = pressure is solved)
#BP[-1,:,:] = 0       # Right boundary closed (no flow, flag 0 = pressure is solved)
#BP[:,:,0] = 0        # Bottom boundary closed (no flow, flag 0 = pressure is solved)
BP[:,:,-1] = 1        # Top boundary open (pressure prescribed, Dirichlet-BC)

# ## HEAT TRANSPORT
# BT[0,:,:] = 1        # Left boundary open (Dirichlet-BC)
# BT[-1,:,:] = 1       # Right boundary open (Dirichlet-BC)
# BT[:,0,:] = 1        # Bottom boundary open (Dirichlet-BC)
# BT[:,-1,:] = 1       # Top boundary open (Dirichlet-BC)

## SPECIES TRANSPORT
BC[L == 3] = 1        # Inlet: concentration prescribed (Dirichlet-BC), no PHREEQC call
#BC[0,:,:] = 0        # Left boundary closed (no flow, flag 0 = transport is solved)
#BC[-1,:,:] = 0       # Right boundary closed (no flow, flag 0 = transport is solved)
#BC[:,:,0] = 0        # Bottom boundary closed (no flow, flag 0 = transport is solved)
BC[:,:,-1] = 1        # Top boundary open (concentration prescribed, Dirichlet-BC)

# INITIAL CONDITIONS ###########################################################

## FLOW
# FLUIDFLOW = False -> the flow field is prescribed instead of being solved.
# The given seepage (pore water) velocity of 1.5 m/d becomes a Darcy flux when
# multiplied by the water-filled porosity; 86400 = s/d.
qz += phi * 1.5 / 86400   # Darcy flux in z-direction (m/s)

# PRESSURE
P[:,:,:] = 1e5                   # Initial air pressure


# # TEMPERATURE
T[:,:,:] = 20                    # Initial column temperature

C_sand[0] = 0.0                  # initial pyrite (mol/m^3) in sand
C_inclusion[0] = 0               # initial pyrite (mol/m^3) in inclusion
C_sand[1] = 0.0                  # initial goethite (mol/m^3) in sand
C_inclusion[1] = 0.0             # initial goethite (mol/m^3) in inclusion
C[:,:,:,14] = 0.000001           # initial O2 (mol/kgw) in column


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
    if sys.platform == 'linux': # workaround for the phreeqpy platform string
        os.system('sed -i s/linux2/linux/g '+str(phreeqc_mod).split("'")[3])

    phreeqc = phreeqc_mod.IPhreeqc()
    db_path = os.path.join(CFDIR, "phreeqc.dat")
    if not os.path.exists(db_path):
        # abort here instead of silently continuing without chemistry
        raise FileNotFoundError(
            f"The PHREEQC database file '{db_path}' was not found.")
    phreeqc.load_database(db_path)


    def initialise_equilibrium(C, C_mineral, P, T, rho_f, phi):
        # PHREEQC input string: initial pore water / pyrite equilibrium of one node
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
            pH         5.8
            Na         0.005
            Cl         0.005
            water       1 # kg
            O(0)       0.000006
            
        EQUILIBRIUM_PHASES 1
            # convert minerals from mol/m^3 to mol/kgw
            Pyrite    0   """+ str(C_mineral[0] / rho_f / phi) +"""

        END"""

        try:
            ec = phreeqc.run_string(input_string)  # PHREEQC run error code
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
        # PHREEQC input string: composition of the infiltrating water
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
            water       1 # kg
            O(0)       0.0005   ## O(0) instead of O2 
            
        END"""

        try:
            ec = phreeqc.run_string(input_string)  # PHREEQC run error code
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
            C[0] = 1e-3         # Tracer
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




    def chemical_reactions(C, P, T, rho_f, phi, Kx, Ky, Kz, dt_chem, i, j, k):
        #print(C[5])
        # PHREEQC input string: equilibrium + pyrite/goethite kinetics of one node
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
            #pe		   """+ str(C[2]) +"""
            press      """+ str(P / 101325) +"""
            temp       """+ str(T) +"""
            #density   """+ str(rho_f * 1e-3) +"""
            pH         """+ str(C[1]) +""" 
            Na         """+ str(C[3]) +"""
            Cl         """+ str(C[4]) +"""
            #S(6)       """+ str(C[8]) +"""
            S           """+ str(C[5]) +"""
            water       1 # kg
            O(0)       """+ str(C[6]) +"""
            Fe         """+ str(C[7]) +"""

#######
#Pyrite
#######
        # Williamson, M.A. and Rimstidt, J.D., 1994,
    KINETICS 1
    Pyrite
            -tol    1e-8
            -m     """+ str(C[12] / rho_f / phi) +"""
            -parms  0.3     0.67     .5      -0.11
    Goethite
            -tol    1e-8
            -m     """+ str(C[13] / rho_f / phi) +"""
            -parms  5.e-15
    -steps  """+ str(dt_chem) +""" seconds

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

    10  REM Dissolution in presence of DO
    20  if M <= 0 THEN GOTO 200
    30  if (SI("Pyrite") >= 0) THEN GOTO 200
#40  log_rate = -8.19 + PARM(3)*LM("O2") + PARM(4)*LM("H+")
    40  rate_battistel = 10^-8.19 * TOT("O2")^PARM(3)*TOT("H+")^PARM(4) * (2.8*M0) * ((M/M0)^PARM(2))*(1-10^SI("Pyrite"))
#50  log_area = PARM(1) + LOG10(M0) + PARM(2)*LOG10(M/M0)
#60  moles = 10^(log_area + log_rate) * TIME
    60  moles = rate_battistel * TIME
    200 SAVE moles
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

        END"""

        try:
            ec = phreeqc.run_string(input_string)  # PHREEQC run error code
            pout1 = phreeqc.get_selected_output_array()  # full selected output (unused)
            pout = numpy.array(phreeqc.get_selected_output_row(-1), dtype=float)
            #print(phreeqc.get_selected_output_row(0))
            #print(pout2[16])
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
            # (9) 'HCO3(mol/kgw)', (10) 'Pyrite', (11) 'd_Pyrite', (12) 'Goethite', 
            # (13) 'd_Goethite', (14) 'CO2(g)', (15) 'O2(g)', (16) 'Kin: Pyrite total', 
            # (17) 'Kin: Pyrite delta', (18) 'Kin: Goethite total', 
            # (19) 'Kin: Goethite delta', (20) 'Density_(kg/m^3)'
            
            # Oxygen partitioning between gas phase and dissolved Ox
            # Equations taken and modified from Battistel et al. 2019
            # Change in gas concentration
            d_cg_dt = (lambda_ / theta_g) * (H * pout[6] - C[14]) # pout[6] = c_w[t -1]
            # Change in water concentration
            d_cw_dt = -(lambda_ / theta_w) * (H * pout[6] - C[14]) # C[14] = c_g[t-1]
            # Update concentrations
            C[14] = C[14] + d_cg_dt * dt_chem
            C[6] = pout[6] + d_cw_dt * dt_chem   # C[6] = c_w[t]
            
            # IONS (indices need to be adapted to amount of ions and minerals)
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

            return C, P, T, rho_f, phi, Kx, Ky, Kz

        except:
            print('phreeqc error')
            print(phreeqc.get_error_string())
            #print(C)
            return C, P, T, rho_f, phi, Kx, Ky, Kz

    #---------------------------------------------------------------------------
    # INITIAL CHEMICAL EQUILIBRIUM / INFUX COMPOSITION
    #---------------------------------------------------------------------------
for _i in range(nx):      # loop index, x-direction
    for _j in range(ny):  # loop index, y-direction
        for _k in range(nz):  # loop index, z-direction
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
dt =     58.00  # initial time step size (s), value from Battistel et al. (2019)
# dt and dt_max must stay small: the gas-water O2 exchange (lambda_) and the
# mineral kinetics are integrated explicitly over dt, and tse.py calls PHREEQC
# once per dt (DTCHEM = 1 s is only a minimum), so a large dt integrates the
# kinetics too coarsely.
dt_max = 58.00 # maximum time step size (s)

# the tse.py default (100) is too small for CONV_C = 1e-12: dt would collapse
# by repeated time step bisection; this run needs ~50-190 iterations per step
MAXIT_C = 20000  # max. iterations of the species transport solver (-)

## IMPLICIT OR EXPLICIT SOLUTION (0 <= OMEGA <= 1)
OMEGA_C = 1      # species transport weighting factor (0 = explicit, 1 = implicit) (-)
OMEGA_T = 1      # heat transport weighting factor (0 = explicit, 1 = implicit) (-)

## SUCCESSIVE OVER RELAXATION (0 <= SOR <= 2, SOR = 1 -> GAUSS-SEIDEL/JACOBI)
## SOR < 0: UNDERRELAXATION (SLOWER CONVERGENCE, MAY BE USEFUL FOR TRANSPORT)
## SOR > 1: OVERRELAXATION (FASTER CONVERGENCE)
## OPTIMISED SOR:
## SOR_P = 2/(1 + NUMPY.PI/NX) # NX SHOULD BE THE NUMBER OF ACTIVE FLOW ELEMENTS
SOR_P = 1.0 #2/(1+ numpy.pi/nx)  # over-relaxation factor, flow solver (-)
SOR_C = 1.0                      # over-relaxation factor, species transport (-)
SOR_T = 1.0                      # over-relaxation factor, heat transport (-)

## CONVERGENCE CRITERION FOR L2-NORM ###########################################
CONV_P = 1e-12   # convergence criterion, pore pressure (relative L2-norm) (-)
CONV_C = 1e-12   # convergence criterion, species transport (relative L2-norm) (-)
CONV_T = 1e-5    # convergence criterion, heat transport (relative L2-norm) (-)

## SMOLARKIEWICZ ANTI-DIFFUSION STEP ###########################################
SMOLAR_C = False # Smolarkiewicz anti-diffusion correction, species transport
SMOLAR_T = False # Smolarkiewicz anti-diffusion correction, heat transport

# OUTPUT PARAMETERS ############################################################
PRINTOUT = 1        # console output interval (number of time steps) (-)
PLOTSAVE = True     # save results to pickle (*.p) files
PLOTTIME = numpy.linspace(0,1,50) * t  # times at which results are saved (s)
