# Licence: GNU General Public License, Version 3, 29 June 2007
# Copyright © 2026 GFZ Helmholtz Centre for Geosciences, Potsdam, Germany
# Citation: Kempka, T. (2020): Verification of a Python-based TRANSPORT
# Simulation Environment for density-driven fluid flow and coupled transport of
# heat and chemical species. Advances in Geosciences, 54, 67-77.
# https://doi.org/10.5194/adgeo-54-67-2020

'''
    Program to convert TRANSPORTSE simulation output files to Paraview data

    Usage (run in the directory containing the simulation output):
        python tse2pvd.py [output file prefix] [-n NAMES]

    The output file prefix corresponds to the config file name and defaults
    to 'config', i.e. config-#####.p files are converted to sim#####.vts
    files plus a sim.pvd collection and massbal.csv.

    SPECIES NAMES
    The layers of C are exported under their physical names (pH, pe, the
    dissolved species, the saturation indices and the minerals) instead of
    Species_0 ... Species_n. Names are looked up in NAME_TABLES, keyed by the
    number of C layers, or taken from -n/--names (comma separated list or
    @file with one name per line). If the number of layers is not covered by a
    table, generic Species_n names are used and a warning is printed.
    NOTE: the previous heuristic "layer 0 = Tracer, last layer = pH" is gone,
    because it mislabelled the Battistel runs: layer 1 holds the pH, while the
    last layer (14) holds the gas phase oxygen pool c_g, which used to be
    exported as "pH".

    O2 EXCHANGE PARAMETERS
    theta_w, theta_g, lambda_ and H are scalar config values and are NOT part
    of the *.p files. They are read (without importing, i.e. without running
    PHREEQC again) from <prefix>.py if that file sits in the current directory
    and are exported as constant fields, together with the kinetic water-gas
    O2 exchange rate of the water phase:
        O2_exch_w = -(lambda_ / theta_w) * (H * O2_aq - c_gas)   [mol/kgw/s]
    (the gas phase term uses +lambda_ / theta_g instead of -lambda_ / theta_w).
'''

import argparse
import ast
import glob
import os
import sys

import joblib
import numpy
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup

# SPECIES LAYER NAMES ##########################################################
# (name, unit, description) per layer of C, keyed by the number of C layers
# (IONS + MINERALS in the config). The PHREEQC coupling of all Battistel
# configs of this course (1D-Exp1-3, 2D-Exp1) writes IONS = 12 + MINERALS = 3.
NAME_TABLES = {
    15: [
        ('Tracer',   '-',        'conservative tracer'),
        ('pH',       '-',        'pore water pH (PHREEQC)'),
        ('pe',       '-',        'pore water redox potential (PHREEQC)'),
        ('Na',       'mol/kgw',  'total sodium'),
        ('Cl',       'mol/kgw',  'total chloride'),
        ('S_total',  'mol/kgw',  'total sulfur (pyrite dissolution product)'),
        ('O2_aq',    'mol/kgw',  'dissolved oxygen, PHREEQC total O(0)'),
        ('Fe',       'mol/kgw',  'total iron'),
        ('SO4',      'mol/kgw',  'SO4-2 molality'),
        ('HCO3',     'mol/kgw',  'HCO3- molality'),
        ('SI_CO2_g', 'log10 SI', 'saturation index of CO2(g)'),
        ('SI_O2_g',  'log10 SI', 'saturation index of O2(g)'),
        ('Pyrite',   'mol/m3',   'pyrite amount per bulk volume (kinetic)'),
        ('Goethite', 'mol/m3',   'goethite amount per bulk volume (kinetic)'),
        ('c_gas',    'mol/kgw',  'O2 in the gas phase, c_gas = H * O2_aq at '
                                 'equilibrium'),
    ],
}

# scalar config parameters of the kinetic water-gas O2 exchange:
# config variable -> (exported field name, unit, description)
PARAM_FIELDS = {
    'theta_w': ('theta_w',     '-',   'volumetric water content'),
    'theta_g': ('theta_g',     '-',   'volumetric gas content'),
    'lambda_': ('lambda_exch', '1/s', 'kinetic water-gas mass transfer '
                                      'coefficient'),
    'H':       ('Henry_O2',    '-',   "Henry's coefficient of O2, c_gas = H * "
                                      'O2_aq at equilibrium'),
}

# command line arguments (prefix kept as positional argument, default 'config')
argp = argparse.ArgumentParser(
    description='Convert TRANSPORTSE output files to Paraview data.')
argp.add_argument('prefix', nargs='?', default='config',
                  help='output file prefix = config file name (default: '
                       'config)')
argp.add_argument('-n', '--names', default='',
                  help='species layer names, comma separated or @file with one '
                       'name per line (overrides NAME_TABLES)')
args = argp.parse_args()
prefix = args.prefix


def species_names(nspecies, override):
    '''
        physical names of the C layers, returned as list of
        (name, unit, description) tuples
    '''
    if override:
        if override.startswith('@'):
            with open(override[1:]) as nfile:
                names = [line.strip() for line in nfile if line.strip()]
        else:
            names = [entry.strip() for entry in override.split(',')
                     if entry.strip()]
        if len(names) != nspecies:
            sys.exit('-n/--names provides %d names, but the output has %d '
                     'species layers.' % (len(names), nspecies))
        return [(name, '', '') for name in names]

    if nspecies in NAME_TABLES:
        return NAME_TABLES[nspecies]

    print('WARNING: no species name table for %d layers, using generic '
          'Species_n names (pass -n NAME_0,NAME_1,... or add a table to '
          'NAME_TABLES)' % nspecies)
    return [('Species_%d' % n, '', '') for n in range(nspecies)]


def config_scalars(prefix):
    '''
        scalar O2 exchange parameters of <prefix>.py, read with ast (importing
        the config would set up PHREEQC and re-initialise the whole model)
    '''
    cfile = prefix + '.py'
    if not os.path.exists(cfile):
        print('note: %s not found - the scalar O2 exchange parameters '
              '(theta_w, theta_g, lambda_, H) are not exported' % cfile)
        return {}

    try:
        with open(cfile) as source:
            tree = ast.parse(source.read())
    except SyntaxError:
        print('note: %s could not be parsed - O2 exchange parameters are not '
              'exported' % cfile)
        return {}

    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in PARAM_FIELDS and
                isinstance(node.value, ast.Constant) and
                isinstance(node.value.value, (int, float))):
            values[node.targets[0].id] = float(node.value.value)
    return values


# collect output files of the selected run only (prefix-#####.p)
pfiles = sorted(glob.glob(prefix + '-[0-9]*.p'))
if not pfiles:
    sys.exit('No TRANSPORTSE output files "%s-#####.p" found in the current '
             'directory.' % prefix)

vtkg = VtkGroup("sim")

# scalar O2 exchange parameters (empty if <prefix>.py is not available)
cfg_params = config_scalars(prefix)

names = None        # species layer names, resolved from the first output file
layer = {}          # name -> C layer index, resolved together with names
massbal = None

for num, filename in enumerate(pfiles):
    with open(filename, 'rb') as pfile:
        data = joblib.load(pfile)

    # current file format (34 elements, incl. WC and e)
    if len(data) == 34:
        lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt, P, C, \
            T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f, nue, cp_f, l_f, \
            BP, BC, BT, L, mass, WP, WT, WC, e = data
    # previous file format (32 elements, no WC and e)
    elif len(data) == 32:
        lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt, P, C, \
            T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f, nue, cp_f, l_f, \
            BP, BC, BT, L, mass, WP, WT = data
        e = numpy.zeros((nx, ny, nz))  # flow error not stored in old files
        print('note: %s uses the previous file format (WC, e not stored)'
              % filename)
    else:
        print('skipping %s: unsupported file format (%d elements)'
              % (filename, len(data)))
        continue

    # resolve the layer names once per run (all files share the C layout)
    if names is None:
        names = species_names(numpy.shape(C)[-1], args.names)
        layer = dict((name, n) for n, (name, unit, desc) in enumerate(names))

    # mass balance table: first column is the simulation time (s)
    row = numpy.hstack((numpy.array([t_sim]), numpy.atleast_1d(mass)))
    massbal = row.copy() if massbal is None else numpy.vstack((massbal, row))

    print(filename)

    # geometric assignments only once
    if num == 0:
        # Dimensions
        if numpy.shape(dx) == ():
            dx = numpy.ones(nx)*lx/nx
            dy = numpy.ones(ny)*ly/ny
            dz = numpy.ones(nz)*lz/nz

        # Coordinates
        X = numpy.cumsum(numpy.append([0], dx))
        Y = numpy.cumsum(numpy.append([0], dy))
        Z = numpy.cumsum(numpy.append([0], dz))

        x = numpy.zeros((nx + 1, ny + 1, nz + 1))
        y = numpy.zeros((nx + 1, ny + 1, nz + 1))
        z = numpy.zeros((nx + 1, ny + 1, nz + 1))

        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x[i, j, k] = X[i]
                    y[i, j, k] = Y[j]
                    z[i, j, k] = Z[k]

    # NLTE output: export fluid phase temperature (layer 0) only
    if numpy.ndim(T) == 4:
        T = T[:, :, :, 0]

    # export all transported species under their physical names
    cellData = {"Temperature": T,
                "Pressure": P,
                "Porosity": phi,
                "Kx": Kx,
                "Ky": Ky,
                "Kz": Kz,
                "Density": rho_f,
                "Viscosity": nue,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "BP": BP,
                "BC": BC,
                "BT": BT,
                "L": L,
                "WP": WP,
                "WT": WT,
                "e": e,
                }

    # NOTE: WC (species source term) carries a species dimension and is
    # therefore not exported as a single cell field

    for n, (name, unit, desc) in enumerate(names):
        cellData[name] = C[:, :, :, n].copy()

    # scalar O2 exchange parameters as constant fields ...
    for key, value in cfg_params.items():
        field, unit, desc = PARAM_FIELDS[key]
        cellData[field] = numpy.full(numpy.shape(P), value)

    # ... and the water phase exchange rate they produce
    if {'lambda_', 'theta_w', 'H'} <= set(cfg_params) and \
       'O2_aq' in layer and 'c_gas' in layer:
        cellData['O2_exch_w'] = \
            -(cfg_params['lambda_'] / cfg_params['theta_w']) * \
            (cfg_params['H'] * C[:, :, :, layer['O2_aq']] -
             C[:, :, :, layer['c_gas']])

    vtsfile = "sim" + str(num).zfill(5)

    gridToVTK(vtsfile, x, y, z, cellData=cellData)

    # add vts to pvd file
    vtkg.addFile(filepath=vtsfile + ".vts", sim_time=t_sim)

if massbal is not None:
    numpy.savetxt('massbal.csv', massbal, delimiter=',')
else:
    sys.exit('No readable TRANSPORTSE output files found - nothing converted.')

# save pvd file
vtkg.save()

# list the exported species fields, their units and the exchange parameters
print('\nexported species fields (%d layers of C):' % len(names))
for n, (name, unit, desc) in enumerate(names):
    print('  C[..., %2d]  %-10s %-10s %s' %
          (n, name, '(%s)' % unit if unit else '', desc))

if cfg_params:
    print('\nO2 exchange parameters read from %s.py:' % prefix)
    for key, value in cfg_params.items():
        field, unit, desc = PARAM_FIELDS[key]
        print('  %-12s = %-10.4g %-10s %s' %
              (field, value, '(%s)' % unit if unit else '', desc))
    if {'lambda_', 'theta_w', 'H'} <= set(cfg_params) and \
       'O2_aq' in layer and 'c_gas' in layer:
        print('  O2_exch_w    : kinetic O2 exchange rate of the water phase '
              '(mol/kgw/s)')
