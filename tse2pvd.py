# Licence: GNU General Public License, Version 3, 29 June 2007
# Copyright © 2026 GFZ Helmholtz Centre for Geosciences, Potsdam, Germany
# Citation: Kempka, T. (2020): Verification of a Python-based TRANSPORT
# Simulation Environment for density-driven fluid flow and coupled transport of
# heat and chemical species. Advances in Geosciences, 54, 67-77.
# https://doi.org/10.5194/adgeo-54-67-2020

'''
    Program to convert TRANSPORTSE simulation output files to Paraview data

    Usage (run in the directory containing the simulation output):
        python tse2pvd.py [output file prefix]

    The output file prefix corresponds to the config file name and defaults
    to 'config', i.e. config-#####.p files are converted to sim#####.vts
    files plus a sim.pvd collection and massbal.csv.
'''

import glob
import sys

import joblib
import numpy
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup

# output file prefix (optional first command line argument, default 'config')
prefix = sys.argv[1] if len(sys.argv) > 1 else 'config'

# collect output files of the selected run only (prefix-#####.p)
pfiles = sorted(glob.glob(prefix + '-[0-9]*.p'))
if not pfiles:
    sys.exit('No TRANSPORTSE output files "%s-#####.p" found in the current '
             'directory.' % prefix)

vtkg = VtkGroup("sim")

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

    # export all transported species (species 0 keeps its legacy name)
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

    for n in range(numpy.shape(C)[-1]):
        cellData["Tracer" if n == 0 else "Species_%d" % n] = \
            C[:, :, :, n].copy()

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
